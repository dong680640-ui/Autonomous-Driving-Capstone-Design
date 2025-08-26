########################
# For Mission 2 (Real) #
########################

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from interfaces_pkg.msg import CarData, LaneData, SegmentGroup, BoolMultiArray, MotionCommand
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import numpy as np
import cv_bridge

import time

## <Parameter> #####################################################################################

# 구독 토픽 이름
SUB_TOPIC_CAR = "car_data" 
SUB_TOPIC_LANE = "lane_data"
SUB_TOPIC_TRAFFIC = "traffic_data"
SUB_TOPIC_LIDAR = "lidar_data"
SUB_TOPIC_YOLO = "segmented_data"
SUB_TOPIC_LIDAR_DISTANCE = "lidar_processed"
SUB_TOPIC_YOLO_FOR_CROSSWALK = "crosswalk_data"

# 발행 토픽 이름
PUB_TOPIC_NAME = "command_data"

# 연산 주기 설정
PERIOD = 0.1

# 차량 범퍼 위치
BUMPER_POSITION = [300, 462]

# 보정 상수
K_Stanley_1 = 0.35
K_Angle_1 = 0.01

K_Stanley_2 = 0.35
K_Angle_2 = 0.01

# 디버그 모드
DEBUG = False

# 기준 속도
SPEED = 100

#2차선에서 1차선으로
SPEED_LANE1_CHANGE = 100
#1차선에서 2차선으로
SPEED_LANE2_CHANGE = 180

# 물체 접근시 후진 발동 거리
BACK_UP_DISTANCE = 0.3

# Lane Change Mode 유지 사이클
MAINTAIN_LANE_CHANGE = 70

Y_DISTANCE_THRESHOLD_MIN = 250 # 충분히 멀어졌다고 간주할 y 값 (측정 후 조정정)
Y_DISTANCE_THRESHOLD_MAX = 260
Y_DECREASE_COUNT = 20          # y 좌표가 지속적으로 감소한 횟수

######################################################################################################

class motion_planner(Node):
    def __init__(self):
        super().__init__("motion_planner")

        self.qos_sub = QoSProfile( # Subscriber QOS 설정
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
                )
        
        self.qos_pub = QoSProfile( # Publisher QOS 설정
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
                )
        
        # Subsciption 선언
        self.sub_car = self.create_subscription(CarData, SUB_TOPIC_CAR, self.update_car_data, self.qos_sub)
        self.sub_lane = self.create_subscription(LaneData, SUB_TOPIC_LANE, self.update_lane_data, self.qos_sub)
        self.sub_traffic = self.create_subscription(String, SUB_TOPIC_TRAFFIC, self.update_traffic_data, self.qos_sub)
        self.sub_lidar = self.create_subscription(BoolMultiArray, SUB_TOPIC_LIDAR, self.update_lidar_data, self.qos_sub)
        self.sub_yolo = self.create_subscription(SegmentGroup, SUB_TOPIC_YOLO, self.update_yolo_data, self.qos_sub)
        self.sub_lidar_distance = self.create_subscription(LaserScan, SUB_TOPIC_LIDAR_DISTANCE, self.update_lidar_distance, self.qos_sub)
        self.sub_yolo_for_crosswalk = self.create_subscription(SegmentGroup, SUB_TOPIC_YOLO_FOR_CROSSWALK, self.update_yolo_data_cross, self.qos_sub)

        # Publisher 선언
        self.command_publisher = self.create_publisher(MotionCommand, PUB_TOPIC_NAME, self.qos_pub)

        # CV Bridge Object 선언    
        self.bridge = cv_bridge.CvBridge()

        # 데이터 저장 레지스터 선언
        self.car_data = None
        self.lane_data = None
        self.traffic_data = None
        self.lidar_data = None
        self.yolo_data = None
        self.lidar_distance_data = None
        self.yolo_data_cross = None

        # State 저장 레지스터 선언 (0, 1, 2, 3) | 0은 초기화 상태를 의미함
        self.state = 0

        # Lane 위치 저장 레지스터 선언 (1, 2)
        self.lane_state = None

        # Timer 선언
        self.timer = self.create_timer(PERIOD, self.motion_decision_callback)

        # 전송 데이터 기억
        self.steer_angle_reg = 0 # send_command 함수에서 사용
        self.traffic_reg = [] # update_traffic_data 함수에서 사용
        self.crosswalk_reg = [] # update_yolo_data 함수에서 사용
        self.car1_y_history=[] # 전방 car1의 y 좌표 저장용 리스트트

        # 카운트 저장소 선언
        self.cnt = 0
        self.cnt_for_stop = 0

        # y 좌표 기록 주기를 제어하기 위한 카운터
        self.y_record_cnt = 0
        
##### <변수 업데이트를 위한 함수 선언> #####################################################################

    def update_car_data(self, msg):
        self.car_data = msg # car position

        # 차량 존재 여부 업데이트
        self.car_1, self.car_2, self.car_location_data = self.extract_car_data()

#######################################################################

    def update_lane_data(self, msg):
        self.lane_data = msg # angle, center position

#######################################################################

    def update_traffic_data(self, msg):
        self.traffic_data = msg # R, Y, G, N

        # Traffic Light 데이터 기록
        self.traffic_reg.append(self.traffic_data.data)
        
        # 4개로 Register 크기 제한
        if len(self.traffic_reg) > 4:
            self.traffic_reg.pop(0)

#######################################################################

    def update_lidar_data(self, msg):
        self.lidar_data = msg # T/F

#######################################################################

    def update_yolo_data(self, msg):
        self.yolo_data = msg # segmentation data

#######################################################################

    def update_lidar_distance(self, msg):
        self.lidar_distance_data = msg # lidar distance data

#######################################################################
   
    def update_yolo_data_cross(self, msg):
        self.yolo_data_cross = msg

        # Croswalk 감지 여부 기록
        self.crosswalk_reg.append(len(self.yolo_data_cross.crosswalk) > 0)
        
        # 15개로 Register 크기 제한
        if len(self.crosswalk_reg) > 15:
            self.crosswalk_reg.pop(0)

######################################################################################################


    # 제어 명령 전송 함수
    def send_command(self, steer_angle:int, left_speed:int, right_speed:int):
        msg = MotionCommand()

        # 조향각 데이터가 비어있는 경우
        if steer_angle == None:
            # 이전 조향각 반영
            steer_angle = self.steer_angle_reg

        else:
            # 조향각 업데이트
            self.steer_angle_reg = steer_angle

        msg.steering = steer_angle
        msg.left_speed = left_speed
        msg.right_speed = right_speed

        if DEBUG == True:
            msg = MotionCommand()

        self.command_publisher.publish(msg)


    # Stanley Method 기반 조향각 계산 함수
    def calculate_steering_angle(self, target_point:list, car_center_point:list, path_slope:float, vehicle_speed:int, k_angle, k_stanley):
            # Heading Error
            heading_error = path_slope * k_angle

            # 횡방향 오차 계산
            lateral_error = target_point[0] - car_center_point[0]

            # 조향각 계산
            steering_angle = heading_error + np.arctan(k_stanley * lateral_error / (vehicle_speed + 1e-6))*(180/np.pi)
            return int(np.clip(steering_angle, -30, 30)) # 각도 제한 (-30~30)


    # 차량 존재 여부 확인 함수
    def extract_car_data(self):
            car_1 = False
            car_2 = False
            car_location_data = []
            
            # Lane 1
            try:
                lane_1 = np.array(self.yolo_data.lane_1).reshape(-1, 2)

                y_min = np.min(lane_1[:, 1])
                temp = lane_1[(y_min <= lane_1[:, 1]) & (lane_1[:, 1] <= y_min + 10)]

                # 도로 상단부 좌우측 좌표값 추출
                x_1_min, y_1_min = temp[np.argmin(temp[:, 0])]
                x_1_max, y_1_max = temp[np.argmax(temp[:, 0])]

                # 회전 구간에서 도로가 완전하게 보이지 않는 경우를 제거하기 위한 조건
                if abs(x_1_min - x_1_max) > 50:
                    x_1 = (x_1_min + x_1_max)/2

                else:
                    x_1 = 0                   
  
            # Lane 1이 보이지 않을 경우
            except:
                x_1 = 0
                y_1_min = 0

            # Lane 2 
            try:
                lane_2 = np.array(self.yolo_data.lane_2).reshape(-1, 2)

                y_min = np.min(lane_2[:, 1])
                temp = lane_2[(y_min <= lane_2[:, 1]) & (lane_2[:, 1] <= y_min + 10)]

                # 도로 상단부 좌우측 좌표값 추출
                x_2_min, y_2_min = temp[np.argmin(temp[:, 0])]
                x_2_max, y_2_max = temp[np.argmax(temp[:, 0])]

                # 회전 구간에서 도로가 완전하게 보이지 않는 경우를 제거하기 위한 조건
                if abs(x_2_min - x_2_max) > 50:
                    x_2 = (x_2_min + x_2_max)/2

                else:
                    x_2 = 640

            # Lane 2가 보이지 않을 경우
            except:
                x_2 = 640
                y_2_min = 0

            
            for x, y in list(zip(self.car_data.x, self.car_data.y)):
                # 거리 비교
                d1 = abs(x - x_1)                
                d2 = abs(x - x_2)

                # 2차선에 위치한 경우
                if d1 > d2:

                    # 차량이 인식된 차선 면적보다 위에 있는 경우
                    if y <= y_2_min:
                        car_2 = True
                        car_location_data.append(2)

                    # 차량이 인식된 차선 면적보다 아래에 있는 경우 + 차선이 인식되지 않은 경우
                    else:
                        # 차량이 1차선으로 판명되는 경우
                        if x < 320/2:
                            car_1 = True
                            car_location_data.append(1)   
                        # 차량이 2차선으로 판명되는 경우
                        else:
                            car_2 = True
                            car_location_data.append(2)                            

        
                # 1차선에 위치한 경우
                elif d1 < d2:

                    # 차량이 인식된 차선 면적보다 위에 있는 경우
                    if y <= y_1_min:
                        car_1 = True
                        car_location_data.append(1)
                    
                    # 차량이 인식된 차선 면적보다 아래에 있는 경우  + 차선이 인식되지 않은 경우
                    else:
                        # 차량이 1차선으로 판명되는 경우
                        if x < 320/2:
                            car_1 = True
                            car_location_data.append(1)
                        # 차량이 2차선으로 판명되는 경우
                        else:
                            car_2 = True
                            car_location_data.append(2)   

            self.get_logger().info(f"detect: {car_1} {car_2}")

            return car_1, car_2, car_location_data


    # 판단 로직 작성부
    def motion_decision_callback(self):
        try:
            # State 0 : init_mode
            if self.state == 0:
                self.state = self.init_mode()

            # State 1 : drive_mode
            elif self.state == 1:
                self.state = self.drive_mode()

            # State 2 : lane_change_mode
            elif self.state == 2:
                self.state = self.lane_change_mode()

            # State 3 : search_mode
            elif self.state == 3:
                self.state = self.search_mode()

            # State 4 : stop_mode
            elif self.state == 4:
                self.state = self.stop_mode()

        except Exception as e:
            self.get_logger().warn(f"{e}")


### <State 정의 함수> ####################################################################

    # State 0
    def init_mode(self) -> int:      
        # 데이터가 전부 수신되었을 경우, 처리 시작

        if self.car_data != None and self.lane_data != None and self.traffic_data != None and self.lidar_data != None and self.yolo_data != None and self.yolo_data_cross != None:

            # 차량 위치 결정
            d_1 = abs(self.lane_data.lane1_x - BUMPER_POSITION[0])
            d_2 = abs(self.lane_data.lane2_x - BUMPER_POSITION[0])

            # 2차선 위치 조건
            if(d_1 > d_2):
                self.lane_state = 2
        
            # 1차선 위치 조건
            else:
                self.lane_state = 1

            # 0의 지령값 설정
            self.send_command(steer_angle = 0, left_speed = 0, right_speed = 0)

            # 주행 모드로 반환
            return 1
        

        # 데이터가 전부 수신되지 않았을 경우, 오류 전송
        else:
            self.get_logger().warn("data is not yet accepted")
            self.get_logger().warn(f"{self.car_data != None}, {self.lane_data != None}, {self.traffic_data != None}, {self.lidar_data != None}, {self.yolo_data != None}, {self.yolo_data_cross != None}")
            return 0

########################################################################################

    # State 1
    def drive_mode(self) -> int:
        self.get_logger().info(f"drive_mode : {self.lane_state}")                               

        # 신호등이 빨간색인 경우
        if self.traffic_data.data == "R":
            # 횡단보도가 감지된 경우
            if len(self.yolo_data_cross.crosswalk) != 0:
                # 1차선에서 횡단보도의 위치가 420 이상 및 빨간색이 3번 연속 검출된 경우
                if self.lane_state == 1 and np.array(self.yolo_data_cross.crosswalk).reshape(-1, 2)[:, 1].max() > 420 and self.traffic_reg.count("R") >= 3:
                    # 정지
                    return 4
    
                # 2차선에서 횡단보도의 위치가 430 이상 및 빨간색이 3번 연속 검출된 경우
                elif self.lane_state == 2 and np.array(self.yolo_data_cross.crosswalk).reshape(-1, 2)[:, 1].max() > 430 and self.traffic_reg.count("R") >= 3:
                    # 정지
                    return 4
                
            # 그외의 경우
            else:
                pass


        # 1차선에 차량이 존재하고, 본인이 1차선에 있는 경우
        if self.car_1 and self.lane_state == 1:
            # 2차선에 차량이 없고, LIDAR로 2차선 장애물이 감지되지 않은 경우
            if self.car_2 == False and self.lidar_data.data[1] == False:
                    # 차선 변경
                    return 2                 

         # 본인이 2차선에 있는 경우, 2차선에 차량이 존재
        if (len(self.yolo_data.car) != 0) and (not self.car_1) and (self.lane_state == 2):
            # 2차선 차량이 감지되고, LIDAR로 1차선 장애물이 감지되지 않은 경우
            if self.lidar_data.data[0] == False:
                    # 차선 변경
                    return 2    


        # 1차선에 있을 경우, 속도 및 조향 설정
        if self.lane_state == 1:
            steer_angle = self.calculate_steering_angle(target_point = [self.lane_data.lane1_x, self.lane_data.lane1_y],
                                                        car_center_point = BUMPER_POSITION, 
                                                        vehicle_speed = SPEED,
                                                        path_slope = self.lane_data.slope1,
                                                        k_angle=K_Angle_1,
                                                        k_stanley=K_Stanley_1)
            # Differential 구현
            self.send_command(steer_angle = steer_angle, left_speed = SPEED, right_speed = SPEED) 

        # 2차선에 있을 경우, 속도 및 조향 설정
        elif self.lane_state == 2:
            steer_angle = self.calculate_steering_angle(target_point = [self.lane_data.lane2_x, self.lane_data.lane2_y],
                                                        car_center_point = BUMPER_POSITION, 
                                                        vehicle_speed = SPEED,
                                                        path_slope = self.lane_data.slope2, 
                                                        k_angle=K_Angle_2,
                                                        k_stanley=K_Stanley_2)
            # Differential 구현
            self.send_command(steer_angle = steer_angle, left_speed = SPEED, right_speed = SPEED) 

        # 계속 주행
        return 1

########################################################################################
    # State 2
    def lane_change_mode(self) -> int:
        # 2차선에 위치해 있을 경우
        if self.lane_state == 2:
            self.get_logger().info(f"lane_change_mode : 2 -> 1")
            steer_angle = self.calculate_steering_angle(target_point = [self.lane_data.lane1_x, self.lane_data.lane1_y],
                                                        car_center_point = BUMPER_POSITION, 
                                                        vehicle_speed = SPEED_LANE1_CHANGE,
                                                        path_slope=self.lane_data.slope1,
                                                        k_angle=K_Angle_1,
                                                        k_stanley=K_Stanley_1)
            
            # 조향 각도 제한 [좌:-30, 우:30]
            self.send_command(steer_angle = int(np.clip(steer_angle*3, -30, 30)), left_speed = SPEED_LANE1_CHANGE, right_speed = SPEED_LANE1_CHANGE) 
            
            # Search Mode로의 변동 조건
            if self.cnt > MAINTAIN_LANE_CHANGE:
                self.lane_state = 1
                self.cnt = 0
                return 3
            

        # 1차선에 위치해 있을 경우
        if self.lane_state == 1:
            self.get_logger().info(f"lane_change_mode : 1 -> 2")
            steer_angle = self.calculate_steering_angle(target_point = [self.lane_data.lane2_x, self.lane_data.lane2_y],
                                                        car_center_point = BUMPER_POSITION, 
                                                        vehicle_speed = SPEED_LANE2_CHANGE,
                                                        path_slope=self.lane_data.slope2,
                                                        k_angle=K_Angle_2,
                                                        k_stanley=K_Stanley_2)

            # 조향 각도 제한 [좌:-30, 우:30]
            self.send_command(steer_angle = int(np.clip(steer_angle*3, -30, 30)), left_speed = SPEED_LANE2_CHANGE, right_speed = SPEED_LANE2_CHANGE) 
            
            # Driving Mode로의 변동 조건
            if self.cnt > MAINTAIN_LANE_CHANGE:
                self.lane_state = 2
                self.cnt = 0
                return 1

        # 카운트 증가
        self.cnt += 1

        # 기존 상태 유지
        return 2
    
########################################################################################

    # State 3
    def search_mode(self) -> int:
        self.get_logger().info(f"search_mode : {self.lane_state}")

        # 정지 명령 
        self.send_command(steer_angle=0, left_speed=0, right_speed=0)       

        if self.car_data is not None and self.car_1 == True:
            # 1차선 차량의 인덱스 구하기
            car1_index = self.car_location_data.index(1)
            car1_y = self.car_data.y[car1_index]

            # y 좌표 히스토리 저장
            self.y_record_cnt += 1
            self.car1_y_history.append(car1_y)

            if len(self.car1_y_history) > Y_DECREASE_COUNT:
                self.car1_y_history.pop(0)

            # y 좌표가 감소 추세인지 확인
            if len(self.car1_y_history) == Y_DECREASE_COUNT:
                car_hist_post_process = []
                self.get_logger().info(f"debug: {self.car1_y_history}")

                for k in range(0, Y_DECREASE_COUNT - 1, 4):
                    car_hist_post_process.append(sum(self.car1_y_history[k:k+4])/4)

                decreasing = all(car_hist_post_process[i] > car_hist_post_process[i + 1] for i in range(len(car_hist_post_process) - 1))

                if decreasing and (Y_DISTANCE_THRESHOLD_MIN < self.car1_y_history[-1] < Y_DISTANCE_THRESHOLD_MAX):
                    # y 좌표 히스토리 초기화
                    self.car1_y_history = []  

                    # drive_mode 진입
                    return 1  

        # 차량 위치 결정
        d_1 = abs(self.lane_data.lane1_x - BUMPER_POSITION[0])
        d_2 = abs(self.lane_data.lane2_x - BUMPER_POSITION[0])

        # 2차선 위치 조건 (재위치)
        if (d_1 > d_2) and self.car_1 == False:
            
            # 차선 변경
            self.lane_state = 2

            # 메모리 초기화
            self.car1_y_history = []

            # 5초 지연
            time.sleep(5)

            # 주행 모드
            return 1

        # 현 상태 유지
        return 3

########################################################################################

    # State 4
    def stop_mode(self) -> int:
        self.get_logger().info(f"stop_mode : {self.lane_state}")
        self.send_command(steer_angle = 0, left_speed = 0, right_speed = 0)

        # 신호등이 빨간색이 아닌 경우
        if self.traffic_data.data != "R":
            
            # 카운트 초기화
            self.cnt_for_stop = 0

            # 주행
            return 1   

        # 카운트 증가
        self.cnt_for_stop += 1       

        # 상태 유지
        return 4

########################################################################################

def main():
    rclpy.init()
    motion_planner_node = motion_planner()
    rclpy.spin(motion_planner_node)

    motion_planner_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()