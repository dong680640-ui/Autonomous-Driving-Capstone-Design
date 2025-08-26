#############
# Prototype #
#############

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from interfaces_pkg.msg import CarData, LaneData, SegmentGroup, BoolMultiArray
from std_msgs.msg import String, Bool, Int8MultiArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import numpy as np

## <Parameter> #####################################################################################

# 구독 토픽 이름
SUB_TOPIC_CAR = "car_data" 
SUB_TOPIC_LANE = "lane_data"
SUB_TOPIC_TRAFFIC = "traffic_data"
SUB_TOPIC_LIDAR = "lidar_data"
SUB_TOPIC_YOLO = "segmented_data"
SUB_TOPIC_DEPTH = "depth_data"

# 발행 토픽 이름
PUB_TOPIC_NAME = "command_data"

# 연산 주기 설정
PERIOD = 0.1

# 차량 범퍼 위치
BUMPER_POSITION = [320, 462]

# 보정 상수
K_Stanley_1 = 0.32
K_Angle_1 = 0.17 - 0.03

K_Stanley_2 = 0.3
K_Angle_2 = 0.113 - 0.01

K_Stanley_1_Turn = K_Stanley_1 * 1.3
K_Stanley_2_Turn = K_Stanley_2 * 1
K_Angle_1_Turn = K_Angle_1 * 1
K_Angle_2_Turn = K_Angle_2 * 1

# 디버그 모드
DEBUG = False

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
        self.sub_depth = self.create_subscription(Image, SUB_TOPIC_DEPTH, self.update_depth_data, self.qos_sub)

        # Publisher 선언
        self.command_publisher = self.create_publisher(Int8MultiArray, PUB_TOPIC_NAME, self.qos_pub)

        # 데이터 저장 레지스터 선언
        self.car_data = None
        self.lane_data = None
        self.traffic_data = None
        self.lidar_data = None
        self.yolo_data = None
        self.depth_data = None

        # State 저장 레지스터 선언 (0, 1, 2, 3) | 0은 초기화 상태를 의미함
        self.state = 0

        # Lane 위치 저장 레지스터 선언 (1, 2)
        self.lane_state = None

        # Timer 선언
        self.timer = self.create_timer(PERIOD, self.motion_decision_callback)

        # 전송 데이터 기억
        self.steer_angle_reg = 0 # send_command 함수에서 사용
        self.traffic_reg_yolo = [] # update_yolo_data 함수에서 사용
        self.traffic_reg = [] # update_traffic_data 함수에서 사용

        # 카운트 저장소 선언
        self.cnt = 0

##### <변수 업데이트를 위한 함수 선언> #####################################################################

    def update_car_data(self, msg):
        self.car_data = msg # car position

    def update_lane_data(self, msg):
        self.lane_data = msg # angle, center position

    def update_traffic_data(self, msg):
        self.traffic_data = msg # R, Y, G, N

        # Traffic Light 데이터 기록
        self.traffic_reg.append(self.traffic_data.data)
        
        # 4개로 Register 크기 제한
        if len(self.traffic_reg) > 4:
            self.traffic_reg.pop(0)

    def update_lidar_data(self, msg):
        self.lidar_data = msg # T/F

    def update_yolo_data(self, msg):
        self.yolo_data = msg # segmentation data

        # Traffic Light 감지 여부 기록
        self.traffic_reg_yolo.append(len(self.yolo_data.traffic_light) > 0)
        
        # 10개로 Register 크기 제한
        if len(self.traffic_reg_yolo) > 10:
            self.traffic_reg_yolo.pop(0)

    def update_depth_data(self, msg):
        self.depth_data = msg # Depth Image

######################################################################################################

    # 제어 명령 전송 함수 (-128 ~ 127)
    def send_command(self, steer_angle:int, left_speed:int, right_speed:int):
        msg = Int8MultiArray()

        # 조향각 데이터가 비어있는 경우
        if steer_angle == None:
            # 이전 조향각 반영
            steer_angle = self.steer_angle_reg

        else:
            # 조향각 업데이트
            self.steer_angle_reg = steer_angle

        msg.data = [int(np.clip(int(steer_angle), -128, 127)),
                    int(np.clip(int(left_speed), -128, 127)),
                    int(np.clip(int(right_speed), -128, 127))]

        if DEBUG == True:
            msg = Int8MultiArray()     

        self.command_publisher.publish(msg)


    # Stanley Method 기반 조향각 계산 함수
    def calculate_steering_angle(self, target_point:list, car_center_point:list, path_slope:float, vehicle_speed:int, k_angle, k_stanley):
            # Heading Error
            heading_error = path_slope * k_angle

            # 횡방향 오차 계산
            lateral_error = target_point[0] - car_center_point[0]

            # 조향각 계산
            steering_angle = heading_error + np.arctan(k_stanley * lateral_error / (vehicle_speed + 1e-6))*(180/np.pi)
            return int(np.clip(steering_angle, -40, 40)) # 각도 제한 (-40~40)


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

            # State 3 : stop_mode
            elif self.state == 3:
                self.state = self.stop_mode()

        except Exception as e:
            self.get_logger().warn(f"{e}")
            pass


### <State 정의 함수> ####################################################################

    # State 0
    def init_mode(self) -> int:      
        # 데이터가 전부 수신되었을 경우, 처리 시작 (1 : 전체 확인 | 2 : LIDAR 제외 | 3 : DEPTH 제외 | 4 : LIDAR & DEPTH 제외)

        #if self.car_data != None and self.lane_data != None and self.traffic_data != None and self.lidar_data != None and self.yolo_data != None and self.depth_data != None:
        #if self.car_data != None and self.lane_data != None and self.traffic_data != None and self.yolo_data != None and self.depth_data != None:
        #if self.car_data != None and self.lane_data != None and self.traffic_data != None and self.lidar_data != None and self.yolo_data != None:
        if self.car_data != None and self.lane_data != None and self.traffic_data != None and self.yolo_data != None:

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
            self.get_logger().warn(f"{self.car_data != None}, {self.lane_data != None}, {self.traffic_data != None}, {self.lidar_data != None}, {self.yolo_data != None}, {self.depth_data != None}")

            return 0

########################################################################################

    # State 1
    def drive_mode(self) -> int:
        self.get_logger().info(f"drive_mode : {self.lane_state}")


        # 1차선에 위치하고 신호등이 5번 감지된 경우
        if self.lane_state == 1 and len(self.yolo_data.traffic_light) > 0 and self.traffic_reg_yolo.count(True) >= 5:
            return 2

        # 2차선에 위치하고 신호등이 2번 감지된 경우
        elif self.lane_state == 2 and len(self.yolo_data.traffic_light) > 0 and self.traffic_reg_yolo.count(True) >= 2:
            return 2

        # 1차선에 있을 경우, 속도 및 조향 설정
        if self.lane_state == 1:
            steer_angle = self.calculate_steering_angle(target_point = [self.lane_data.lane1_x, self.lane_data.lane1_y],
                                                        car_center_point = BUMPER_POSITION, 
                                                        vehicle_speed = 120,
                                                        path_slope = self.lane_data.slope1,
                                                        k_angle=K_Angle_1,
                                                        k_stanley=K_Stanley_1)
            # Differential 구현
            if steer_angle > 20:
                self.send_command(steer_angle = steer_angle, left_speed = 120, right_speed = 120) 
            elif steer_angle <-20:  
                self.send_command(steer_angle = steer_angle, left_speed = 120, right_speed = 120) 
            else:
               self.send_command(steer_angle = steer_angle, left_speed = 120, right_speed = 120) 

        # 2차선에 있을 경우, 속도 및 조향 설정
        elif self.lane_state == 2:
            steer_angle = self.calculate_steering_angle(target_point = [self.lane_data.lane2_x, self.lane_data.lane2_y],
                                                        car_center_point = BUMPER_POSITION, 
                                                        vehicle_speed = 120,
                                                        path_slope = self.lane_data.slope2, 
                                                        k_angle=K_Angle_2,
                                                        k_stanley=K_Stanley_2)
            # Differential 구현
            if steer_angle > 20:
                self.send_command(steer_angle = steer_angle, left_speed = 120, right_speed = 120) 
            elif steer_angle <-20:  
                self.send_command(steer_angle = steer_angle, left_speed = 120, right_speed = 120) 
            else:
               self.send_command(steer_angle = steer_angle, left_speed = 120, right_speed = 120) 

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
                                                        vehicle_speed = 120,
                                                        path_slope=self.lane_data.slope1,
                                                        k_angle=K_Angle_1_Turn,
                                                        k_stanley=K_Stanley_1_Turn)
            
            # 조향 각도 제한 [좌:-20, 우:40]
            self.send_command(steer_angle = int(np.clip(steer_angle, -20, 40)), left_speed = 120, right_speed = 120) 
            
            # Driving Mode로의 변동 조건 (Count 90 이상)
            if self.cnt > 90:
                self.lane_state = 1
                self.cnt = 0
                return 1
            

        # 1차선에 위치해 있을 경우
        if self.lane_state == 1:
            self.get_logger().info(f"lane_change_mode : 1 -> 2")
            steer_angle = self.calculate_steering_angle(target_point = [self.lane_data.lane2_x, self.lane_data.lane2_y],
                                                        car_center_point = BUMPER_POSITION, 
                                                        vehicle_speed = 120,
                                                        path_slope=self.lane_data.slope2,
                                                        k_angle=K_Angle_2_Turn,
                                                        k_stanley=K_Stanley_2_Turn)

            # 조향 각도 제한 [좌:-40, 우:20]
            self.send_command(steer_angle = int(np.clip(steer_angle, -40, 20)), left_speed = 120, right_speed = 120) 
            
            # Driving Mode로의 변동 조건 (Count 90 이상)
            if self.cnt > 90:
                self.lane_state = 2
                self.cnt = 0
                return 1

        # 카운트 증가
        self.cnt += 1

        # 기존 상태 유지
        return 2
    
########################################################################################

    # State 3
    def stop_mode(self) -> int:
        self.get_logger().info(f"stop_mode : {self.lane_state}")
        self.send_command(steer_angle = 0, left_speed = 0, right_speed = 0)

        # 신호등이 빨간색이 아닐 경우
        if self.traffic_data.data != "R":
            return 1
        
        # 신호등이 빨간색일 경우
        else:
            return 3

########################################################################################

def main():
    rclpy.init()
    motion_planner_node = motion_planner()
    rclpy.spin(motion_planner_node)

    motion_planner_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()