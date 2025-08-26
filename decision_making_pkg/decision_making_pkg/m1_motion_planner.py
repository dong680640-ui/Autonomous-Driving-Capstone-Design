########################
# For Mission 1 (Real) #
########################

import rclpy
from rclpy.node import Node
from interfaces_pkg.msg import CarData, LaneData, SegmentGroup, MotionCommand
from std_msgs.msg import String
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import numpy as np

## <Parameter> #####################################################################################

# 구독 토픽 이름
SUB_TOPIC_CAR = "car_data" 
SUB_TOPIC_LANE = "lane_data"
SUB_TOPIC_TRAFFIC = "traffic_data"
SUB_TOPIC_YOLO = "segmented_data"
SUB_TOPIC_YOLO_FOR_CROSSWALK = "crosswalk_data"

# 발행 토픽 이름
PUB_TOPIC_NAME = "command_pre_data"

# 연산 주기 설정
PERIOD = 0.1

# 차량 범퍼 위치
BUMPER_POSITION = [294, 462]

# 보정 상수
K_Stanley_1 = 1.0
K_Angle_1 = 0.025

K_Stanley_2 = 0.9
K_Angle_2 = 0.05

K_Stanley_1_Turn = K_Stanley_1 * 1
K_Stanley_2_Turn = K_Stanley_2 * 1
K_Angle_1_Turn = K_Angle_1 * 1
K_Angle_2_Turn = K_Angle_2 * 1

# 디버그 모드
DEBUG = False

# 기준 속도
SPEED = 255
SPEED_LANE_CHANGE = 255

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
        self.sub_yolo = self.create_subscription(SegmentGroup, SUB_TOPIC_YOLO, self.update_yolo_data, self.qos_sub)
        self.sub_yolo_for_crosswalk = self.create_subscription(SegmentGroup, SUB_TOPIC_YOLO_FOR_CROSSWALK, self.update_yolo_data_cross, self.qos_sub)

        # Publisher 선언
        self.command_publisher = self.create_publisher(MotionCommand, PUB_TOPIC_NAME, self.qos_pub)

        # 데이터 저장 레지스터 선언
        self.car_data = None
        self.lane_data = None
        self.traffic_data = None
        self.yolo_data = None
        self.yolo_data_cross = None

        # State 저장 레지스터 선언 (0, 1, 2) | 0은 초기화 상태를 의미함
        self.state = 0

        # Lane 위치 저장 레지스터 선언 (1, 2)
        self.lane_state = None

        # Timer 선언
        self.timer = self.create_timer(PERIOD, self.motion_decision_callback)

        # 전송 데이터 기억
        self.steer_angle_reg = 0 # send_command 함수에서 사용
        self.crosswalk_reg = [] # update_yolo_data 함수에서 사용

        # 카운트 저장소 선언
        self.cnt = 0

##### <변수 업데이트를 위한 함수 선언> #####################################################################

    def update_car_data(self, msg):
        self.car_data = msg # car position

    def update_lane_data(self, msg):
        self.lane_data = msg # angle, center position

    def update_traffic_data(self, msg):
        self.traffic_data = msg # R, Y, G, N

    def update_yolo_data(self, msg):
        self.yolo_data = msg # segmentation data

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

        #self.get_logger().info(f"angle:{steer_angle}")
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

        except Exception as e:
            self.get_logger().warn(f"{e}")


### <State 정의 함수> ####################################################################

    # State 0
    def init_mode(self) -> int:      
        # 데이터가 전부 수신되었을 경우, 처리 시작
        if self.car_data != None and self.lane_data != None and self.traffic_data != None and self.yolo_data != None and self.yolo_data_cross != None:

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
            self.get_logger().warn(f"{self.car_data != None}, {self.lane_data != None}, {self.traffic_data != None}, {self.yolo_data != None}, {self.yolo_data_cross != None}")

            return 0

########################################################################################

    # State 1
    def drive_mode(self) -> int:
        self.get_logger().info(f"drive_mode : {self.lane_state}")
        
        # 1차선에서 횡단보도가 연속 10번 감지되었을 경우 + 신호등이 감지된 경우
        if self.lane_state == 1 and len(self.yolo_data_cross.crosswalk) > 0 and len(self.yolo_data.traffic_light) > 0 and self.crosswalk_reg.count(True) >= 10:
            # 횡단보도의 위치가 365 이상인 경우 + 카운트가 일정 수준 이상인 경우
            if 365 < np.array(self.yolo_data_cross.crosswalk).reshape(-1, 2)[:, 1].max() and self.cnt > 50:
                return 2

        # 2차선에서 횡단보도가 연속 10번 감지되었을 경우 + 신호등이 감지된 경우
        elif self.lane_state == 2 and len(self.yolo_data_cross.crosswalk) > 0 and len(self.yolo_data.traffic_light) > 0 and self.crosswalk_reg.count(True) >= 10:
            # 횡단보도의 위치가 345 이상인 경우 + 카운트가 일정 수준 이상인 경우
            if 345 < np.array(self.yolo_data_cross.crosswalk).reshape(-1, 2)[:, 1].max() and self.cnt > 50:
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
            if steer_angle < -28:
                self.send_command(steer_angle=steer_angle, left_speed = SPEED-45, right_speed = SPEED)

            elif steer_angle < -26:
                self.send_command(steer_angle=steer_angle, left_speed = SPEED-30, right_speed = SPEED)

            else:
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
            if steer_angle > 25:
                self.send_command(steer_angle = steer_angle, left_speed = SPEED, right_speed = SPEED)

            elif steer_angle <-25:  
                self.send_command(steer_angle = steer_angle, left_speed = SPEED-20, right_speed = SPEED) 

            else:
                self.send_command(steer_angle = steer_angle, left_speed = SPEED, right_speed = SPEED) 

        # 카운트 증가
        self.cnt += 1

        # 계속 주행
        return 1

########################################################################################

    # State 2
    def lane_change_mode(self) -> int:
        
        # 카운트 초기화
        self.cnt = 0

        # 2차선에 위치해 있을 경우
        if self.lane_state == 2:
            self.get_logger().info(f"lane_change_mode : 2 -> 1")
            steer_angle = self.calculate_steering_angle(target_point = [self.lane_data.lane1_x, self.lane_data.lane1_y],
                                                        car_center_point = BUMPER_POSITION, 
                                                        vehicle_speed = SPEED_LANE_CHANGE,
                                                        path_slope=self.lane_data.slope1,
                                                        k_angle=K_Angle_1_Turn,
                                                        k_stanley=K_Stanley_1_Turn)
            
            # 조향 각도 제한 [좌:-30, 우:30]
            self.send_command(steer_angle = int(np.clip(steer_angle, -30, 30)), left_speed = SPEED_LANE_CHANGE, right_speed = SPEED_LANE_CHANGE) 
            
            # Driving Mode로 이동
            self.lane_state = 1
            return 1

        # 1차선에 위치해 있을 경우
        if self.lane_state == 1:
            self.get_logger().info(f"lane_change_mode : 1 -> 2")
            steer_angle = self.calculate_steering_angle(target_point = [self.lane_data.lane2_x, self.lane_data.lane2_y],
                                                        car_center_point = BUMPER_POSITION, 
                                                        vehicle_speed = SPEED_LANE_CHANGE,
                                                        path_slope=self.lane_data.slope2,
                                                        k_angle=K_Angle_2_Turn,
                                                        k_stanley=K_Stanley_2_Turn)

            # 조향 각도 제한 [좌:-30, 우:30]
            self.send_command(steer_angle = int(np.clip(steer_angle, -30, 30)), left_speed = SPEED_LANE_CHANGE, right_speed = SPEED_LANE_CHANGE) 
            
            # Driving Mode로 이동
            self.lane_state = 2
            return 1

########################################################################################

def main():
    rclpy.init()
    motion_planner_node = motion_planner()
    rclpy.spin(motion_planner_node)

    motion_planner_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()