########################
# For Mission 3 (Real) #
########################

import rclpy
from rclpy.node import Node
from interfaces_pkg.msg import CarData, SegmentGroup, MotionCommand, BoolMultiArray, LineData
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from scipy.interpolate import splrep, splev
import numpy as np
import cv_bridge
import time

## <Parameter> #####################################################################################

# 구독 토픽 이름
SUB_TOPIC_CAR_REAR = "car_data_rear" 
SUB_TOPIC_LIDAR = "lidar_data"
SUB_TOPIC_YOLO_REAR = "segmented_data_rear"
SUB_TOPIC_LINE = "line_data_rear"

# 발행 토픽 이름
PUB_TOPIC_NAME = "command_data"

# 연산 주기 설정
PERIOD = 0.1

# 차량 후방 중심점 위치
BUMPER_POSITION = [320, 462]

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
        self.sub_car_rear = self.create_subscription(CarData, SUB_TOPIC_CAR_REAR, self.update_car_rear_data, self.qos_sub)
        self.sub_lidar = self.create_subscription(BoolMultiArray, SUB_TOPIC_LIDAR, self.update_lidar_data, self.qos_sub)
        self.sub_yolo_rear = self.create_subscription(SegmentGroup, SUB_TOPIC_YOLO_REAR, self.update_yolo_rear_data, self.qos_sub)
        self.sub_line = self.create_subscription(LineData, SUB_TOPIC_LINE, self.update_line_data, self.qos_sub)

        # Publisher 선언
        self.command_publisher = self.create_publisher(MotionCommand, PUB_TOPIC_NAME, self.qos_pub)

        # CV Bridge Object 선언    
        self.bridge = cv_bridge.CvBridge()

        # 데이터 저장 레지스터 선언
        self.car_rear_data = None
        self.lidar_data = None
        self.yolo_rear_data = None
        self.line_data = None

        # State 저장 레지스터 선언 | 0은 초기화 상태를 의미함
        self.state = 0

        # Timer 선언
        self.timer = self.create_timer(PERIOD, self.motion_decision_callback)

        # 전송 데이터 기억
        self.steer_angle_reg = 0 # send_command 함수에서 사용

##### <변수 업데이트를 위한 함수 선언> #####################################################################

    def update_car_rear_data(self, msg):
        self.car_rear_data = msg # car position        

#######################################################################

    def update_lidar_data(self, msg):
        self.lidar_data = msg # T/F

#######################################################################

    def update_yolo_rear_data(self, msg):
        self.yolo_rear_data = msg # segmentation data

#######################################################################

    def update_line_data(self, msg):
        self.line_data = msg # line data

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


    # Pure Pursuit 기반 조향각 계산 함수
    def calculate_steering_angle(self, target_point:list, car_center_point:list, lookahead_dist=20, car_width=1):

        # 오차 계산
        alpha = np.pi + np.arctan((target_point[1]-car_center_point[1])/(target_point[0]-car_center_point[0]+1e-6))

        # 조향각 계산
        angle = np.arctan(-2*car_width*np.sin(alpha)/lookahead_dist) * 180/np.pi

        return angle
    

    # 판단 로직 작성부
    def motion_decision_callback(self):
        try:
            # State 0 : init_mode
            if self.state == 0:
                self.state = self.init_mode()

            # State 1 : search_mode
            elif self.state == 1:
                self.state = self.search_mode()

            # State 2 : turn_mode
            elif self.state == 2:
                self.state = self.turn_mode()

            # State 3 : stop_mode_1
            elif self.state == 3:
                self.state = self.stop_mode_1()

            # State 4 : back_up_mode
            elif self.state == 4:
                self.state = self.back_up_mode()

            # State 5 : stop_mode_2
            elif self.state == 5:
                self.state = self.stop_mode_2()

            # State 6 : forward_mode
            elif self.state == 6:
                self.state = self.forward_mode()

        except Exception as e:
            self.get_logger().warn(f"{e}")


### <State 정의 함수> ####################################################################

    # State 0
    def init_mode(self) -> int:      
        # 데이터가 전부 수신되었을 경우, 처리 시작
        if self.car_rear_data != None and self.lidar_data != None and self.yolo_rear_data != None and self.line_data != None:

            # 0의 지령값 설정
            self.send_command(steer_angle = 0, left_speed = 0, right_speed = 0)

            # 주행 모드로 반환
            return 1
        

        # 데이터가 전부 수신되지 않았을 경우, 오류 전송
        else:
            self.get_logger().warn("data is not yet accepted")
            self.get_logger().warn(f"{self.car_rear_data != None}, {self.lidar_data != None}, {self.yolo_rear_data != None}, {self.line_data != None}")
            return 0

########################################################################################

    # State 1
    def search_mode(self) -> int:
        self.get_logger().info(f"search_mode")   

        # 정속 주행
        self.send_command(steer_angle = 7, left_speed = 100, right_speed = 100)
        
        # 우측 LIDAR에 사물이 감지된 경우
        if self.lidar_data.data[1] == True:
            # 지연
            time.sleep(1.7)
            return 2

        # 현 상태 유지
        return 1

########################################################################################

    # State 2
    def turn_mode(self) -> int:
        self.get_logger().info(f"turn_mode")   

        # 좌측 조향 운전
        self.send_command(steer_angle = -30, left_speed = 40, right_speed = 150)

        # 지연
        time.sleep(4.5)

        # 다음 단계로 이동
        return 3

########################################################################################

    # State 3
    def stop_mode_1(self) -> int:
        self.get_logger().info(f"stop_mode_1")   

        # 정지
        self.send_command(steer_angle = 0, left_speed = 0, right_speed = 0)

        # 1초 지연
        time.sleep(1)

        # 후진 단계로 이동
        return 4

########################################################################################

    # State 4
    def back_up_mode(self) -> int:
        self.get_logger().info(f"back_up_mode")

        # LIDAR 양쪽에 장애물 감지시 정지
        if self.lidar_data.data[0] == True and self.lidar_data.data[1] == True:
            return 5

        # 좌우 차선이 감지된 경우
        if len(self.line_data.left) !=0 and len(self.line_data.right) !=0:
            self.get_logger().info(f"debug:line_center")
            x1_l, y1_l, x2_l, y2_l = self.line_data.left
            x1_r, y1_r, x2_r, y2_r = self.line_data.right

            x_max = (x1_l + x1_r)/2
            y_max = (y1_l + y1_r)/2

            x_min = (x2_l + x2_r)/2
            y_min = (y2_l + y2_r)/2

            # Blank가 감지된 경우
            if len(self.yolo_rear_data.blank_box) > 0:
                self.get_logger().info(f"blank:True")
                x_u, y_u, x_l, y_l = self.yolo_rear_data.blank_box

                x_blank = (x_u + x_l)/2
                y_blank = (y_u + y_l)/2

                y_spline = np.array([y_max, y_blank, y_min, BUMPER_POSITION[1]])

                x_spline = np.array([x_max, x_blank, x_min, BUMPER_POSITION[0]])[np.argsort(y_spline)]
                y_spline = np.array([y_max, y_blank, y_min, BUMPER_POSITION[1]])[np.argsort(y_spline)]               

            # Blank가 감지되지 않은 경우
            else:
                self.get_logger().info(f"blank:False")
                y_spline = np.array([y_max, y_min, BUMPER_POSITION[1]])

                x_spline = np.array([x_max, x_min, BUMPER_POSITION[0]])[np.argsort(y_spline)]
                y_spline = np.array([y_max, y_min, BUMPER_POSITION[1]])[np.argsort(y_spline)]

            spline_params = splrep(y_spline, x_spline, k=2)

            y_spline = np.linspace(0, 640, 641)
            x_spline = splev(y_spline, spline_params)  

            y_target = 460
            x_target = np.nan_to_num(x_spline[y_target], nan=int(BUMPER_POSITION[0]/2))

            target_point = [x_target, y_target]

            # 조향각 계산
            angle = self.calculate_steering_angle(target_point, BUMPER_POSITION, 7)

        # 2개의 차량이 감지된 경우
        elif len(self.car_rear_data.x) == 2:
            self.get_logger().info(f"debug:car_center")

            # 조향각 계산
            target_point = [sum(self.car_rear_data.x)/2, sum(self.car_rear_data.y)/2]
            angle = self.calculate_steering_angle(target_point, BUMPER_POSITION, 7)

        # 1개의 차량이 감지된 경우 + 차선이 1개 감지된 경우(L, R)
        elif len(self.car_rear_data.x) == 1 and ((len(self.line_data.left) == 0 and len(self.line_data.right) != 0) or (len(self.line_data.left) != 0 and len(self.line_data.right) == 0)):
            if len(self.line_data.left) != 0:
                line = self.line_data.left
            else:
                line = self.line_data.right

            # 선 좌표 추출
            x1, y1, x2, y2 = line

            x_line = (x1 + x2)/2
            y_line = (y1 + y2)/2

            # 차량 좌표 추출
            x_car = self.car_rear_data.x[0]
            y_car = self.car_rear_data.y[0]

            # / 편향 
            if x_line < x_car:
                self.get_logger().info(f"debug:correction-/ (1)")
                self.send_command(steer_angle = 30, left_speed = -20, right_speed = -20)

                # 지연
                time.sleep(2)

                # 현 상태 유지
                return 4

            # \ 편향
            else:
                self.get_logger().info(f"debug:correction-\\ (1)")
                self.send_command(steer_angle = -30, left_speed = -20, right_speed = -20)

                # 지연
                time.sleep(2)

                # 현 상태 유지
                return 4
            
        # 가운데 선이 검출된 경우
        elif len(self.line_data.center) != 0:
            x1_c, y1_c, x2_c, y2_c = self.line_data.center
            theta = np.arctan((y1_c - y2_c)/(x1_c - x2_c + 1e-6))*(180/np.pi)

            # / 편향 
            if theta < -1:
                self.get_logger().info(f"debug:correction-/ (2)")
                self.send_command(steer_angle = 15, left_speed = -20, right_speed = -20)

                # 현 상태 유지
                return 4

            # \ 편향                
            elif theta > 1:
                self.get_logger().info(f"debug:correction-\\ (2)")
                self.send_command(steer_angle = -15, left_speed = -20, right_speed = -20)

                # 현 상태 유지
                return 4
            
            else:
                self.get_logger().info(f"debug:none")
                angle = 0

        # 그 외의 경우
        else:
            self.get_logger().info(f"debug:none")
            angle = 0

        # 각도 제한
        angle = angle + 7
        angle = int(np.clip(angle, -30, 30))

        # 후진 진행
        self.send_command(steer_angle = angle, left_speed = -50, right_speed = -50)

        # 현 상태 유지
        return 4

########################################################################################

    # State 5
    def stop_mode_2(self) -> int:
        self.get_logger().info(f"stop_mode_2")   

        # 정지
        self.send_command(steer_angle = 0, left_speed = 0, right_speed = 0)

        # 4초 지연
        time.sleep(4)

        # 마무리 단계로 이동
        return 6

########################################################################################

    # State 6
    def forward_mode(self) -> int:
        self.get_logger().info(f"forward_mode")   

        # 직진
        self.send_command(steer_angle = 0, left_speed = 150, right_speed = 150)

        # 1.1초 지연
        time.sleep(1.1)

        # 우회전
        self.send_command(steer_angle = 30, left_speed = 150, right_speed = 150)

        # 5초 지연
        time.sleep(5)

        # 정지
        self.send_command(steer_angle = 0, left_speed = 0, right_speed = 0)

        # 0.5초 지연
        time.sleep(0.5)

        # 후진
        self.send_command(steer_angle = -30, left_speed = -150, right_speed = -150)

        # 3초 지연
        time.sleep(1.7)

        # 정지
        self.send_command(steer_angle = 0, left_speed = 0, right_speed = 0)

        # 0.5초 지연
        time.sleep(0.5)

        # 직진
        self.send_command(steer_angle = 7, left_speed = 150, right_speed = 150)

        # Trapping
        while True:
            continue

########################################################################################

def main():
    rclpy.init()
    motion_planner_node = motion_planner()
    rclpy.spin(motion_planner_node)

    motion_planner_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
