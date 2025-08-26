import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from interfaces_pkg.msg import BoolMultiArray

from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

import logging


## <Parameter> #######################################################################################

# 구독 토픽 이름
SUB_TOPIC_NAME = 'lidar_processed' 

# 발행 토픽 이름
PUB_TOPIC_NAME = 'lidar_data'

# 로깅 여부
LOG = True

# 감지 카운트
COUNT = 3

# 각도 설정 (좌)
START_ANGLE_L = 0  # 감지 각도 범위의 시작 값
END_ANGLE_L = 15   # 감지 각도 범위의 끝 값
        
# 범위 설정 (좌)
RANGE_MIN_L = 0.3  # 감지 거리 범위의 최소값 [m]
RANGE_MAX_L = 2.5  # 감지 거리 범위의 최대값 [m]

# 각도 설정 (우)
START_ANGLE_R = 165  # 감지 각도 범위의 시작 값
END_ANGLE_R = 180   # 감지 각도 범위의 끝 값
        
# 범위 설정 (우)
RANGE_MIN_R = 0.3  # 감지 거리 범위의 최소값 [m]
RANGE_MAX_R = 2.5  # 감지 거리 범위의 최대값 [m]

# 영역 내의 점 카운트 상한
DOT_COUNT = 1

######################################################################################################


## <LIDAR> ###########################################################################################
#        270
#       #######  (Motor)
#     0 # 본체 ######### 180     (Counter-Clockwise)
#       #######
#         90
######################################################################################################


class lidar_object_detector(Node):
    def __init__(self):
        super().__init__('lidar_object_detector')

        # QOS 선언
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # Publisher / Subscriber 선언
        self.subscriber = self.create_subscription(LaserScan, SUB_TOPIC_NAME, self.lidar_callback, self.qos_profile)
        self.publisher = self.create_publisher(BoolMultiArray, PUB_TOPIC_NAME, self.qos_profile) 

        # 감지 카운트를 위한 저장소
        self.detection_reg_l = []
        self.state_reg_l = bool()

        self.detection_reg_r = []
        self.state_reg_r = bool()

        # 로깅 여부 설정
        if LOG == False: 
            self.get_logger().set_level(logging.FATAL)


    def lidar_callback(self, msg):
        # 수신받은 거리 데이터 추출
        ranges = msg.ranges

        # 감지 여부 추출
        detected_l = self.detect_object(ranges=ranges, start_angle=START_ANGLE_L, end_angle=END_ANGLE_L, range_min=RANGE_MIN_L, range_max=RANGE_MAX_L)
        detected_r = self.detect_object(ranges=ranges, start_angle=START_ANGLE_R, end_angle=END_ANGLE_R, range_min=RANGE_MIN_R, range_max=RANGE_MAX_R)
        
        # 감지 카운트
        detection_result_l = self.check_consecutive_detections_l(detected_l, COUNT)
        detection_result_r = self.check_consecutive_detections_r(detected_r, COUNT)

        # 메시지 전송
        detection_msg = BoolMultiArray()
        detection_msg.data.append(detection_result_l)
        detection_msg.data.append(detection_result_r)

        self.publisher.publish(detection_msg)

        self.get_logger().info(f'Detection Result = {detection_result_l} {detection_result_r}')

    def detect_object(self, ranges, start_angle, end_angle, range_min, range_max):
        # 항상 360 출력
        total_angle = len(ranges)
    
        # 0 ~ 360 범위로 고정
        if start_angle > end_angle:
            end_angle += total_angle

        # 점 개수를 카운트하기 위한 변수 선언
        dot_cnt = 0

        # 설정 구간 내에 값 존재 여부 확인
        for i in range(start_angle, end_angle + 1):
            i = i % total_angle

            # 카운트 증가 조건
            if range_min <= ranges[i] <= range_max:
                dot_cnt += 1

            if dot_cnt > DOT_COUNT:
                return True
        
        # 감지되지 않았을 경우 반환값
        return False
    

    def check_consecutive_detections_l(self, detection, cnt):
        # 레지스터에 감지 데이터 추가
        self.detection_reg_l.append(detection)

        # 레지스터 저장 개수 제한
        if len(self.detection_reg_l) > cnt:
            self.detection_reg_l.pop(0)

        # T -> F 변환 조건
        if self.state_reg_l and self.detection_reg_l.count(False) >= cnt:
            self.state_reg_l = False
        
        # F -> T 변환 조건
        elif not self.state_reg_l and self.detection_reg_l.count(True) >= cnt:
            self.state_reg_l = True

        return self.state_reg_l
    
    def check_consecutive_detections_r(self, detection, cnt):
        # 레지스터에 감지 데이터 추가
        self.detection_reg_r.append(detection)

        # 레지스터 저장 개수 제한
        if len(self.detection_reg_r) > cnt:
            self.detection_reg_r.pop(0)

        # T -> F 변환 조건
        if self.state_reg_r and self.detection_reg_r.count(False) >= cnt:
            self.state_reg_r = False
        
        # F -> T 변환 조건
        elif not self.state_reg_r and self.detection_reg_r.count(True) >= cnt:
            self.state_reg_r = True

        return self.state_reg_r


def main(args=None):
    rclpy.init(args=args)
    object_detector_node = lidar_object_detector()

    rclpy.spin(object_detector_node)
    
    object_detector_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()