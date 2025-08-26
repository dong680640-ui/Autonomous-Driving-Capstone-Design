###########
# 후방 전용 #
###########

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from interfaces_pkg.msg import CarData, SegmentGroup

import logging


## <Parameter> #######################################################################################

# 구독 토픽 이름
SUB_TOPIC_NAME = "segmented_data_rear"

# 배포 토픽 이름
PUB_TOPIC_NAME = "car_data_rear"

# 로깅 여부
LOG = True

######################################################################################################


class CarDetector(Node):
    def __init__(self):
        super().__init__('car_info_extractor_rear')

        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value

        # QoS settings
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self.subscriber = self.create_subscription(SegmentGroup, self.sub_topic, self.yolov8_detections_callback, self.qos_profile)
        self.publisher = self.create_publisher(CarData, self.pub_topic, self.qos_profile)
    
        # 로깅 여부 설정
        if LOG == False: 
            self.get_logger().set_level(logging.FATAL)


    def yolov8_detections_callback(self, detection_msg: SegmentGroup):
        car_bbox = detection_msg.car  # int32[] 형태

        # Message 선언
        car_data = CarData()

        # 데이터가 없거나, 좌표 개수가 4의 배수가 아니라면 return
        if not car_bbox or len(car_bbox) % 4 != 0:
            self.get_logger().info(f"x = {car_data.x} | y = {car_data.y}")
            self.publisher.publish(car_data)          
            return  

        # 4개씩 묶어서 (x1, y1, x2, y2) 좌표 추출
        for i in range(0, len(car_bbox), 4):
            x1, y1, x2, y2 = car_bbox[i:i+4]

            # 중심 좌표 계산
            car_center_x = (x1 + x2) / 2.0
            car_center_y = (y1 + y2) / 2.0

            # 면적 계산
            car_area = abs(x1 - x2) * abs(y1 - y2)

            # 메시지에 데이터 추가
            car_data.x.append(car_center_x)
            car_data.y.append(car_center_y)
            car_data.area.append(car_area)
            car_data.xyxy.extend([x1, y1, x2, y2])

        # 결과 Publish
        self.get_logger().info(f"x = {car_data.x} | y = {car_data.y}")
        self.publisher.publish(car_data)


def main(args=None):
    rclpy.init(args=args)
    node = CarDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()
  
  
if __name__ == '__main__':
    main()