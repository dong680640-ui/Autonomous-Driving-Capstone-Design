import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from interfaces_pkg.msg import SegmentGroup
from std_msgs.msg import String

import logging


## <Parameter> #######################################################################################

# 구독 토픽 이름
SUB_SEGMENTATION_TOPIC_NAME = "segmented_data"
SUB_IMAGE_TOPIC_NAME = "image_publisher"

# 발행 토픽 이름
PUB_TOPIC_NAME = "traffic_data"

# 로깅 여부
LOG = True

######################################################################################################


class TrafficLightDetector(Node):
    def __init__(self):
        super().__init__('traffic_light_detector')

        self.cv_bridge = CvBridge()

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.hsv_ranges = {
                'red1': (np.array([0, 100, 95]), np.array([10, 255, 255])),
                'red2': (np.array([160, 100, 95]), np.array([179, 255, 255])),
                'yellow': (np.array([20, 100, 95]), np.array([30, 255, 255])),
                'green': (np.array([40, 100, 95]), np.array([90, 255, 255]))
                    }

        self.segmentation_sub = self.create_subscription(SegmentGroup, SUB_SEGMENTATION_TOPIC_NAME, self.segment_callback, self.qos_profile)
        self.img_sub = self.create_subscription(Image, SUB_IMAGE_TOPIC_NAME, self.image_callback, self.qos_profile)

        self.publisher = self.create_publisher(String, PUB_TOPIC_NAME, self.qos_profile)

        self.seg_data = []

        # 로깅 여부 설정
        if LOG == False: 
            self.get_logger().set_level(logging.FATAL)


    def segment_callback(self, msg):
        self.seg_data = msg.traffic_light


    def image_callback(self, msg):
        img = self.cv_bridge.imgmsg_to_cv2(msg)

        tl_val = get_traffic_light_color(img, self.seg_data, self.hsv_ranges) 

        # 메시지 전송
        color_msg = String()
        color_msg.data = tl_val
        self.publisher.publish(color_msg)
        
        self.get_logger().info(f'traffic light: {color_msg.data}') 


def get_traffic_light_color(cv_image, xyxy, hsv_ranges):
    if len(xyxy) == 0:
        return 'N'

    else:        
        x_min, y_min, x_max, y_max = xyxy

    roi = cv_image[y_min:y_max, x_min:x_max]       
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_lower1, red_upper1 = hsv_ranges['red1']
    red_lower2, red_upper2 = hsv_ranges['red2']
    yellow_lower, yellow_upper = hsv_ranges['yellow']
    green_lower, green_upper = hsv_ranges['green']
        
    red_mask1 = cv2.inRange(hsv_roi, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_roi, red_lower2, red_upper2)
    red_mask = red_mask1 + red_mask2

    yellow_mask = cv2.inRange(hsv_roi, yellow_lower, yellow_upper)
        
    green_mask = cv2.inRange(hsv_roi, green_lower, green_upper)
        
    red_ratio = cv2.countNonZero(red_mask) / (roi.size / 3)
    yellow_ratio = cv2.countNonZero(yellow_mask) / (roi.size / 3)
    green_ratio = cv2.countNonZero(green_mask) / (roi.size / 3)
        
    max_ratio = max(red_ratio, yellow_ratio, green_ratio)

    if max_ratio < 0.01:  # 1% 이하일 경우 무시
        return 'N'

    elif max_ratio == red_ratio:
        return 'R'
        
    elif max_ratio == yellow_ratio:
        return 'Y'
        
    elif max_ratio == green_ratio:
        return 'G'
        
    else:
        return 'N'


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetector()

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