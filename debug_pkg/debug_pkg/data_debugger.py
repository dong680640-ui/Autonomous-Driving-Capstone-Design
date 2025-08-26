### 시험용 ###
# Motion Planner로 인가되는 정보를 분석하는 Node

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from interfaces_pkg.msg import SegmentGroup, LaneData, CarData
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import numpy as np
import cv2
import cv_bridge


## <Parameter> #####################################################################################

# 노드 이름
NODE_NAME = "yolo_debugger"

# 구독 토픽 이름
SUB_TOPIC_NAME_IMG = "image_publisher"
SUB_TOPIC_NAME_LANE_DATA = "lane_data"

# 영상 크기
FRAME_SIZE = [640, 480]

# 범퍼 위치
BUMPER_POSITION = [325, 479]

######################################################################################################


class data_debugger(Node):
    def __init__(self):
        super().__init__("data_debugger")

        self.qos_sub = QoSProfile( # Subscriber QOS 설정
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
                )
        
        self.subscriber_img = self.create_subscription(Image, SUB_TOPIC_NAME_IMG, self.debug_callback, self.qos_sub)
        self.subscriber_lane_data = self.create_subscription(LaneData, SUB_TOPIC_NAME_LANE_DATA, self.lane_data_update, self.qos_sub)

        self.bridge = cv_bridge.CvBridge()

        # Lane Data
        self.lane_data = None

        
    def lane_data_update(self, msg):
        self.lane_data = msg

    def debug_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg)

        src_mat = [[227, 323],
                   [407, 323],
                   [487, 440],
                   [160, 440]]
        
        dst_mat = [[round(FRAME_SIZE[0] * 0.3), round(FRAME_SIZE[1] * 0.0)],
                   [round(FRAME_SIZE[0] * 0.7), round(FRAME_SIZE[1] * 0.0)], 
                   [round(FRAME_SIZE[0] * 0.7), round(FRAME_SIZE[1] * 1.0)],
                   [round(FRAME_SIZE[0] * 0.3), round(FRAME_SIZE[1] * 1.0)]]

        # 행렬 변환 연산
        img = bird_view_converter(img, srcmat=src_mat, dstmat=dst_mat)

        # Bumper 위치
        cv2.line(img, (BUMPER_POSITION[0], BUMPER_POSITION[1]), (BUMPER_POSITION[0], 0), (255, 0, 0), 3)        
        cv2.circle(img, [BUMPER_POSITION[0], BUMPER_POSITION[1]], 5, (255, 0, 0), thickness=5)

        # 수집한 데이터를 화면에 출력
        if self.lane_data != None:
            cv2.putText(img, f"{int(self.lane_data.slope1)}", [0, 240], cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.line(img, (BUMPER_POSITION[0], BUMPER_POSITION[1]), (self.lane_data.lane1_x, self.lane_data.lane1_y), (0, 0, 255), 3)
            cv2.circle(img, [self.lane_data.lane1_x, self.lane_data.lane1_y], 5, (0, 0, 255), thickness=5)
            
            cv2.putText(img, f"{int(self.lane_data.slope2)}", [320, 240], cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.line(img, (BUMPER_POSITION[0], BUMPER_POSITION[1]), (self.lane_data.lane2_x, self.lane_data.lane2_y), (0, 0, 255), 3)
            cv2.circle(img, [self.lane_data.lane2_x, self.lane_data.lane2_y], 5, (0, 0, 255), thickness=5)

        cv2.imshow("DATA_DEBUG", img)
        cv2.waitKey(1)


def bird_view_converter(img, srcmat, dstmat):
    srcmat = np.float32(srcmat)
    dstmat = np.float32(dstmat)
    transform_mat = cv2.getPerspectiveTransform(srcmat, dstmat)

    img = cv2.warpPerspective(img, transform_mat, (img.shape[1], img.shape[0]))

    return img


def main():
    rclpy.init()
    data_debugger_node = data_debugger()
    rclpy.spin(data_debugger_node)

    data_debugger_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()