### 시험용 ###
# yolov8 node에서 발행한 topic 데이터 점검을 위한 코드

import rclpy
from rclpy.node import Node
from interfaces_pkg.msg import SegmentGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import numpy as np
import cv2

import os

## <Parameter> #####################################################################################

# 노드 이름
NODE_NAME = "yolo_debugger"

# 구독 토픽 이름
SUB_TOPIC_NAME = "segmented_data"

# CV 처리 영상 출력 여부
DEBUG = True

# 영상 크기
FRAME_SIZE = [640, 480]

######################################################################################################


class yolo_debugger(Node):
    def __init__(self, node_name, sub_topic_name, frame_size, debug):
        super().__init__(node_name)

        self.qos_sub = QoSProfile( # Subscriber QOS 설정
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
                )
        
        self.subscriber = self.create_subscription(SegmentGroup, sub_topic_name, self.debug_callback, self.qos_sub)

        self.frame_size = frame_size

        self.debug = debug

    def debug_callback(self, msg):
        lane_1 = np.array(msg.lane_1).reshape(-1, 2)
        lane_2 = np.array(msg.lane_2).reshape(-1, 2)
        traffic_light = np.array(msg.traffic_light).reshape(-1, 2)
        car = np.array(msg.car).reshape(-1, 2)
        crosswalk = np.array(msg.crosswalk).reshape(-1, 2)

        os.system("clear")
        print(f"lane_1        | {lane_1.shape}")
        print(f"lane_2        | {lane_2.shape}")
        print(f"traffic_light | {traffic_light.shape}")
        print(f"car           | {car.shape}")
        print(f"crosswalk     | {crosswalk.shape}")


        if self.debug == True:
            img_background = np.zeros([self.frame_size[1], self.frame_size[0]])
            total_point = np.concatenate([lane_1, lane_2, traffic_light, car, crosswalk], 0)

            for k in range(total_point.shape[0]):
                img_background[int(total_point[k][1])][int(total_point[k][0])] = 1

            cv2.imshow("DEBUG", img_background)
            cv2.waitKey(5)


def main():
    rclpy.init()
    yolo_debugger_node = yolo_debugger(NODE_NAME, SUB_TOPIC_NAME, FRAME_SIZE, DEBUG)
    rclpy.spin(yolo_debugger_node)

    yolo_debugger_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()