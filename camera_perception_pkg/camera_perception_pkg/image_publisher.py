#################
# 전방 카메라 전용 #
#################

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import cv2
import cv_bridge

import os
import time
import logging


## <Parameter> #######################################################################################

# 영상 소스
FRAME_SRC = "/dev/video2"
#FRAME_SRC = "/home/user/ros2_merge/src/camera_perception_pkg/camera_perception_pkg/lib/test_video.mp4"

# 영상 크기 (가로, 세로)
FRAME_SIZE = [640, 480]

# 노드 이름
NODE_NAME = "image_publisher"

# 토픽 이름
TOPIC_NAME = "image_publisher"

# 전송 주기
PUBLISH_PERIOD = 0.03

# 로깅 여부
LOG = True

# 카메라 밝기 설정
BRIGHTNESS = 128

######################################################################################################


## <로그 출력> #########################################################################################
# DEBUG	self.get_logger().debug("msg")
# INFO	self.get_logger().info("msg")
# WARN	self.get_logger().warn("msg")
# ERROR	self.get_logger().error("msg")
# FATAL	self.get_logger().fatal("msg")
#######################################################################################################


## <QOS> ##############################################################################################
# Reliability : RELIABLE(신뢰성 보장), BEST_EFFORT(손실 감수, 최대한 빠른 전송)
# Durability : VOLATILE(전달한 메시지 제거), TRANSIENT_LOCAL(전달한 메시지 유지) / (Subscriber가 없을 때에 한함)
# History : KEEP_LAST(depth 만큼의 메시지 유지), KEEP_ALL(모두 유지)
# Liveliness : 활성 상태 감시
# Deadline : 최소 동작 보장 
#######################################################################################################


class image_publisher(Node):
    def __init__(self, frame_src, node_name, topic_name, publish_period, log):
        super().__init__(node_name)

        self.cap = cv2.VideoCapture(frame_src) # 영상 프레임 출력을 위한 Object 변수를 선언
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0]) # 영상 가로 길이 지정
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1]) # 영상 세로 길이 지정
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)

        self.qos = QoSProfile( # QOS 설정
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
                )
        
        self.publisher = self.create_publisher(Image, topic_name, self.qos) # Topic 선언
        self.timer = self.create_timer(publish_period, self.publish_callback) # Topic 발행을 위한 Timer 선언
        self.bridge = cv_bridge.CvBridge() # CV Bridge Object 선언

        if log == False: # 로깅 여부 설정
            self.get_logger().set_level(logging.FATAL)


    def publish_callback(self):
        ret, frame = self.cap.read() # 영상 프레임 읽기

        if ret == True: # 정상
            self.publisher.publish(self.bridge.cv2_to_imgmsg(frame))
            self.get_logger().info("frame published")

        else: # 오류
            self.get_logger().warn("unable to read frame") # 오류 출력
            time.sleep(0.5) # 시간 지연, 오류 정정 시간 확보


    def shutdown(self):
        self.cap.release() # Object 변수 해제
        cv2.destroyAllWindows() # CV 창 닫기


def main():
    rclpy.init()
    image_publisher_node = image_publisher(FRAME_SRC, NODE_NAME, TOPIC_NAME, PUBLISH_PERIOD, LOG)
    rclpy.spin(image_publisher_node)

    image_publisher_node.shutdown()
    image_publisher_node.destroy_node()
    rclpy.shutdown()


if __name__== "__main__":
    main()