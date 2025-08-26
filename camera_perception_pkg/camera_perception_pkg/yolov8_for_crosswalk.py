#################
# 전방 카메라 전용 #
#################

# Crosswalk 감지 목적

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from interfaces_pkg.msg import SegmentGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import torch
from ultralytics import YOLO

import cv2
import cv_bridge

import os
import logging


## <Parameter> #######################################################################################

# 노드 이름
NODE_NAME = "yolov8_for_crosswalk"

# 발행 토픽 이름
TOPIC_NAME = "crosswalk_data"

# 라벨 이름 (crosswalk)
LABEL_NAME = ["crosswalk"]

# 구독 토픽 이름
SUB_TOPIC_NAME = "image_publisher"

# PT 파일 이름 지정 (확장자 포함, 해당 실행 파일의 디렉터리는 이미 앞에 포함되어 있다는 점에 유념)
PT_NAME = "lib/pt/best.cross.0511.0017.pt"

# CV 처리 영상 출력 여부
DEBUG = True

# 로깅 여부
LOG = True

# Thread 수 (CPU 전용, 본인의 CPU Core 수보다 약간 적게 설정)
THREAD = 4

# RGB <-> BGR 반전 모드 (주의 : 학습 모델이 반전된 사진을 기반으로 형성되었을 시에 사용)
INV_COLOR = False

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


class yolov8(Node):
    def __init__(self, node_name, topic_name : list, sub_topic_name, pt_name, debug, label_name, thread, log):
        super().__init__(node_name)

        # 속도 최적화를 위한 설정
        torch.set_flush_denormal(True)

        # YOLO Model 선언
        self.model = YOLO(os.path.dirname(__file__) + "/" + pt_name) 
        self.model.model.eval()

        # YOLO Model 컴파일
        self.model.model = torch.compile(self.model.model, mode="max-autotune")

        if torch.cuda.is_available(): # Nvidia GPU 설정
            self.model.to("cuda")

        else: # CPU 설정
            torch.set_num_threads(thread)
            self.model.to("cpu")

        self.qos_pub = QoSProfile( # Publisher QOS 설정
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
                )

        self.qos_sub = QoSProfile( # Subscriber QOS 설정
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
                )

        # 라벨 이름 변수 선언
        self.label_0 = label_name[0]    

        # 디버그 변수 선언
        self.debug = debug

        # Publisher 선언
        self.publisher = self.create_publisher(SegmentGroup, topic_name, self.qos_pub) 

        # Subscriber 선언
        self.subscriber = self.create_subscription(Image, sub_topic_name, self.recognizer_callback, self.qos_sub)

        # CV Bridge Object 선언
        self.bridge = cv_bridge.CvBridge()

        # 로깅 여부 설정
        if log == False: 
            self.get_logger().set_level(logging.FATAL)


    def recognizer_callback(self, img_msg):
        # Publishing을 위한 Message 선언
        msg = SegmentGroup()

        frame = self.bridge.imgmsg_to_cv2(img_msg) # Frame 수령 및 처리

        # RGB <-> BGR 반전 처리
        if INV_COLOR == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        predicted = self.model.predict(frame, verbose=False) # Frame Segmentation 처리

        if self.debug == True: # 디버깅(화면 출력) 여부 결정
            cv2.imshow("YOLO-CROSSWALK", predicted[0].plot())
            cv2.waitKey(5)

        # 카운트를 위한 변수 선언
        cnt_0 = 0

        # 확률에 대한 내림차순 정렬 (이미 Ultralytics의 전처리 과정에 해당 작업이 포함되어 있기에 제외함, Deprecated)
        # predict_box = self.predicted[0].boxes[torch.argsort(self.predicted[0].boxes.conf, descending=True)]
        # predict_mask = self.predicted[0].masks[torch.argsort(self.predicted[0].boxes.conf, descending=True)]

        # Box, Mask 변수 선언
        predict_box = predicted[0].boxes
        predict_mask = predicted[0].masks

        self.get_logger().info(f"{len(predict_box.conf.tolist())} object(s) detected | value = {predict_box.conf.tolist()}")

        # Box : 상자 / Keypoint : 관절 표현 / Mask : 영역 표시
        for n, predict_val in enumerate(predict_box):
            name = predicted[0].names[int(predict_val.cls.item())].strip()

            if name == self.label_0.strip() and cnt_0 == 0:
                msg.crosswalk = predict_box[n].xyxy[0].to(torch.int16).flatten().tolist()
                cnt_0 += 1   

                # Polygon 형식
                # self.msg_2.data = self.predicted[0].masks[n].xy[0].flatten().tolist() 

                # Box 형식
                # self.msg_2.data = self.predicted[0].boxes[n].xyxy[0]       

        self.publisher.publish(msg)      


    def shutdown(self):
        cv2.destroyAllWindows() # CV 창 닫기


def main():
    rclpy.init()
    yolov8_node = yolov8(NODE_NAME, TOPIC_NAME, SUB_TOPIC_NAME, PT_NAME, DEBUG, LABEL_NAME, THREAD, LOG)
    rclpy.spin(yolov8_node)

    yolov8_node.shutdown()
    yolov8_node.destroy_node()
    rclpy.shutdown()


if __name__== "__main__":
    main() 
