import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image

import os
import cv2
import cv_bridge
import torch
import torchvision

import logging

from .lib.depth_estimator.midas.midas_net_custom import MidasNet_small
from .lib.depth_estimator.midas.transforms import Resize, NormalizeImage, PrepareForNet


## <Parameter> #######################################################################################

# 구독 토픽 이름
SUB_TOPIC_NAME = "image_publisher"

# 배포 토픽 이름
PUB_TOPIC_NAME = "depth_data"

# 로깅 여부
LOG = True

# 화면 출력
DEBUG = True

# Thread 수 (CPU 전용, 본인의 CPU Core 수보다 약간 적게 설정)
THREAD = 2

######################################################################################################


class DepthExtractor(Node):
    def __init__(self):
        super().__init__('depth_extractor')

        # QoS settings
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self.subscriber = self.create_subscription(Image, SUB_TOPIC_NAME, self.depth_estimation_callback, self.qos_profile)
        self.publisher = self.create_publisher(Image, PUB_TOPIC_NAME, self.qos_profile)
        
        # CV Bridge Object 선언
        self.bridge = cv_bridge.CvBridge() 

        # 로깅 여부 설정
        if LOG == False: 
            self.get_logger().set_level(logging.FATAL)

        # 속도 최적화를 위한 설정
        torch.set_flush_denormal(True)
        
        # GPU 사용여부 확인
        if torch.cuda.is_available():
            device = torch.device("cuda") 

        else:
            torch.set_num_threads(THREAD)
            device = torch.device("cpu")

        # MIDAS 모델 호출
        name = os.path.dirname(__file__) + "/lib/depth_estimator/" + "midas_v21_small_256_16bit.pt"
        param = torch.load(name, map_location=device)
        self.model = MidasNet_small()  

        # 모델에 파라미터 대입
        self.model.load_state_dict(param)
        
        # 추론 모드 설정
        self.model.eval()

        # 모델 컴파일
        self.model = torch.compile(self.model, fullgraph=False, mode="reduce-overhead", backend="eager")

        # 이미지 변환 Object 선언
        self.transform = torchvision.transforms.Compose(
                [
                lambda img: {"image": img / 255.0},
                #Resize(
                #    256,
                #    256,
                #    resize_target=None,
                #    keep_aspect_ratio=True,
                #    ensure_multiple_of=32,
                #    resize_method="upper_bound",
                #    image_interpolation_method=cv2.INTER_CUBIC,
                #),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
                lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
            ]
        )

    def depth_estimation_callback(self, msg: Image):
        # 이미지 변환
        img = self.bridge.imgmsg_to_cv2(msg)

        # 이미지 가공
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        input_batch = self.transform(img)
        
        # 연산 처리
        with torch.no_grad():
            frame = self.model(input_batch).squeeze().cpu().numpy()

        # 이미지 크기 복원
        frame = cv2.resize(frame, (640, 480))

        if DEBUG == True:
            frame = cv2.normalize(frame, None, 0, 1, cv2.NORM_MINMAX)
            cv2.imshow("MIDAS", frame)
            cv2.waitKey(5)

        # 데이터 전송
        self.publisher.publish(self.bridge.cv2_to_imgmsg(frame))


def main(args=None):
    rclpy.init(args=args)
    node = DepthExtractor()
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