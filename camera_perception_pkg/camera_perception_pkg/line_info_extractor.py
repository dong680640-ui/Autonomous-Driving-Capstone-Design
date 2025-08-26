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

from interfaces_pkg.msg import SegmentGroup, LineData

import logging
import numpy as np


## <Parameter> #######################################################################################

# 구독 토픽 이름
SUB_TOPIC_NAME = "segmented_data_rear"

# 배포 토픽 이름
PUB_TOPIC_NAME = "line_data_rear"

# 로깅 여부
LOG = True

# CV 처리 영상 출력 여부
DEBUG = True

# 영상 크기
FRAME_SIZE = [640, 480]

######################################################################################################


class LineDetector(Node):
    def __init__(self):
        super().__init__('line_info_extractor_rear')

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
        self.publisher = self.create_publisher(LineData, self.pub_topic, self.qos_profile)
    
        # 로깅 여부 설정
        if LOG == False: 
            self.get_logger().set_level(logging.FATAL)


    def yolov8_detections_callback(self, msg):
        # 메시지 선언
        result = LineData()

        # 빈 이미지 생성
        img_base = np.zeros([FRAME_SIZE[1], FRAME_SIZE[0]]).astype(np.uint8)
        img_hough = np.zeros([FRAME_SIZE[1], FRAME_SIZE[0]]).astype(np.uint8)
        img_hough_post = np.zeros([FRAME_SIZE[1], FRAME_SIZE[0]]).astype(np.uint8)

        # 점 데이터
        if len(msg.line) != 0: # 차선이 감지된 경우
            point = np.array(msg.line).reshape(-1, 2).astype(np.int32)
        elif len(msg.blank_mask) != 0: # 주차 공간이 감지된 경우
            point = np.array(msg.blank_mask).reshape(-1, 2).astype(np.int32)
        else: # 아무것도 감지되지 않은 경우
            point = []           

        # 점 데이터 연결 및 이미지에 투사
        if len(point) != 0:
            cv2.polylines(img_base, [point], isClosed=False, color=255, thickness=2)

            # Hough 변환
            lines = cv2.HoughLinesP(
                img_base, rho=0.5, theta=np.pi/180, threshold=50, 
                minLineLength=50, maxLineGap=100
            )
        else:
            lines = None

        # 거리 정보 및 각도 정보 저장 레지스터
        d1 = []
        d2 = []
        grad = []

        # 차선이 감지된 경우
        if lines is not None and len(msg.line) != 0:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                val = np.array([[x1, y1], [x2, y2]])

                # y축 기준 내림차순 정렬
                val = val[val[:, 1].argsort()[::-1]]            
                
                # 거리 및 각도 계산 (각도 : 양수-\ | 음수-/)
                d1.append((val[0][0] - 0)**2 + (val[0][1] - 480)**2)
                d2.append((val[0][0] - 640)**2 + (val[0][1] - 480)**2)
                grad.append(np.arctan((y1 - y2)/(x1 - x2 + 1e-6)) * 180 / np.pi)

                # 선 삽입
                cv2.line(img_hough, (x1, y1), (x2, y2), 255, 1)

                # LRC 추출
                line_l, line_r, line_c = lrc_extractor(d1, d2, grad, lines)

        # 주차 공간이 감지된 경우
        elif lines is not None and len(msg.blank_mask) != 0:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                val = np.array([[x1, y1], [x2, y2]])

                # y축 기준 내림차순 정렬
                val = val[val[:, 1].argsort()[::-1]]
                
                # 거리 및 각도 계산 (각도 : 양수-\ | 음수-/)
                d1.append((val[1][0] - 0)**2 + (val[1][1] - 0)**2)
                d2.append((val[1][0] - 640)**2 + (val[1][1] - 0)**2)
                grad.append(np.arctan((y1 - y2)/(x1 - x2 + 1e-6)) * 180 / np.pi)

                # 선 삽입
                cv2.line(img_hough, (x1, y1), (x2, y2), 255, 1)

                # LRC 추출
                line_l, line_r, line_c = lrc_extractor(d1, d2, grad, lines)

        # 아무것도 감지되지 않은 경우
        else:
            line_l = []
            line_r = []
            line_c = []


        # 감지 결과 출력을 위한 String
        detection = ""

        # 좌우 차선의 길이를 평준화하기 위한 조건
        if line_l != [] and line_r != []:
            x1_l, y1_l, x2_l, y2_l = line_l
            x1_r, y1_r, x2_r, y2_r = line_r

            y_max = max(y1_l, y2_l, y1_r, y2_r)
            y_min = min(y1_l, y2_l, y1_r, y2_r)

            grad_l = (x2_l - x1_l)/(y2_l - y1_l + 1e-6)
            grad_r = (x2_r - x1_r)/(y2_r - y1_r + 1e-6)

            new_x1_l = grad_l*(y_max - y1_l) + x1_l
            new_x2_l = grad_l*(y_min - y1_l) + x1_l

            new_x1_r = grad_r*(y_max - y1_r) + x1_r
            new_x2_r = grad_r*(y_min - y1_r) + x1_r

            line_l = [int(new_x1_l), y_max, int(new_x2_l), y_min]
            line_r = [int(new_x1_r), y_max, int(new_x2_r), y_min]


        if line_l != []:
            x1, y1, x2, y2 = line_l
            cv2.line(img_hough_post, (x1, y1), (x2, y2), 255, 1)
            detection += "L"

        if line_r != []:
            x1, y1, x2, y2 = line_r
            cv2.line(img_hough_post, (x1, y1), (x2, y2), 255, 1) 
            detection += "R"

        if line_c != []:
            x1, y1, x2, y2 = line_c
            cv2.line(img_hough_post, (x1, y1), (x2, y2), 255, 1) 
            detection += "C"


        # 글자 길이 확인
        (_, h), _ = cv2.getTextSize(text = detection,
                                    fontFace = cv2.FONT_HERSHEY_COMPLEX, 
                                    fontScale=1,
                                    thickness=1)

        # 글자 삽입
        cv2.putText(img = img_hough_post,
                    text = detection,
                    org=[5, 5+h],
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1,
                    color=255,
                    thickness=1)     

        # 이미지 출력
        if DEBUG ==  True:
            img_concat = np.concatenate((img_base, img_hough, img_hough_post), axis=1)
            img_concat = cv2.resize(img_concat, (1280, 320))
            cv2.imshow("LINE", img_concat)
            cv2.waitKey(1)

        # 결과값 할당
        result.left = list(map(int, line_l))
        result.right = list(map(int, line_r))
        result.center = list(map(int, line_c))

        # 결과 Publish
        self.publisher.publish(result)


def lrc_extractor(d1, d2, grad, lines):
    line_l = []
    line_r = []
    line_c = []

    # 인덱스 추출
    d1_idx = d1.index(min(d1))
    d2_idx = d2.index(min(d2))
    grad_idx = grad.index(min(grad, key=abs))

    # 가운데 선 추출
    if abs(grad[grad_idx]) < 10:
        line_c.extend(lines[grad_idx][0])

    # 두 인덱스가 같을 경우 무시
    if d1_idx == d2_idx:
        pass

    else:
        # 두 인덱스에 대한 정보 추출
        x1_l, y1_l, x2_l, y2_l = lines[d1_idx][0]
        x1_r, y1_r, x2_r, y2_r = lines[d2_idx][0]

        # 중앙점 계산
        x_lc = (x1_l + x2_l)/2
        y_lc = (y1_l + y2_l)/2                
        x_rc = (x1_r + x2_r)/2
        y_rc = (y1_r + y2_r)/2

        # 두 선 사이의 이격 거리가 100 이상인 경우
        if np.sqrt((x_lc-x_rc)**2 + (y_lc-y_rc)**2) > 100:

            # 좌측 차선 추출
            if grad[d1_idx] < -20 and d1_idx != grad_idx:
                line_l.extend(lines[d1_idx][0])
    
            # 우측 차선 추출
            if grad[d2_idx] > 20 and d2_idx != grad_idx:
                line_r.extend(lines[d2_idx][0])

        # 두 선 사이의 이격 거리가 너무 가까운 경우
        else:
            # 좌측 차선인 경우
            if (grad[d1_idx] + grad[d2_idx])/2 < 0:
                line_l.extend(lines[d1_idx][0])

            # 우측 차선인 경우
            elif (grad[d1_idx] + grad[d2_idx])/2 > 0:
                line_r.extend(lines[d2_idx][0])

    return line_l, line_r, line_c


def main(args=None):
    rclpy.init(args=args)
    node = LineDetector()
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