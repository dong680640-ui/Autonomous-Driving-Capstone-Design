import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from interfaces_pkg.msg import LaneData, SegmentGroup

import cv2
from cv_bridge import CvBridge

import numpy as np

import os, sys
import logging

## <Parameter> #######################################################################################

# Subscribe할 토픽 이름
SUB_TOPIC_NAME = "segmented_data"

# Publish할 토픽 이름
PUB_TOPIC_NAME = "lane_data"

# 화면에 이미지를 처리하는 과정을 띄울것인지 여부: True, 또는 False 중 택1하여 입력
SHOW_IMAGE = True

# 영상 크기 (가로, 세로)
FRAME_SIZE = [640, 480]

# 로깅 여부
LOG = True

######################################################################################################


class lane_info_extractor(Node):
    def __init__(self):
        super().__init__('lane_info_extractor')

        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.show_image = self.declare_parameter('show_image', SHOW_IMAGE).value

        self.cv_bridge = CvBridge()

        # QoS 설정
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # 구독 설정
        self.subscriber = self.create_subscription(SegmentGroup, self.sub_topic, self.yolov8_detections_callback, self.qos_profile)
        self.publisher = self.create_publisher(LaneData, self.pub_topic, self.qos_profile)

        # 로깅 여부 설정
        if LOG == False: 
            self.get_logger().set_level(logging.FATAL)


    def yolov8_detections_callback(self, msg:SegmentGroup):
        # Lane 정보 수신
        lane1_masks = np.array(msg.lane_1, dtype=int).reshape(-1, 2) 
        lane2_masks = np.array(msg.lane_2, dtype=int).reshape(-1, 2)

        # 정보가 없을 경우 반환
        if len(msg.lane_1) == 0 and len(msg.lane_2) == 0:
            return

        # 공백 이미지 생성 
        lane1_edge_image = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0]))
        lane2_edge_image = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0]))

        # 도로 선 그리기
        cv2.polylines(lane1_edge_image, [lane1_masks], isClosed=True, color=255, thickness=1, lineType=cv2.LINE_AA)
        cv2.polylines(lane2_edge_image, [lane2_masks], isClosed=True, color=255, thickness=1, lineType=cv2.LINE_AA)

        # 변환 행렬 선언
        src_mat = [[227, 323],
                   [407, 323],
                   [487, 440],
                   [160, 440]]
        
        dst_mat = [[round(FRAME_SIZE[0] * 0.3), round(FRAME_SIZE[1] * 0.0)],
                   [round(FRAME_SIZE[0] * 0.7), round(FRAME_SIZE[1] * 0.0)], 
                   [round(FRAME_SIZE[0] * 0.7), round(FRAME_SIZE[1] * 1.0)],
                   [round(FRAME_SIZE[0] * 0.3), round(FRAME_SIZE[1] * 1.0)]]

        
        # 행렬 변환 연산
        lane1_bird_image = bird_view_converter(lane1_edge_image, srcmat=src_mat, dstmat=dst_mat)
        lane2_bird_image = bird_view_converter(lane2_edge_image, srcmat=src_mat, dstmat=dst_mat)

        # ROI 추출
        roi_image1 = roi_extractor(lane1_bird_image, cutting_idx=300)
        roi_image2 = roi_extractor(lane2_bird_image, cutting_idx=300)

        # 화면 출력
        if self.show_image:
            #cv2.imshow('ORIGINAL', np.concatenate((lane1_edge_image, lane2_edge_image), axis=1))
            #cv2.imshow('BIRD', np.concatenate((lane1_bird_image, lane2_bird_image), axis=1))
            cv2.imshow('ROI', np.concatenate((roi_image1, roi_image2), axis=1))

            #cv2.imshow('LANE', np.concatenate((np.concatenate((lane1_edge_image, lane2_edge_image), axis=1), np.concatenate((roi_image1, roi_image2), axis=1)), axis=0))
            
            cv2.waitKey(1)

        # 기울기 추출
        grad1 = dominant_gradient(roi_image1, theta_limit=70)
        grad2 = dominant_gradient(roi_image2, theta_limit=70)

        # 중심점 추출
        lane1_point_y = 450
        lane1_point_x = get_lane_center(roi_image1, detection_height=lane1_point_y-300, 
                              detection_thickness=10, lane_width=300)
        lane2_point_y = 450
        lane2_point_x = get_lane_center(roi_image2, detection_height=lane2_point_y-300, 
                              detection_thickness=10, lane_width=300)

        # Message 생성 및 전송
        lane = LaneData()

        # slope (None일 경우 기울기는 0으로 상정됨)
        if grad1 is not None:
            lane.slope1 = grad1
    
        if grad2 is not None:
            lane.slope2 = grad2

        # lane1_x
        if lane1_point_x is not None:
            lane.lane1_x = round(lane1_point_x)
        else:
            lane.lane1_x = 0 # Lane1이 인식되지 않을 경우 왼쪽 끝으로 고정

        # lane1_y
        if lane1_point_y is not None:
            lane.lane1_y = round(lane1_point_y)

        # lane2_x
        if lane2_point_x is not None:
            lane.lane2_x = round(lane2_point_x)
        else:
            lane.lane2_x = FRAME_SIZE[0] - 1 # Lane2가 인식되지 않을 경우 오른쪽 끝으로 고정

        # lane2_y
        if lane2_point_y is not None:
            lane.lane2_y = round(lane2_point_y)

        self.get_logger().info(f"{grad1}, {grad2}")
        self.get_logger().info(f"({lane1_point_x}, {lane1_point_y}), ({lane2_point_x}, {lane2_point_y})")

        self.publisher.publish(lane)

    
def bird_view_converter(img, srcmat, dstmat):
        srcmat = np.float32(srcmat)
        dstmat = np.float32(dstmat)
        transform_mat = cv2.getPerspectiveTransform(srcmat, dstmat)

        img = cv2.warpPerspective(img, transform_mat, (img.shape[1], img.shape[0]))

        return img


def roi_extractor(img, cutting_idx):
    return img[cutting_idx:]
 

def dominant_gradient(image, theta_limit):
    right_limit_radian = np.deg2rad(90 + (90 - theta_limit))
    left_limit_radian = np.deg2rad(90 - (90 - theta_limit))
    _, width = (image.shape[0], image.shape[1])
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    image_original = image.copy()
    try:
        lines = cv2.HoughLines(image, 1, np.pi / 180, int(width * 0.0390625))
        angles = []
        if lines is not None:
            for line in lines:
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * -b)
                    y1 = int(y0 + 1000 * a)
                    x2 = int(x0 - 1000 * -b)
                    y2 = int(y0 - 1000 * a)
                    if theta < right_limit_radian and theta > left_limit_radian:
                        continue
                    angle = np.arctan((x2 - x1) / (y1 - y2)) * 180 / np.pi
                    angles.append(angle)
                    cv2.line(image_original, (x1, y1), (x2, y2), (255, 255, 255))
        if len(angles) == 0:
            result = 0.0
            return result
        result = np.median(angles)
        return result
    except Exception as e:
        _, _, tb = sys.exc_info()
        print(f'gradient detection error = {e}, error line = {tb.tb_lineno}')
        exception_image_path = './exception_image/'
        try:
            if not os.path.exists(exception_image_path):
                os.mkdir(exception_image_path)
        except OSError:
            print('Error: Creating directory. ' + exception_image_path)
        return (0, None)


def get_lane_center(cv_image: np.array, detection_height: int, detection_thickness: int, lane_width: int) -> int:
        detection_area_upper_bound = detection_height - int(detection_thickness/2)
        detection_area_lower_bound = detection_height + int(detection_thickness/2)

        detected_x_coords = np.sort(np.where(cv_image[detection_area_upper_bound:detection_area_lower_bound,:]!=0)[1])

        if (detected_x_coords.shape[0] < 5):
            line_x_axis_pixel = None
            center_pixel = None
            return None

        cut_outliers_array = detected_x_coords[1:-1]
        difference_array = cut_outliers_array[1:] - cut_outliers_array[:-1]

        max_diff_idx_left = np.argmax(difference_array)
        max_diff_idx_right = np.argmax(difference_array)+1
        left_val = cut_outliers_array[max_diff_idx_left]
        right_val = cut_outliers_array[max_diff_idx_right]

        if abs(left_val - right_val) < (lane_width/3):
            line_x_axis_pixel = cut_outliers_array[round((cut_outliers_array.shape[0])/2)]
            center_pixel = None
        else:
            line_x_axis_pixel = None
            center_pixel = (left_val + right_val)/2

        if center_pixel == None and line_x_axis_pixel == None:
            road_target_point_x = None
        else:
            road_target_point_x = center_pixel
            if road_target_point_x == None and line_x_axis_pixel != None:
                if cut_outliers_array[-1] > 540:
                    road_target_point_x = line_x_axis_pixel + (lane_width/2)
                    if road_target_point_x < (639-lane_width):
                        road_target_point_x = (639-lane_width)
                    elif road_target_point_x > 639:
                        road_target_point_x = 639
                elif cut_outliers_array[0] < 100:
                    road_target_point_x = line_x_axis_pixel - (lane_width/2)
                    if road_target_point_x > (lane_width-1):
                        road_target_point_x = (lane_width-1)
                    elif road_target_point_x < 0:
                        road_target_point_x = 0
                else:
                    road_target_point_x = left_val + int((left_val - right_val) / 2)
                    
        return road_target_point_x


def main(args=None):
    rclpy.init(args=args)
    node = lane_info_extractor()
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