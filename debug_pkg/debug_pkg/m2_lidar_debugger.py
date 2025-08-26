import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from interfaces_pkg.msg import BoolMultiArray

from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

import cv2
import numpy as np

import math


## <Parameter> #######################################################################################

# 구독 토픽 이름
SUB_TOPIC_NAME_CART = 'lidar_cartesian'
SUB_TOPIC_NAME_TF = 'lidar_data'

# 출력 화면 크기 (가로, 세로)
SIZE = [900, 900]

# 좌측
MIN_L = 0.3 # 최소 거리 [m]
MAX_L = 1.0 # 최대 거리 [m]
MIN_ANGLE_L = -30 # 최소 각도
MAX_ANGLE_L = -5 # 최대 각도

# 우측
MIN_R = 0.3 # 최소 거리 [m]
MAX_R = 1.0 # 최대 거리 [m]
MIN_ANGLE_R = 185 # 최소 각도
MAX_ANGLE_R = 210 # 최대 각도

# 중앙
MIN_C = 0.0 # 최소 거리 [m]
MAX_C = 1.0 # 최대 거리 [m]
MIN_ANGLE_C = -90 # 최소 각도
MAX_ANGLE_C = -90 # 최대 각도

# 확장 계수
K = 200

######################################################################################################


## <LIDAR> ###########################################################################################
#        270
#       #######  (Motor)
#     0 # 본체 ######### 180     (Counter-Clockwise)
#       #######
#         90
#
#
#       <--x    y      - 부호      y  x-->
#               |      ---->>     |
#               v                 v
######################################################################################################


class lidar_debugger(Node):
    def __init__(self):
        super().__init__("lidar_debugger")

        # QOS 선언
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # Publisher / Subscriber 선언
        self.subscriber_cart = self.create_subscription(Float32MultiArray, SUB_TOPIC_NAME_CART, self.disp_callback, self.qos_profile)
        self.subscriber_tf = self.create_subscription(BoolMultiArray, SUB_TOPIC_NAME_TF, self.tf_callback, self.qos_profile)

        # Bool 저장을 위한 레지스터 선언
        self.bool = [False, False, False]


    def tf_callback(self, msg):
        self.bool = msg.data


    def disp_callback(self, msg):
        msg = np.array(msg.data).reshape(-1, 2).tolist()
        background = np.zeros([SIZE[1], SIZE[0]])

        center_x = int(SIZE[0]/2)
        center_y = int(SIZE[1]/2)

        for x, y in msg:
            if int(y*K + center_y) >= 0 and int(x*K + center_x) >= 0:
                try:
                    background[int(y*K + center_y)][int(x*K + center_x)] = 1
                except:
                    pass

        # Left Side
        cv2.circle(background, [center_x, center_y], int(MIN_L*K), 255, thickness=1)
        cv2.circle(background, [center_x, center_y], int(MAX_L*K), 255, thickness=1)

        cv2.line(img = background, 
                 pt1 = [center_x, center_y],
                 pt2 = [int((-math.cos(MIN_ANGLE_L*math.pi/180)*SIZE[0] + center_x)),
                        int((math.sin(MIN_ANGLE_L*math.pi/180)*SIZE[1] + center_y))], 
                 color = 255, 
                 thickness=1) 

        cv2.line(img = background, 
                 pt1 = [center_x, center_y],
                 pt2 = [int((-math.cos(MAX_ANGLE_L*math.pi/180)*SIZE[0] + center_x)),
                        int((math.sin(MAX_ANGLE_L*math.pi/180)*SIZE[1] + center_y))], 
                 color = 255, 
                 thickness=1) 
        
        if self.bool[0] == True:
            (_, h), _ = cv2.getTextSize(text = "Detected",
                                        fontFace = cv2.FONT_HERSHEY_COMPLEX, 
                                        fontScale=1,
                                        thickness=1)

            cv2.putText(img = background,
                        text = "Detected",
                        org=[5, 5+h],
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1,
                        color=255,
                        thickness=1)


        # Right Side
        cv2.circle(background, [center_x, center_y], int(MIN_R*K), 255, thickness=1)
        cv2.circle(background, [center_x, center_y], int(MAX_R*K), 255, thickness=1)

        cv2.line(img = background, 
                 pt1 = [center_x, center_y],
                 pt2 = [int((-math.cos(MIN_ANGLE_R*math.pi/180)*SIZE[0] + center_x)),
                        int((math.sin(MIN_ANGLE_R*math.pi/180)*SIZE[1] + center_y))], 
                 color = 255, 
                 thickness=1) 

        cv2.line(img = background, 
                 pt1 = [center_x, center_y],
                 pt2 = [int((-math.cos(MAX_ANGLE_R*math.pi/180)*SIZE[0] + center_x)),
                        int((math.sin(MAX_ANGLE_R*math.pi/180)*SIZE[1] + center_y))], 
                 color = 255, 
                 thickness=1) 

        if self.bool[1] == True:
            (w, h), _ = cv2.getTextSize(text = "Detected",
                                        fontFace = cv2.FONT_HERSHEY_COMPLEX, 
                                        fontScale=1,
                                        thickness=1)

            cv2.putText(img = background,
                        text = "Detected",
                        org=[SIZE[0] - w - 5, 5+h],
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1,
                        color=255,
                        thickness=1)


        # Center Side
        cv2.circle(background, [center_x, center_y], int(MIN_C*K), 255, thickness=1)
        cv2.circle(background, [center_x, center_y], int(MAX_C*K), 255, thickness=1)

        cv2.line(img = background, 
                 pt1 = [center_x, center_y],
                 pt2 = [int((-math.cos(MIN_ANGLE_C*math.pi/180)*SIZE[0] + center_x)),
                        int((math.sin(MIN_ANGLE_C*math.pi/180)*SIZE[1] + center_y))], 
                 color = 255, 
                 thickness=1) 

        cv2.line(img = background, 
                 pt1 = [center_x, center_y],
                 pt2 = [int((-math.cos(MAX_ANGLE_C*math.pi/180)*SIZE[0] + center_x)),
                        int((math.sin(MAX_ANGLE_C*math.pi/180)*SIZE[1] + center_y))], 
                 color = 255, 
                 thickness=1) 

        if self.bool[2] == True:
            (w, h), _ = cv2.getTextSize(text = "Detected",
                                        fontFace = cv2.FONT_HERSHEY_COMPLEX, 
                                        fontScale=1,
                                        thickness=1)

            cv2.putText(img = background,
                        text = "Detected",
                        org=[int(SIZE[0]/2 - w/2), 5+h],
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=1,
                        color=255,
                        thickness=1)




        cv2.imshow("LIDAR", background)
        cv2.waitKey(5)


def main():
    rclpy.init()
    lidar_debugger_node = lidar_debugger()

    rclpy.spin(lidar_debugger_node)

    lidar_debugger_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
