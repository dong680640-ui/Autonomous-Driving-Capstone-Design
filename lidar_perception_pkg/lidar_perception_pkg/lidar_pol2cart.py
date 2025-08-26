import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

import math
import logging


## <Parameter> #######################################################################################

# 구독 토픽 이름
SUB_TOPIC_NAME = 'lidar_processed' 

# 발행 토픽 이름
PUB_TOPIC_NAME = 'lidar_cartesian'

# 로깅 여부
LOG = True

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


class lidar_pol2cart(Node):
    def __init__(self):
        super().__init__("lidar_pol2cart")

        # QOS 선언
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # Publisher / Subscriber 선언
        self.subscriber = self.create_subscription(LaserScan, SUB_TOPIC_NAME, self.conv_callback, self.qos_profile)
        self.publisher = self.create_publisher(Float32MultiArray, PUB_TOPIC_NAME, self.qos_profile) 

        # 로깅 여부 설정
        if LOG == False: 
            self.get_logger().set_level(logging.FATAL)


    def conv_callback(self, msg):
        # 수신받은 거리 데이터 추출
        ranges = msg.ranges
        coordinates = []

        # pol -> cart
        for k in range(len(ranges)):
            if abs(ranges[k]) != float('inf'):
                # X Coordinate (- 부호를 삽입하여 "<--x"를 "x-->"로 변환)
                coordinates.append(-math.cos(k*math.pi/180) * ranges[k])

                # Y Coordinate
                coordinates.append(math.sin(k*math.pi/180) * ranges[k])

        # 발행
        msg = Float32MultiArray()
        msg.data = coordinates
        self.publisher.publish(msg)
        self.get_logger().info("published")


def main():
    rclpy.init()
    lidar_pol2cart_node = lidar_pol2cart()

    rclpy.spin(lidar_pol2cart_node)

    lidar_pol2cart_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()