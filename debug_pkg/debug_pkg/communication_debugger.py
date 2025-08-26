### 시험용 ###
# serial_communicator node에 신호를 전송하기 위한 코드

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8MultiArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import random

## <Parameter> #####################################################################################

# 노드 이름
NODE_NAME = "communication_debugger"

# 발행 토픽 이름
TOPIC_NAME = "command_data"

# 주기
TIMER_PERIOD = 0.1

######################################################################################################

class communication_debugger(Node):
    def __init__(self, node_name, topic_name, timer_period):
        super().__init__(node_name)

        self.qos = QoSProfile( # QOS 설정
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
                )
               
        self.publisher = self.create_publisher(Int8MultiArray, topic_name, self.qos)
        self.timer =self.create_timer(timer_period, self.send_callback)

        self.msg = Int8MultiArray()

    def send_callback(self):

        # 0~255 (8 bit, 1 Byte)
        steer_angle = int(random.random()*255-128) # -128 ~ 127
        left_speed = int(random.random()*255-128)  # -128 ~ 127
        right_speed = int(random.random()*255-128) # -128 ~ 127

        self.msg.data = [steer_angle, left_speed, right_speed]

        self.publisher.publish(self.msg)
        self.get_logger().info(f"{self.msg.data}")


def main():
    rclpy.init()
    debugger_node = communication_debugger(NODE_NAME, TOPIC_NAME, TIMER_PERIOD)
    rclpy.spin(debugger_node)

    debugger_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()