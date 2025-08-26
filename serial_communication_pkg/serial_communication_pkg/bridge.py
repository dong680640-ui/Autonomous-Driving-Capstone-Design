import rclpy
from rclpy.node import Node
from interfaces_pkg.msg import MotionCommand
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy


## <Parameter> #######################################################################################

# 노드 이름
NODE_NAME = "bridge"

# 발행 토픽 이름
TOPIC_NAME = "command_data"

# 구독 토픽 이름
SUB_TOPIC_NAME = "command_pre_data"

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


class bridge_node(Node):
    def __init__(self, node_name, topic_name : list, sub_topic_name):
        super().__init__(node_name)

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

        # Publisher 선언
        self.publisher = self.create_publisher(MotionCommand, topic_name, self.qos_pub) 

        # Subscriber 선언
        self.subscriber = self.create_subscription(MotionCommand, sub_topic_name, self.send_callback, self.qos_sub)


    def send_callback(self, msg):
        # Publishing을 위한 Message 선언
        self.publisher.publish(msg)      


def main():
    rclpy.init()
    bridge_node_param = bridge_node(NODE_NAME, TOPIC_NAME, SUB_TOPIC_NAME)
    rclpy.spin(bridge_node_param)

    bridge_node_param.destroy_node()
    rclpy.shutdown()


if __name__== "__main__":
    main() 