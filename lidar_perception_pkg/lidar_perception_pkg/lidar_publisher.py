import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

import numpy as np

import logging
import sys
import os

#### These modules are unused
# import tf2_ros
# import geometry_msgs.msg


## <Parameter> #######################################################################################

# 발행 토픽 이름
PUB_TOPIC_NAME_FOR_RAW = "lidar_raw"
PUB_TOPIC_NAME_FOR_PROCESSED = "lidar_processed"

# LIDAR 장치 주소
LIDAR_PORT = '/dev/ttyUSB0'

# 로깅 여부
LOG = True

# Lidar 정보 송신 주기
PERIOD = 0.01

######################################################################################################


class lidar_publisher(Node):
    def __init__(self):
        super().__init__('lidar_publisher')

        # QOS 설정
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # Publisher 선언
        self.publisher = self.create_publisher(LaserScan, PUB_TOPIC_NAME_FOR_RAW, self.qos_profile)
        self.publisher_processed = self.create_publisher(LaserScan, PUB_TOPIC_NAME_FOR_PROCESSED, self.qos_profile)

        # 공백 변수 선언
        self.lidar = None
        self.lidar_sensor_data_generator = None

        # 좌표 변환 모듈 선언 (Unused)
        # self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Timer 선언
        self.timer = self.create_timer(PERIOD, self.publish_lidar_data)
        
        # Lidar 초기화
        self.initialize_lidar()

        # 로깅 여부 설정
        if LOG == False: 
            self.get_logger().set_level(logging.FATAL)


    def initialize_lidar(self):
        # LIDAR 초기화
        try:
            self.lidar = rplidar.RPLidar(LIDAR_PORT)
            self.lidar_sensor_data_generator = self.lidar.iter_scans()

        except rplidar.RPLidarException as e:
            self.get_logger().error(f'Failed to initialize LIDAR: {e}')
            self.destroy_node()
            rclpy.shutdown()
    

    def reset_lidar(self):
        # LIDAR 재시작
        try:
            self.lidar.stop()
            self.lidar.stop_motor()
            self.lidar.disconnect()

        except rplidar.RPLidarException as e:
            self.get_logger().error(f'Failed to reset LIDAR: {e}')
        
        self.initialize_lidar()


    def publish_lidar_data(self):
        # 좌표 변환 모듈 관련 파라미터 (Unused)
        # transform = geometry_msgs.msg.TransformStamped()
        # transform.header.stamp = self.get_clock().now().to_msg()
        # transform.header.frame_id = 'base_link'
        # transform.child_frame_id = 'laser_frame'
        # transform.transform.translation.x = 0.0
        # transform.transform.translation.y = 0.0
        # transform.transform.translation.z = 0.0

        # 좌표 변환 데이터 전송 (Unused)
        # self.tf_broadcaster.sendTransform(transform)

        try:
            scan = next(self.lidar_sensor_data_generator)
            scan = np.array(scan)

            # 메시지 설정
            msg = LaserScan()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'laser_frame'  # frame id of your lidar sensor
            msg.angle_min = 0.0  # Minimum angle of the scan [rad]
            msg.angle_max = 2 * np.pi  # Maximum angle of the scan [rad]
            msg.angle_increment = 2 * np.pi / 360.0  # Angular distance between measurements [rad]
            msg.time_increment = 0.0  # Time between measurements [seconds]
            msg.scan_time = 0.1  # Time between scans [seconds]
            msg.range_min = 0.15  # Minimum range value [m]
            msg.range_max = 12.0  # Maximum range value [m]

            # Range / Intensity 변수 선언
            ranges = [float('inf')] * int((msg.angle_max - msg.angle_min) / msg.angle_increment)
            intensities = [0.0] * int((msg.angle_max - msg.angle_min) / msg.angle_increment)
            
            # 데이터 가공
            for measurement in scan:
                angle = np.radians(measurement[1])  # Convert to radians
                if msg.angle_min <= angle <= msg.angle_max:
                    index = int((angle - msg.angle_min) / msg.angle_increment)
                    if 0 <= index < len(ranges):
                        ranges[index] = measurement[2] / 1000.0  # Distance measurement
                        intensities[index] = measurement[0]  # Intensity measurement
            
            # 메시지에 데이터 삽입
            msg.ranges = ranges
            msg.intensities = intensities

            # 변환된 데이터 추출
            msg_proc = flip_lidar_data(msg, pivot_angle = 0) # pivot_angle = 0 ~ 359
            msg_proc = rotate_lidar_data(msg_proc, offset = 0) # offset = 0 ~ 359

            # 배포
            self.publisher.publish(msg) # 변환 전 데이터
            self.publisher_processed.publish(msg_proc) # 변환 후 데이터
            self.get_logger().info(f'published')

        # 오류 처리
        except StopIteration:
            self.get_logger().error('Failed to get lidar scan')
            return
        
        except rplidar.RPLidarException as e:
            self.get_logger().error(f'RPLidar exception: {e}')
            self.reset_lidar()
            
        except ValueError as e:
            self.get_logger().error(f'ValueError: {e}')
            self.reset_lidar()


    def __del__(self):
        # LIDAR 가동 정지
        try:
            if self.lidar:
                self.lidar.stop()
                self.lidar.stop_motor()
                self.lidar.disconnect()

        except rplidar.RPLidarException as e:
            self.get_logger().error(f'Failed to properly shutdown LIDAR: {e}')


def rotate_lidar_data(msg, offset=0):
    offset = int(offset)

    if offset < 0 or offset >= 360:
        raise ValueError('offset must be between 0 and 359')
    
    msg.ranges = msg.ranges[offset:] + msg.ranges[:offset]
    msg.intensities = msg.intensities[offset:] + msg.intensities[:offset]

    return msg


def flip_lidar_data(msg, pivot_angle):
    pivot_angle = int(pivot_angle)

    if pivot_angle < 0 or pivot_angle >= 360:
        raise ValueError('pivot_angle must be between 0 and 359')
    
    length = len(msg.ranges)
    flipped_ranges = [0] * length
    flipped_intensities = [0] * length

    for i in range(length):
        new_angle = (2 * pivot_angle - i) % length
        flipped_ranges[new_angle] = msg.ranges[i]
        flipped_intensities[new_angle] = msg.intensities[i]

    msg.ranges = flipped_ranges
    msg.intensities = flipped_intensities
    
    return msg


def main(args=None):
    rclpy.init(args=args)
    lidar_publisher_node = lidar_publisher()

    try:
        rclpy.spin(lidar_publisher_node)

    except KeyboardInterrupt:
        pass

    finally:
        lidar_publisher_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    #### Following module is provided by vendor
    #### https://github.com/Roboticia/RPLidar/blob/master/rplidar.py
    from lib import rplidar
    
    main()


else:
    #### Following module is provided by vendor
    #### https://github.com/Roboticia/RPLidar/blob/master/rplidar.py
    from .lib import rplidar

    main()