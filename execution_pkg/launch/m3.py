#!/usr/bin/env python3

import os
import subprocess
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
            
    return LaunchDescription([     
        ###################################################
        
        # camera_perception_pkg <Start>                 
        #Node(
        #    package='camera_perception_pkg', 
        #    executable='image_publisher',
        #    output='screen'
        #),       

        #Node(
        #    package='camera_perception_pkg', 
        #    executable='image_publisher_rear',
        #    output='screen'
        #),       

        #Node(
        #    package='camera_perception_pkg', 
        #    executable='yolov8',
        #    output='screen'
        #),

        Node(
            package='camera_perception_pkg', 
            executable='yolov8_rear',
            output='screen'
        ),

        #Node(
        #    package='camera_perception_pkg', 
        #    executable='car_info_extractor',
        #    output='screen'
        #),
        
        Node(
            package='camera_perception_pkg', 
            executable='car_info_extractor_rear',
            output='screen'
        ),        
        
        Node(
            package='camera_perception_pkg', 
            executable='line_info_extractor',
            output='screen'
        ),       
        # camera_perception_pkg <End>
        
        ###################################################
        
        # lidar_perception_pkg <Start>
        Node(
            package='lidar_perception_pkg', 
            executable='lidar_publisher',
            output='screen'
        ),

        Node(
            package='lidar_perception_pkg', 
            executable='lidar_object_detector',
            output='screen'
        ),
        
         Node(
            package='lidar_perception_pkg', 
            executable='lidar_pol2cart',
            output='screen'
        ),
        # lidar_perception_pkg <End>		

        ###################################################
        
        # debug_pkg <Start>
        #Node(
        #    package='debug_pkg', 
        #    executable='data_debugger',
        #    output='screen'
        #),
                
        Node(
            package='debug_pkg', 
            executable='lidar_debugger',
            output='screen'
        ),
        # debug_pkg <End>

        ###################################################
        
        # decision_making_pkg <Start>
        Node(
            package='decision_making_pkg', 
            executable='motion_planner',
            output='screen'
        ),
        # decision_making_pkg <End>

        ###################################################  

        # serial_communication_pkg <Start>
        Node(
            package='serial_communication_pkg', 
            executable='serial_communicator',
            output='screen'
        ),
         # serial_communication_pkg <End>

        ###################################################   
    ])
