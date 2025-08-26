from setuptools import find_packages, setup

package_name = 'camera_perception_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_publisher = camera_perception_pkg.image_publisher:main',
            'yolov8 = camera_perception_pkg.yolov8:main',
            'yolov8_for_crosswalk = camera_perception_pkg.yolov8_for_crosswalk:main',
            'image_publisher_rear = camera_perception_pkg.image_publisher_rear:main',
            'yolov8_rear = camera_perception_pkg.yolov8_rear:main',
            'car_info_extractor = camera_perception_pkg.car_info_extractor:main',
            'car_info_extractor_rear = camera_perception_pkg.car_info_extractor_rear:main',
            'traffic_light_detector = camera_perception_pkg.traffic_light_detector:main',
            'lane_info_extractor = camera_perception_pkg.lane_info_extractor:main',
            'line_info_extractor = camera_perception_pkg.line_info_extractor:main',
            'depth_extractor = camera_perception_pkg.depth_extractor:main',
        ],
    },
)
