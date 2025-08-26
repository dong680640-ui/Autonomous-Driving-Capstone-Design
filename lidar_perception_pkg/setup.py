from setuptools import find_packages, setup

package_name = 'lidar_perception_pkg'

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
            'lidar_publisher = lidar_perception_pkg.lidar_publisher:main',
			'lidar_object_detector = lidar_perception_pkg.lidar_object_detector:main',
   			'lidar_pol2cart = lidar_perception_pkg.lidar_pol2cart:main',
        ],
    },
)
