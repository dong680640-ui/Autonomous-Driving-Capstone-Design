from setuptools import find_packages, setup

package_name = 'debug_pkg'

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
            'yolo_debugger = debug_pkg.yolo_debugger:main',
            'communication_debugger = debug_pkg.communication_debugger:main',
            'lidar_debugger = debug_pkg.lidar_debugger:main',
            'data_debugger = debug_pkg.data_debugger:main'
        ],
    },
)
