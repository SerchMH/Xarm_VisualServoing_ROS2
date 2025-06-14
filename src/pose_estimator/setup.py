from setuptools import find_packages, setup

package_name = 'pose_estimator'

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
    maintainer='alex',
    maintainer_email='alex@todo.todo',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'match_pointcloud = pose_estimator.pc_matcher:main',
            'xarm_pose_controller = pose_estimator.pose_controller:main',
            'xarm_xy_mover = pose_estimator.xarm_mover:main',
        ],
    },
)
