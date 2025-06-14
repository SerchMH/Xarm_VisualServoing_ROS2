from setuptools import find_packages
from setuptools import setup

setup(
    name='xarm_msgs',
    version='0.0.0',
    packages=find_packages(
        include=('xarm_msgs', 'xarm_msgs.*')),
)
