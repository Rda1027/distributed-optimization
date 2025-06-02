from setuptools import find_packages, setup
from glob import glob

package_name = 'aggregative'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ("share/" + package_name, glob("launch/launch.py")),
        ("share/" + package_name, glob("resource/rviz_config.rviz"))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Group 6',
    maintainer_email='',
    description='Aggregative optimization',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f"agent_aggregative = {package_name}.Agent:main",
            f"scenario_visualizer = {package_name}.Visualizer:main"
        ],
    },
)
