# Distributed optimization problems

## Introduction

Project for the Distributed Autonomous Systems course at the University of Bologna (A.Y. 2024-2025).


## Description

Implementation of the tasks of multi-robot target localization and multi-robot positioning.


## Installation

To install the required Python libraries, run from the `src` directory the following command:
```
pip install -r requirements.txt
```

To build the ROS2 (Humble Hawksbill) package, run from the `src/task2/ros2` directory the following commands:
```
. /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select aggregative
```


## Execution

The experiment scripts only run the portions of code that have been indicated by some input flags. The available flags can be checked with:
```
python src/task1/main_quadratic.py --help
python src/task1/main_tracking.py --help
python src/task2/main.py --help
```

To run the ROS2 package, run the following commands from the `src/task2/ros2` directory:
```
. /opt/ros/humble/setup.bash
. install/setup.bash
ros2 launch aggregative launch.py
```