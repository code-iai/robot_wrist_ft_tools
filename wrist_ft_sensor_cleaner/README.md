# wrist_ft_sensor_cleaner
This package provides two nodes: ```ft_cleaner``` and ```gravity_publisher```.

```ft_cleaner``` is a tool to estimate and remove the force/torque offset as well as the weight and center of mass of objects attached to the sensor, 
e.g. a gripper, from the sensor readings using linear regression.

```gravity_publisher``` published the gravity vector in the sensors frame which is required by ```ft_cleaner```.

## Dependencies:
Common dependencies:
* python 2.7.3
* numpy
* rospy

```ft_cleaner```:
* scipy
* sklearn

```gravity_publisher```:
* PyKDL
* ros-indigo-kdl-parser-py

## Quickstart
Use 

```roslaunch wrist_ft_sensor_cleaner ft_cleaner_pr2.launch```

for the pr2 sensor or 

```roslaunch wrist_ft_sensor_cleaner ft_cleaner_ur5.launch```

for the ur5.

## ft_cleaner

### Parameters
* ```path_to_saved_params``` path to a yaml file in which the estimated parameters are stored
* ```ft_topic``` name of the raw input force/torque topic 
* ```acceleration_topic``` name of the acceleration topic

### Services
* ```~/recalibration_start``` [OUT, std_srvs/Trigger]: starts the calibration mode
* ```~/recalibration_stop``` [OUT, std_srvs/Trigger]: stops the calibration mode and saves the estimated parameters in ```$(param path_to_saved_params)```
* ```~/update_offset``` [OUT, std_srvs/Trigger]: reestimates the force/torque offset from a single measurement, based on the assumption that the attached mass and center of mass have not changed and saves the estimated parameters in ```$(param path_to_saved_params)```

### Topics
* ```/joint_states``` [IN, sensor_msgs/JointState]: used to determin if the robot is in motion
* ```$(param ft_topic)``` [IN, geometry_msgs/WrenchStamped]: raw input force/torque topic
* ```$(param acceleration_topic)``` [IN, sensor_msgs/Imu]: has to provide the acceleration of the ft sensor in the sensors frame and its orientation in a fixed frame
* ```~/status``` [OUT, wrist_ft_sensor_cleaner/FTRecalibrationInfo]: during the calibration mode, the current parameters as well as two quality measures are published
* ```$(param ft_topic)_clean``` [OUT, geometry_msgs/WrenchStamped]: ```$(param ft_topic)``` with offset removed
* ```$(param ft_topic)_zeroed``` [OUT, geometry_msgs/WrenchStamped]: ```$(param ft_topic)``` with offset and weight of attached mass removed

### How to calibrate
If mass and center of mass in params.yaml are not correct:
* Call ```/recalibration_start``` to put the node into calibration mode.
* Move sensor into different orientations that are as diverse as possible.
* Keep the robot motionless until a message is published on ```~/status```.
* Try to maximize ```score1``` and ```score2``` from ```~/status```.
* Get at least 7 measurements.
* Stop using ```/recalibration_stop```.

While in calibration mode, the sensor has to be moved into different orientations that are as diverse as possible. 
This can be done by hand, just take care that you do not touch the sensor or the mass attached to it when the node takes a new measurement!
New measurements for the parameter estimation will be recorded if the robot is not in motion for 1 second.
After each measurement the new parameter estimate will be published on ```~status```.

```score1``` shows how evenly the sensor orientation durings the measurements are distributed in the quaternion space.
Values above 0.8 after 7 measurements generally result in decent estimates. The maximum value depends on the number of measurements. 
It starts by 2 and converges to 1 as the number of measurements increase.

```score2``` tests how good the estimated parameters predict the measurements used for the model training.
A value that is far below 1 can indicate high levels of noise, wrong data from the acceleration topic, that offset/mass/center of mass 
have changed (for example due to contacts) during the recalibration or that something is wrong with the sensor.

If mass and center of mass in params.yaml are correct:
* Call ```/update_offset```.

This service updates the offsets as soon as the robot stays still for one second.

## gravity_publisher

### Parameters
* ```urdf``` Path to the robots urdf, if no path is provided, the urdf in ```robot_description``` will be loaded.
* ```start_link``` Start link of the kinematic chain to the ft sensor. The orientation of this frame should not change during the calibration.
* ```end_link``` Link of the ft sensor.
* ```gravity_vector``` gravity vector in the ```start_link``` frame.

### Topics
* ```/joint_states``` [IN, sensor_msgs/JointState]: used to calculate the forward kinematic using KDL
* ```/acceleration``` [OUT, sensor_msgs/Imu]: linear_acceleration contains the gravity in the ft sensor frame and orientation contains the sensors orientation in the start links frame

