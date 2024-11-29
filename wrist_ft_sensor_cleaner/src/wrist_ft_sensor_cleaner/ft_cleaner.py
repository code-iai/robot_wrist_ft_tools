#!/usr/bin/env python3
from collections import deque

import numpy as np
import yaml
import rospy
import scipy
from geometry_msgs.msg import Vector3, WrenchStamped
from rospy.topics import Publisher, Subscriber
from sensor_msgs.msg import Imu, JointState
from sklearn.linear_model import LinearRegression
from std_msgs.msg import Header
from std_srvs.srv import Trigger, TriggerResponse
from wrist_ft_sensor_cleaner.msg import FTRecalibrationInfo
from pr2_msgs.msg import AccelerometerState


class FTCleaner:
    def __init__(self, ft_topic='/ft/l_gripper_motor',
                 acceleration_topic='/acceleration',
                 path_to_saved_params='params.yaml',
                 time_till_record=1.0,
                 not_moving_threshold=0.2):
        """
        :param ft_topic: The name of the force/torque sensor topic.
        :type ft_topic: str, default '/ft/l_gripper_motor'
        :param time_till_record: The time waited from a movement stop until a measurement will be made in seconds.
        :type time_till_record: float, default 1.0
        :param not_moving_threshold: if the euclidean norm of the joint state velocity vector is above this value,
                the robot is considered to be in motion.
        :type not_moving_threshold: float
        :param path_to_saved_params: The path yaml file containing previously estimated values.
        :type path_to_saved_params: str, default ''
        """
        self.in_recalibration_mode = False
        self.path_to_saved_params = path_to_saved_params
        self.time_till_record = time_till_record
        self.not_moving_threshold = not_moving_threshold
        self.reset()
        self.joints_to_ft_ids = None
        self.time_of_last_acc_msg = rospy.Time(0, 0)
        self.last_linear_acceleration = np.zeros(3)
        self.last_orientation = np.zeros(4)
        self.offset = np.zeros(6)
        self.m = 0
        self.c = np.zeros(3)
        self.cm = np.zeros(3)
        self.load_params(self.path_to_saved_params)
        self.acceleration = Subscriber(acceleration_topic, Imu, self.acc_cb, queue_size=100)
        self.ft_clean_pub = Publisher(f"{ft_topic}_clean", WrenchStamped, queue_size=100)
        self.ft_zeroed_pub = Publisher(f"{ft_topic}_zeroed", WrenchStamped, queue_size=100)
        self.status_pub = Publisher("~status", FTRecalibrationInfo, queue_size=100)
        self.joint_state_sub = rospy.Subscriber('joint_states', JointState, self.joint_state_cb, queue_size=100)
        self.recalibrate_srv_start = rospy.Service('~recalibration_start', Trigger, self.start_calibration)
        self.recalibrate_srv_stop = rospy.Service('~recalibration_stop', Trigger, self.stop_calibration)
        self.recalibrate_srv_update_offset = rospy.Service('~update_offset', Trigger, self.update_offset_cb)

        self.clean_buffer = deque(maxlen=10)
        self.zeroed_buffer = deque(maxlen=10)
        self.clean_derivative_buffer = deque(maxlen=10)
        self.zeroed_derivative_buffer = deque(maxlen=10)
        self.prev_clean_ft = None
        self.prev_zeroed_ft = None
        self.prev_time = None
        self.ft_clean_avg_pub = Publisher(f"{ft_topic}_clean_avg", WrenchStamped, queue_size=100)
        self.ft_zeroed_avg_pub = Publisher(f"{ft_topic}_zeroed_avg", WrenchStamped, queue_size=100)
        self.ft_clean_derivative_pub = Publisher(f"{ft_topic}_clean_derivative", WrenchStamped, queue_size=100)
        self.ft_zeroed_derivative_pub = Publisher(f"{ft_topic}_zeroed_derivative", WrenchStamped, queue_size=100)
        self.ft_clean_derivative_avg_pub = Publisher(f"{ft_topic}_clean_derivative_avg", WrenchStamped, queue_size=100)
        self.ft_zeroed_derivative_avg_pub = Publisher(f"{ft_topic}_zeroed_derivative_avg", WrenchStamped,
                                                      queue_size=100)

        self.ft_sub = Subscriber(ft_topic, WrenchStamped, self.ft_cb, queue_size=100)

        rospy.loginfo(f"acceleration topic: {acceleration_topic}")
        rospy.loginfo(f"ft_cleaner for {ft_topic} started")

    def acc_cb(self, data):
        self.time_of_last_acc_msg = data.header.stamp
        self.last_linear_acceleration = np.array([data.linear_acceleration.x,
                                                  data.linear_acceleration.y,
                                                  data.linear_acceleration.z])
        self.last_orientation = [data.orientation.x,
                                 data.orientation.y,
                                 data.orientation.z,
                                 data.orientation.w]

    def update_offset_cb(self, data):
        r = TriggerResponse()
        if not self.in_recalibration_mode:
            self.update_offset = True
            r.success = True
        else:
            r.success = False
        return r

    def reset(self):
        """
        Clears previous measurements, should be called before a new calibration.
        """
        self.recorded = False
        self.in_recalibration_mode = False
        self.update_offset = False
        self.number_of_measurements = 0
        self.last_movement = np.inf
        self.parameter_estimator = ParameterEstimator()

    def start_calibration(self, data):
        r = TriggerResponse()
        if self.in_recalibration_mode:
            rospy.logwarn('Already in recalibration mode.')
            r.message = 'Already in recalibration mode.'
            r.success = False
        else:
            rospy.loginfo('recalibration started')
            self.reset()
            self.in_recalibration_mode = True
            r.success = True
        return r

    def stop_calibration(self, data):
        r = TriggerResponse()
        if self.in_recalibration_mode:
            self.in_recalibration_mode = False
            print("########################################")
            print('recalibration stopped')
            print("########################################")
            self.save_params(self.path_to_saved_params)
            self.print_params(self.parameter_estimator.score(), self.parameter_estimator.score2)
            r.success = True
        else:
            r.success = False
            r.message = 'Not in recalibration mode.'
        return r

    @staticmethod
    def calculate_moving_average(buffer):
        """
        Calculates the moving average of a deque of WrenchStamped messages.
        :param buffer: Deque containing the last N WrenchStamped messages.
        :return: WrenchStamped with averaged values.
        """
        if not buffer:
            return None

        forces = np.array([[msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z] for msg in buffer])
        torques = np.array([[msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z] for msg in buffer])

        avg_wrench = WrenchStamped()
        avg_wrench.header = buffer[-1].header
        avg_force = forces.mean(axis=0)
        avg_torque = torques.mean(axis=0)

        avg_wrench.wrench.force.x, avg_wrench.wrench.force.y, avg_wrench.wrench.force.z = avg_force
        avg_wrench.wrench.torque.x, avg_wrench.wrench.torque.y, avg_wrench.wrench.torque.z = avg_torque

        return avg_wrench

    @staticmethod
    def calculate_derivative(current, previous, current_time, previous_time):
        """
        Computes the derivative of a WrenchStamped message.
        :param current: Current WrenchStamped message.
        :param previous: Previous WrenchStamped message.
        :param current_time: Current timestamp.
        :param previous_time: Previous timestamp.
        :return: WrenchStamped object representing the derivative, or None if not enough data.
        """
        if not previous or not previous_time or current_time <= previous_time:
            return None

        dt = current_time - previous_time

        force_diff = [
            (current.wrench.force.x - previous.wrench.force.x) / dt,
            (current.wrench.force.y - previous.wrench.force.y) / dt,
            (current.wrench.force.z - previous.wrench.force.z) / dt,
        ]
        torque_diff = [
            (current.wrench.torque.x - previous.wrench.torque.x) / dt,
            (current.wrench.torque.y - previous.wrench.torque.y) / dt,
            (current.wrench.torque.z - previous.wrench.torque.z) / dt,
        ]

        derivative = WrenchStamped()
        derivative.header = current.header
        derivative.wrench.force.x, derivative.wrench.force.y, derivative.wrench.force.z = force_diff
        derivative.wrench.torque.x, derivative.wrench.torque.y, derivative.wrench.torque.z = torque_diff

        return derivative

    @staticmethod
    def create_wrench(base_wrench, offset, mass, acceleration=None):
        """
        Creates a WrenchStamped with cleaned or zeroed data.
        :param base_wrench: The base wrench to modify.
        :param offset: The offset or center of mass adjustment (6D: [fx, fy, fz, tx, ty, tz]).
        :param mass: The mass to apply for zeroing.
        :param acceleration: Optional linear acceleration for zeroing ([ax, ay, az]).
        :return: WrenchStamped object with modified forces and torques.
        """
        # Initialize a new WrenchStamped
        wrench = WrenchStamped()
        wrench.header = base_wrench.header

        offset = np.array(offset)
        if offset.size == 3:
            offset = np.concatenate([offset, [0, 0, 0]])

        # Apply offset to forces and torques
        forces = np.array(
            [base_wrench.wrench.force.x, base_wrench.wrench.force.y, base_wrench.wrench.force.z]) - offset[:3]
        torques = np.array(
            [base_wrench.wrench.torque.x, base_wrench.wrench.torque.y, base_wrench.wrench.torque.z]) - offset[3:]

        # If acceleration is provided, apply dynamic adjustments
        if acceleration is not None:
            forces -= mass * np.array(acceleration)
            torques -= np.cross(offset[:3], acceleration)

        # Assign computed forces and torques back to the wrench
        wrench.wrench.force.x, wrench.wrench.force.y, wrench.wrench.force.z = forces
        wrench.wrench.torque.x, wrench.wrench.torque.y, wrench.wrench.torque.z = torques

        return wrench

    def publish_moving_average(self, buffer, publisher):
        """
        Publishes the moving average of a buffer.
        :param buffer: Deque containing WrenchStamped messages.
        :param publisher: ROS publisher to use for publishing.
        """
        avg = self.calculate_moving_average(buffer)
        if avg:
            publisher.publish(avg)

    def ft_cb(self, data):
        """
        Publishes cleaned and zeroed data, their moving averages, and derivatives with their moving averages.
        :param data: The raw sensor reading.
        :type data: WrenchStamped
        """
        self.last_ft = data
        current_time = data.header.stamp.to_sec()

        clean_ft = self.create_wrench(data, self.offset, 0)
        self.ft_clean_pub.publish(clean_ft)

        clean_derivative = self.calculate_derivative(clean_ft, self.prev_clean_ft, current_time, self.prev_time)
        if clean_derivative:
            self.ft_clean_derivative_pub.publish(clean_derivative)
            self.clean_derivative_buffer.append(clean_derivative)
        self.prev_clean_ft = clean_ft

        zeroed_ft = self.create_wrench(clean_ft, self.cm, self.m, self.last_linear_acceleration)
        self.ft_zeroed_pub.publish(zeroed_ft)

        zeroed_derivative = self.calculate_derivative(zeroed_ft, self.prev_zeroed_ft, current_time, self.prev_time)
        if zeroed_derivative:
            self.ft_zeroed_derivative_pub.publish(zeroed_derivative)
            self.zeroed_derivative_buffer.append(zeroed_derivative)
        self.prev_zeroed_ft = zeroed_ft

        self.clean_buffer.append(clean_ft)
        self.zeroed_buffer.append(zeroed_ft)

        self.publish_moving_average(self.clean_buffer, self.ft_clean_avg_pub)
        self.publish_moving_average(self.zeroed_buffer, self.ft_zeroed_avg_pub)
        self.publish_moving_average(self.clean_derivative_buffer, self.ft_clean_derivative_avg_pub)
        self.publish_moving_average(self.zeroed_derivative_buffer, self.ft_zeroed_derivative_avg_pub)

        self.prev_time = current_time

    def in_motion(self, js):
        """
        :param js: the robots joint state
        :type js: JointState
        :return: whether or not the robot is in motion
        :rtype: bool
        """
        return np.linalg.norm(np.array(js.velocity)) > self.not_moving_threshold

    def joint_state_cb(self, data):
        """
        Adds a new measurement to the parameter estimator, if the sensor has not moved for a while.
        :param data: The joint state used to determine whether the sensor is moving.
        :type data: JointState
        """
        if self.in_recalibration_mode:
            if self.in_motion(data) or self.last_movement == np.inf:
                self.last_movement = data.header.stamp.to_sec()
                self.recorded = False

            if not self.recorded \
                    and self.last_ft.header.stamp.to_sec() - self.last_movement > self.time_till_record:
                force_torque = np.array([self.last_ft.wrench.force.x,
                                         self.last_ft.wrench.force.y,
                                         self.last_ft.wrench.force.z,
                                         self.last_ft.wrench.torque.x,
                                         self.last_ft.wrench.torque.y,
                                         self.last_ft.wrench.torque.z])
                self.parameter_estimator.add_observation(self.last_linear_acceleration,
                                                         force_torque,
                                                         self.last_orientation)
                self.offset, self.m, self.c = self.parameter_estimator.get_params()
                self.cm = self.c * self.m
                self.number_of_measurements = self.parameter_estimator.get_number_of_measurements()
                self.print_params()
                self.pub_status(data.header)
                self.recorded = True
        elif self.update_offset and not self.in_motion(data):
            force_torque = np.array([self.last_ft.wrench.force.x,
                                     self.last_ft.wrench.force.y,
                                     self.last_ft.wrench.force.z,
                                     self.last_ft.wrench.torque.x,
                                     self.last_ft.wrench.torque.y,
                                     self.last_ft.wrench.torque.z])
            self.offset[:3] = force_torque[:3] - self.last_linear_acceleration * self.m
            self.offset[3:] = force_torque[3:] - np.cross(self.cm, self.last_linear_acceleration)
            self.save_params(self.path_to_saved_params)
            self.print_params()
            self.pub_status(data.header, score=False)
            self.update_offset = False

    def save_params(self, path):
        """
        Saves the current parameters in a yaml file to disk.
        :param path: The path to the location where the parameter file will be stored.
        :type path: str
        """
        with open(path, 'w') as stream:
            d = {
                'offset': {
                    'force': self.offset[:3].tolist(),
                    'torque': self.offset[3:].tolist()
                },
                'mass': float(self.m),
                'center_of_mass': self.c.tolist()
            }
            yaml.dump(d, stream, default_flow_style=False)
            rospy.loginfo(f'Parameters stored in {path}')

    def load_params(self, path):
        """
        Loads previously estimated parameters from disk.
        :param path: The path to the yaml file containing the parameters.
        :type path: str
        """
        try:
            with open(path, 'r') as stream:
                yaml_dict = yaml.safe_load(stream)
                self.offset = np.array(yaml_dict['offset']['force'] + yaml_dict['offset']['torque'])
                self.m = yaml_dict['mass']
                self.c = np.array(yaml_dict['center_of_mass'])
                self.cm = self.c * self.m
                rospy.loginfo(f'Parameters loaded from {path}')
                self.print_params()
        except Exception as e:
            rospy.logwarn(f'params.yaml not found or broken, please calibrate the sensor. Error: {e}')

    def print_params(self, score1=None, score2=None):
        """
        Prints the currently used parameters and score1/score2/number of measurements, if available.
        :param score1: Score1 returned from the parameter estimator.
        :type score1: float, optional, default None
        :param score2: Score2 returned from the parameter estimator.
        :type score2: float, optional, default None
        """
        if score1 is not None or score2 is not None:
            rospy.loginfo(f"Number of measurements: {self.number_of_measurements}")
            rospy.loginfo(f"score1: {score1}/1.0; score2: {score2}/1.0")
        rospy.loginfo("Force offset (N):")
        rospy.loginfo(list(self.offset[:3]))
        rospy.loginfo("Torque offset (Nm):")
        rospy.loginfo(list(self.offset[3:]))
        rospy.loginfo("Mass (kg):")
        rospy.loginfo(self.m)
        rospy.loginfo("Center of mass (m):")
        rospy.loginfo(list(self.c))
        rospy.loginfo("-------------------------------")

    def pub_status(self, header, score=True):
        """
        Publishes the current parameters to a status topic.
        :param header: The header used for the published msg, should contain frame_id of the sensor and current time.
        :type header: Header
        """
        info = FTRecalibrationInfo()
        info.header = header
        info.offset.force = Vector3(*self.offset[:3])
        info.offset.torque = Vector3(*self.offset[3:])
        info.mass = self.m
        info.center_of_mass = Vector3(*self.c)
        if score:
            info.score1 = self.parameter_estimator.score()
            info.score2 = self.parameter_estimator.score2
        else:
            info.score1 = float('nan')
            info.score2 = float('nan')

        self.status_pub.publish(info)


class ParameterEstimator:
    def __init__(self):
        self.ft_list = []
        self.g_list = []
        self.qs = []

    def _gravity_matrix(self, gravity):
        """
        :param gravity: A 3d vector describing the direction of gravity.
        :type gravity: list(float)
        :return: Feature vector such that the linear regression estimates offset, mass and center of mass.
        :rtype: np.mat
        """
        gravity_matrix = np.mat(((1, 0, 0, 0, 0, 0, gravity[0], 0, 0, 0),
                                 (0, 1, 0, 0, 0, 0, gravity[1], 0, 0, 0),
                                 (0, 0, 1, 0, 0, 0, gravity[2], 0, 0, 0),
                                 (0, 0, 0, 1, 0, 0, 0, 0, gravity[2], -gravity[1]),
                                 (0, 0, 0, 0, 1, 0, 0, -gravity[2], 0, gravity[0]),
                                 (0, 0, 0, 0, 0, 1, 0, gravity[1], -gravity[0], 0)))
        return gravity_matrix

    def add_observation(self, gravity, force_torque, sensor_orientation=None):
        """
        Add measurement. The gravity vector should point towards the ground.
        :param gravity: A 3d vector describing the direction of gravity.
        :type gravity: list(float)
        :param force_torque: A force torque measurement.
        :type force_torque: list(float)
        :param sensor_orientation: The orientation of the force torque sensor as quaternion.
                Optional as it is only used to evaluate the quality of the measurements.
        :type sensor_orientation: list(float), default None
        """
        self.g_list.append(self._gravity_matrix(gravity))
        self.ft_list.append(force_torque)
        if sensor_orientation is not None:
            self.qs.append(sensor_orientation)

    def get_params(self):
        """
        Uses linear regression to estimate offset, load mass and center of mass from added observations.
        :return: offset (N/Nm), mass (Kg), center of mass (m)
        :rtype: tuple(list(float), float, list(float))
        """
        x = LinearRegression(fit_intercept=False)
        X = np.concatenate(self.g_list)
        y = np.concatenate(self.ft_list)
        print(f'x: {X}')
        print(f'y: {y}')
        x.fit(X, y)
        self.offset = x.coef_[:6]
        self.m = x.coef_[6]
        self.center_of_mass = x.coef_[7:10] / x.coef_[6]
        self.score2 = x.score(X, y)
        return self.offset, self.m, self.center_of_mass

    def score(self):
        """
        Evaluates how well the chosen positions are distributed in the possible position space.
        This value should be close to 1.
        A value below 1 indicates that the sensor orientations from the measurements are too close together.
        A value above 1 indicates that there are likely not enough measurements.
        :return: A score describing how well distributed the sensor orientations were.
        :rtype: float
        """
        if len(self.qs) > 1:
            n = len(self.qs) - 1
            return scipy.spatial.distance.cdist(self.qs, self.qs, "cosine").sum() / ((n * (n + 1)))
        else:
            return float('nan')

    def get_number_of_measurements(self):
        return len(self.g_list)


if __name__ == '__main__':
    rospy.init_node("force_torque_cleaner")
    path_to_saved_params = rospy.get_param('~path_to_saved_params', default='params.yaml')
    ft_topic = rospy.get_param('~ft_topic', default='/ft/l_gripper_motor')
    acceleration_topic = rospy.get_param('~acceleration_topic', default='/acceleration')
    ftc = FTCleaner(ft_topic=ft_topic, path_to_saved_params=path_to_saved_params)
    rospy.spin()
