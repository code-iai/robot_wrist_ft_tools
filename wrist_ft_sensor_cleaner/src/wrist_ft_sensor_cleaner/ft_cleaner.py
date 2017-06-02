#!/usr/bin/env python
import numpy as np
import yaml
import rospy
import scipy
from geometry_msgs.msg._Vector3 import Vector3
from geometry_msgs.msg._WrenchStamped import WrenchStamped
from rospy.topics import Publisher, Subscriber
from sensor_msgs.msg._Imu import Imu
from sensor_msgs.msg._JointState import JointState
from sklearn.linear_model.base import LinearRegression
from std_msgs.msg._Header import Header
from std_srvs.srv._Trigger import Trigger, TriggerResponse
from wrist_ft_sensor_cleaner.msg._FTRecalibrationInfo import FTRecalibrationInfo


class FTCleaner(object):
    def __init__(self, ft_topic='/ft/l_gripper_motor',
                 acceleration_topic='/acceleration',
                 time_till_record=1.,
                 online=True,
                 path_to_saved_params='params.yaml'):
        """
        :param ft_topic: The name of the force/torque sensor topic.
        :type ft_topic: str, default '/ft/l_gripper_motor'
        :param time_till_record: The time waited from a movement stop until a measurement will be made in seconds.
        :type time_till_record: float, default 1.0
        :param online: 
        :type online: bool
        :param path_to_saved_params: The path yaml file containing previously estimated values.
        :type path_to_saved_params: str, default ''
        """
        self.path_to_saved_params = path_to_saved_params
        self.time_till_record = time_till_record
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
        self.ft_clean_pub = Publisher("{}_clean".format(ft_topic), WrenchStamped, queue_size=100)
        self.ft_zeroed_pub = Publisher("{}_zeroed".format(ft_topic), WrenchStamped, queue_size=100)
        self.status_pub = Publisher("~status", FTRecalibrationInfo, queue_size=100)
        if online:
            self.ft_sub = Subscriber(ft_topic, WrenchStamped, self.ft_cb, queue_size=100)
            self.joint_state_sub = rospy.Subscriber('joint_states', JointState, self.joint_state_cb, queue_size=100)
        self.recalibrate_srv_start = rospy.Service('~recalibration_start', Trigger, self.start_calibration)
        self.recalibrate_srv_stop = rospy.Service('~recalibration_stop', Trigger, self.stop_calibration)
        self.recalibrate_srv_update_offset = rospy.Service('~update_offset', Trigger, self.update_offset_cb)

        rospy.loginfo("ft cleaner for {} started".format(ft_topic))

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

    def ft_cb(self, data):
        """
        Publishes data with offset and load removed.
        :param data: The raw sensor reading.
        :type data: WrenchStamped
        """
        if abs(data.header.stamp.to_sec() - self.time_of_last_acc_msg.to_sec()) > .1:
            rospy.logwarn('Time difference between force/torque reading and acceleration is too high.')
        self.last_ft = data
        clean_ft = WrenchStamped()
        clean_ft.header.stamp = data.header.stamp
        if data.header.frame_id == '':
            clean_ft.header.frame_id = '/l_force_torque_link'
        else:
            clean_ft.header.frame_id = data.header.frame_id
        clean_ft.wrench.force.x = data.wrench.force.x - self.offset[0]
        clean_ft.wrench.force.y = data.wrench.force.y - self.offset[1]
        clean_ft.wrench.force.z = data.wrench.force.z - self.offset[2]
        clean_ft.wrench.torque.x = data.wrench.torque.x - self.offset[3]
        clean_ft.wrench.torque.y = data.wrench.torque.y - self.offset[4]
        clean_ft.wrench.torque.z = data.wrench.torque.z - self.offset[5]
        self.ft_clean_pub.publish(clean_ft)
        clean_ft.wrench.force.x -= self.m * self.last_linear_acceleration[0]
        clean_ft.wrench.force.y -= self.m * self.last_linear_acceleration[1]
        clean_ft.wrench.force.z -= self.m * self.last_linear_acceleration[2]
        clean_ft.wrench.torque.x -= self.cm[1] * self.last_linear_acceleration[2] - self.cm[2] * \
                                                                                    self.last_linear_acceleration[1]
        clean_ft.wrench.torque.y -= -self.cm[0] * self.last_linear_acceleration[2] + self.cm[2] * \
                                                                                     self.last_linear_acceleration[0]
        clean_ft.wrench.torque.z -= self.cm[0] * self.last_linear_acceleration[1] - self.cm[1] * \
                                                                                    self.last_linear_acceleration[0]
        self.ft_zeroed_pub.publish(clean_ft)

    def joint_state_cb(self, data):
        """
        Adds a new measurement to the parameter estimator, if the sensor has not moved for a while.
        :param data: The joint state used to determine whether the sensor is moving.
        :type data: JointState
        """
        if self.in_recalibration_mode:
            if np.linalg.norm(np.array(data.velocity)) > .2 or self.last_movement == np.inf:
                self.last_movement = data.header.stamp.to_sec()
                self.recorded = False

            # take measurement if the sensor was not moved in the last X seconds.
            if not self.recorded \
                    and self.last_ft.header.stamp.to_sec() - self.last_movement > self.time_till_record:
                force_torque = np.array([self.last_ft.wrench.force.x,
                                         self.last_ft.wrench.force.y,
                                         self.last_ft.wrench.force.z,
                                         self.last_ft.wrench.torque.x,
                                         self.last_ft.wrench.torque.y,
                                         self.last_ft.wrench.torque.z, ])
                self.parameter_estimator.add_observation(self.last_linear_acceleration,
                                                         force_torque,
                                                         self.last_orientation)
                self.offset, self.m, self.c = self.parameter_estimator.get_params()
                self.cm = self.c * self.m
                self.number_of_measurements = self.parameter_estimator.get_number_of_measurements()
                self.print_params()
                self.pub_status(data.header)
                self.recorded = True
        elif self.update_offset and np.linalg.norm(np.array(data.velocity)) < .2:
            force_torque = np.array([self.last_ft.wrench.force.x,
                                     self.last_ft.wrench.force.y,
                                     self.last_ft.wrench.force.z,
                                     self.last_ft.wrench.torque.x,
                                     self.last_ft.wrench.torque.y,
                                     self.last_ft.wrench.torque.z, ])
            self.offset[:3] = force_torque[:3] - self.last_linear_acceleration * self.m
            self.offset[3:] = force_torque[3:] - np.dot(self.cm, self.last_linear_acceleration)
            self.save_params(self.path_to_saved_params)
            self.print_params()
            self.pub_status(data.header, score=False)
            self.update_offset = False

    def save_params(self, path):
        """
        Saves the current parameters in a yaml file to disc.
        :param path: The path to the location where the parameter file will be stored.
        :type path: str
        """
        with open(path, 'w') as stream:
            d = dict()
            d['offset'] = dict()
            d['offset']['force'] = self.offset[:3].tolist()
            d['offset']['torque'] = self.offset[3:].tolist()
            d['mass'] = float(self.m)
            d['center_of_mass'] = self.c.tolist()
            yaml.dump(d, stream, default_flow_style=False)
            rospy.loginfo('Parameters stored in {}'.format(path))

    def load_params(self, path):
        """
        Loads previously estimated parameters from disc.
        :param path: The path to the yaml file containing the parameters.
        :type path: str
        """
        try:
            with open(path, 'r') as stream:
                yaml_dict = yaml.load(stream)
                self.offset = np.array(yaml_dict['offset']['force'] + yaml_dict['offset']['torque'])
                self.m = yaml_dict['mass']
                self.c = np.array(yaml_dict['center_of_mass'])
                self.cm = self.c * self.m
                rospy.loginfo('Parameters loaded from {}'.format(path))
                self.print_params()
        except:
            rospy.logwarn('params.yaml not found or broken, please calibrate the sensor')

    def print_params(self, score1=None, score2=None):
        """
        Prints the currently used parameters and score1/score2/number of measurements, if available.
        :param score1: Score1 returned from the parameter estimator.
        :type score1: float, optional, default None
        :param score2: Score2 returned from the parameter estimator.
        :type score1: float, optional, default None
        """
        if score1 is not None or score2 is not None:
            rospy.loginfo("number of measurements: {}".format(self.number_of_measurements))
            rospy.loginfo("score1: {}/1.0; score2: {}/1.0".format(score1, score2))
        rospy.loginfo("force offset (N):")
        rospy.loginfo(list(self.offset[:3]))
        rospy.loginfo("torque offset (Nm):")
        rospy.loginfo(list(self.offset[3:]))
        rospy.loginfo("mass (kg):")
        rospy.loginfo(self.m)
        rospy.loginfo("center of mass (m):")
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
            info.score1 = np.nan
            info.score2 = np.nan

        self.status_pub.publish(info)


class ParameterEstimator(object):
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
        Uses linear regression to estimate offset, load mass and center of mess from added observations.
        :return: offset (N/Nm), mass (Kg), center of mass (m)
        :rtype: tuple(list(float), float, list(float))
        """
        x = LinearRegression(fit_intercept=False)
        X = np.concatenate(self.g_list)
        y = np.concatenate(self.ft_list)
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
            return np.nan

    def get_number_of_measurements(self):
        return len(self.g_list)


if __name__ == '__main__':
    rospy.init_node("force_torque_cleaner")
    path_to_saved_params = rospy.get_param('~path_to_saved_params', default='params.yaml')
    dirty_ft_topic = rospy.get_param('~ft_topic', default='/ft/l_gripper_motor')
    acceleration_topic = rospy.get_param('~acceleration_topic', default='/acceleration')
    ftc = FTCleaner(ft_topic=dirty_ft_topic, path_to_saved_params=path_to_saved_params)
    rospy.spin()
