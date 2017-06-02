#!/usr/bin/env python
import numpy as np
import PyKDL
import rospy
from kdl_parser_py.urdf import treeFromFile, treeFromParam
from rospy.topics import Publisher, Subscriber
from sensor_msgs.msg._Imu import Imu
from sensor_msgs.msg._JointState import JointState


class GravityPublisher(object):
    def __init__(self, path_to_urdf=None, start_link='base_link', end_link='l_force_torque_link', gravity_vector=(0, 0, -9.8)):
        """
        :param path_to_urdf: The location of the robots urdf file. Will be loaded from the parameter server if None. 
        :type path_to_urdf: str or None
        :param start_link: The start link of a kinematic chain leading to the force/torque sensor. Should be a fixed link.
        :type start_link: str
        :param end_link: The link of the force/torque sensor.
        :type end_link: str
        :param gravity_vector: The direction of gravity relative to the start_link.
        :type gravity_vector: list(float)
        """
        self.joints_to_ft_ids = None
        self.frame_id = end_link

        self.start_link = start_link
        self.end_link = end_link
        self.gravity_vector = gravity_vector

        if path_to_urdf is None or path_to_urdf is "":
            success, tree = treeFromParam('robot_description')
            if not success:
                rospy.logerr('failed to load urdf from parameter server')
            else:
                rospy.loginfo('urdf loaded from parameter server')
        else:
            success, tree = treeFromFile(path_to_urdf)
            if not success:
                rospy.logerr('failed to load urdf from {}'.format(path_to_urdf))
            else:
                rospy.loginfo('urdf loaded from {}'.format(path_to_urdf))
        self.chain = tree.getChain(self.start_link, self.end_link)
        self.joint_names = []
        for i in range(self.chain.getNrOfSegments()):
            joint = self.chain.getSegment(i).getJoint()
            if joint.getTypeName() != 'None':
                self.joint_names.append(str(joint.getName()))
        self.fksolver = PyKDL.ChainFkSolverPos_recursive(self.chain)

        self.acceleration_pub = Publisher("acceleration", Imu, queue_size=100)
        self.joint_state_sub = Subscriber('joint_states', JointState, self.joint_state_cb, queue_size=100)

        rospy.loginfo("kdl gravity publisher started started")

    def joint_state_cb(self, data):
        if self.joints_to_ft_ids is None:
            self.joints_to_ft_ids = []
            for joint_name in self.joint_names:
                self.joints_to_ft_ids.append(data.name.index(joint_name))

        gravity, sensor_orientation = self.kinematic_gravity(data)
        imu = Imu()
        imu.header.frame_id = self.frame_id
        imu.header.stamp = data.header.stamp
        imu.linear_acceleration.x = gravity[0]
        imu.linear_acceleration.y = gravity[1]
        imu.linear_acceleration.z = gravity[2]
        imu.orientation.x = sensor_orientation[0]
        imu.orientation.y = sensor_orientation[1]
        imu.orientation.z = sensor_orientation[2]
        imu.orientation.w = sensor_orientation[3]
        self.acceleration_pub.publish(imu)

    def q_to_jnt_array(self, joint_states):
        """
        :param joint_states: A list of joint states for the kinematic chain leading to the sensor.
        :type joint_states: list(float)
        :return: The input joint array as PyKDL object.
        :rtype: PyKDL.JntArray
        """
        j = PyKDL.JntArray(len(joint_states))
        for i in range(len(joint_states)):
            j[i] = joint_states[i]
        return j

    def kinematic_gravity(self, js):
        """
        Calculates the gravity vector and sensor orientation using the forward kinematic to the sensor.
        :param js: The joint states of the kinematic chain leading to the sensor.
        :type js: list(float)
        :return: The gravity vector and a quaternion describing the sensor orientation relative to the start link of the chain.
        :rtype: tuple(np.array, np.array)
        """
        fk = PyKDL.Frame()
        jnta = self.q_to_jnt_array([js.position[x] for x in self.joints_to_ft_ids])
        self.fksolver.JntToCart(jnta, fk)
        g = fk.M.Inverse() * PyKDL.Vector(*self.gravity_vector)
        q = fk.M.GetQuaternion()
        return np.array([g[0], g[1], g[2]]), q


if __name__ == '__main__':
    rospy.init_node("gravity")
    start_link = rospy.get_param('~start_link', default='base_link')
    end_link = rospy.get_param('~end_link', default='l_force_torque_link')
    gravity = rospy.get_param('~gravity_vector', default='[0,0,-9.8]')
    urdf = rospy.get_param('~urdf', default=None)

    gravity = GravityPublisher(path_to_urdf=urdf, start_link=start_link, end_link=end_link, gravity_vector=eval(gravity))
    rospy.spin()
