#!/usr/bin/env python
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Vector3, Vector3Stamped
from rospy.topics import Publisher, Subscriber
from sensor_msgs.msg._Imu import Imu
from sensor_msgs.msg._JointState import JointState
from tf2_geometry_msgs import do_transform_vector3


class GravityPublisher(object):
    def __init__(self, fixed_frame='base_link', sensor_frame='l_force_torque_link', gravity_vector=(0, 0, -9.8)):
        """
        :param fixed_frame: The start link of a kinematic chain leading to the force/torque sensor. Should be a fixed link.
        :type fixed_frame: str
        :param sensor_frame: The link of the force/torque sensor.
        :type sensor_frame: str
        :param gravity_vector: The direction of gravity relative to the start_link.
        :type gravity_vector: list(float)
        """
        self.joints_to_ft_ids = None
        self.frame_id = sensor_frame

        self.start_link = fixed_frame
        self.end_link = sensor_frame
        self.gravity_vector = Vector3Stamped()
        self.gravity_vector.header.frame_id = self.start_link
        self.gravity_vector.vector = Vector3(*gravity_vector)

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.acceleration_pub = Publisher("acceleration", Imu, queue_size=100)
        self.joint_state_sub = Subscriber('joint_states', JointState, self.joint_state_cb, queue_size=100)

        rospy.loginfo("tf gravity publisher started started")

    def transformPose(self, target_frame, pose):
        transform = self.tfBuffer.lookup_transform(target_frame,
                                                   pose.header.frame_id,
                                                   rospy.Time(),
                                                   rospy.Duration(1.0))
        transformed_vector = do_transform_vector3(pose, transform)
        return transformed_vector

    def joint_state_cb(self, data):
        transform_gravity = self.transformPose(self.end_link, self.gravity_vector)
        imu = Imu()
        imu.header.frame_id = self.frame_id
        imu.header.stamp = data.header.stamp
        imu.linear_acceleration = transform_gravity.vector
        imu.orientation.x = 0
        imu.orientation.y = 0
        imu.orientation.z = 0
        imu.orientation.w = 1
        self.acceleration_pub.publish(imu)


if __name__ == '__main__':
    rospy.init_node("gravity_publisher")
    start_link = rospy.get_param('~fixed_frame', default='table_link')
    end_link = rospy.get_param('~sensor_frame', default='kms40_frame_out')
    gravity = rospy.get_param('~gravity_vector', default='[0,0,-9.8]')

    gravity = GravityPublisher(fixed_frame=start_link, sensor_frame=end_link, gravity_vector=eval(gravity))
    rospy.spin()
