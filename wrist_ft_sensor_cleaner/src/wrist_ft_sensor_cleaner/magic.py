#!/usr/bin/env python
import argparse
import yaml
from scipy.spatial.distance import euclidean, cosine

import PyKDL
import itertools
import rosbag
import rospy
import scipy
import tf_conversions
import time
from contact_predictor_msgs.msg._FloatStamped import FloatStamped
from geometry_msgs.msg._Vector3Stamped import Vector3Stamped
from geometry_msgs.msg._WrenchStamped import WrenchStamped
from pr2_msgs.msg._AccelerometerState import AccelerometerState
from rospy.topics import Publisher, Subscriber
import numpy as np

from sensor_msgs.msg._Imu import Imu
from sensor_msgs.msg._JointState import JointState
from sklearn.linear_model.base import LinearRegression
from tf.transformations import quaternion_matrix

from ati_mini_40_driver.ft_cleaner import FTCleaner
from general_stuff.converter import to_msg, to_list
from general_stuff.good_stuff import Counter
from general_stuff.kdlwrapper import KdlWrapper
from general_stuff.transformer import Transformer

calibration_matrix = None
# all latest
# self.offset = np.array([-5.0889764,-11.99526169,18.03956339,-0.63760545,-0.63439524,0.05756393])
# no vel latest
# self.offset = np.array([-5.60379755, -12.03548631, 18.16928868, -0.63976589, -0.57368265, 0.0272274])
offset = np.array(
    [-5.4937990299158361, -11.910647556483461, 18.877349786726136, -0.61668283579453131, -0.60314614745431439,
     0.035747428793124425])
amp_gain = 50.34

offsets = None
gains = None
frame_id = None


def load_calibration_matrix(path):
    global calibration_matrix

    with open(path, 'r') as stream:
        try:
            yaml_dict = yaml.load(stream)
            calibration_matrix = np.array(yaml_dict['ft_params']['calibration_coeff'])
        except yaml.YAMLError as exc:
            print(exc)


def load_ft_params(path):
    global offsets
    global gains
    global frame_id

    with open(path, 'r') as stream:
        try:
            yaml_dict = yaml.load(stream)
            offsets = np.array(yaml_dict['ft_params']['offsets'])
            gains = np.array(yaml_dict['ft_params']['gains'])
            frame_id = np.array(yaml_dict['ft_params']['frame_id'])
        except yaml.YAMLError as exc:
            print(exc)


def do_magic(x):
    global offset
    global amp
    y = np.array([-932, -982, 3191, 995, 380, -90])
    x = np.array(x)
    x = (x) / (amp_gain * float(1 << 16))
    x = np.dot(calibration_matrix, x)
    # return x - offset
    return x


def bag_to_dict(in_bag_name, start_time=0, ft_topic="/ft/l_gripper_motor"):
    data = []
    end_time = start_time
    muh = 10000
    ope = FTCleaner(online=False)
    ope.in_recalibration_mode = True
    with rosbag.Bag(in_bag_name, "r") as in_bag:
        # start_time = in_bag.get_start_time() - start_time
        c = Counter(in_bag.get_message_count())
        c.printProgress(0)
        for i, (topic, bag_msg, t) in enumerate(in_bag.read_messages()):
            msg = to_msg(bag_msg)
            if isinstance(msg, JointState):
                ope.joint_state_cb(msg)
            elif isinstance(msg, WrenchStamped):
                ope.ft_cb(msg)
            # elif isinstance(msg, AccelerometerState):
            #     ope.acceleration_cb(msg)
            c.printProgress(i + 1)
    # pe.print_params()
    ope.parameter_estimator.print_params()


def calc_mass_etc():
    # return bag_to_dict("../../bags/2017-04-13/3.bag")
    return bag_to_dict("../../../bags/2017-04-13/latest.bag")
    # return bag_to_dict("../../bags/2017-04-26/latest.bag")
    # return bag_to_dict("../../bags/2017-04-28/latest.bag")





if __name__ == '__main__':
    # load_calibration_matrix('../../bags/ft12396_params.yaml')
    # print(calibration_matrix)
    # load_ft_params('../../bags/wg035_revF_params.yaml')
    # device = Magic()
    # print(offsets)
    # print(gains)
    # print(frame_id)
    rospy.init_node("magic")
    calc_mass_etc()
    # try:
    #     asdf = OnlineParamEstimator(time_till_record=1)
    #     rospy.spin()
    # finally:
    #     asdf.parameter_estimator.print_params()
    # a = do_magic([-932, -982, 3191, 995, 380, -90])
    # print(list(device.do_magic([-753, -929, 3105, 983, 438, -204])))
    # print(list(device.do_magic([-1143, -998, 3361, 980, 384, -6])))
    # print(list(do_magic([-1439, -932, 2420, 950, -413, -105])))
    # print(list(do_magic([-932, -982, 3191, 995, 380, -90])))
    # print(list(device.do_magic([-440, -994, 3795, 1017, 352, -75])-a))
    # qs = []
    # for i in range(100):
    #     a = np.random.random(4) *2 -1
    #     a /= np.linalg.norm(a)
    #     qs.append(a)
    # diffs = []
    # t = time.time()
    # for q1, q2 in itertools.combinations(qs, 2):
    #     # diffs.append(euclidean(q1, q2))
    #     diffs.append(cosine(q1, q2))
    # print('coverage: {}/2.0'.format(np.mean(diffs)))
    # # print(time.time() - t)
    # t = time.time()
    # n = len(qs) -1
    # print(scipy.spatial.distance.cdist(qs, qs, "cosine").sum()/((n*(n+1))))
    # print(time.time() - t)
    # for q1, q2 in itertools.combinations(qs, 2):
    #     diffs.append(euclidean(q1, q2))
    # diffs.append(cosine(q1, q2))
    # print('coverage: {}/2.0'.format(np.mean(diffs)))
