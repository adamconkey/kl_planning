#!/usr/bin/env python
import os
import sys
import yaml
import rospy
import numpy as np
from time import time
from tqdm import tqdm
from multiprocessing import Lock
from sensor_msgs.msg import JointState
from ll4ma_isaac.msg import DataLog
from ll4ma_isaac.util.ros_util import log_msgs_to_h5


class BabbleDataCollector:

    def __init__(self, data_dir, n_to_collect, n_steps=300, timeout=45, max_attempts=5, rate=100):
        self._data_dir = data_dir
        self._n_to_collect = n_to_collect
        self._n_steps = n_steps
        self._timeout=timeout
        self._max_attempts = max_attempts
        self._rate = rospy.Rate(rate)

        # Franka joint limits: https://frankaemika.github.io/docs/control_parameters.html
        self._joint_lows = np.array([-2.8973, -1.7628, -2.9873, -3.0718, -2.8973, -0.0175, -2.8973])
        self._joint_highs = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

        if self._data_dir:
            os.makedirs(self._data_dir, exist_ok=True)

        rospy.Subscriber("/isaac/data_log", DataLog, self._data_cb)
        rospy.Subscriber("/panda/joint_states", JointState, self._joint_state_cb)
        self._cmd_pub = rospy.Publisher("/panda/robot_command", JointState, queue_size=1)

        self._joint_cmd = JointState()
        self._joint_state = None

        self._record_data = False
        self._received_data = False

        # This will hold log msgs coming from Isaac, the idea is this will be cleared prior
        # to performing task, then it will accumulate while task is being performed, and at
        # end they can be converted as necessary and saved to disk.
        self._data_msgs = []
        self._mutex = Lock() # For safe handling of msg cache
    
    def run(self):
        rospy.loginfo("Waiting for simulator...")
        while not rospy.is_shutdown() and (not self._received_data or self._joint_state is None):
            self._rate.sleep()
        rospy.loginfo("Simulator is ready!")

        rospy.loginfo("Collecting babble data...")
        self._collect_babble_data()
        rospy.loginfo("Finished collecting data.")        
        
    def start_recording_data(self):
        self._record_data = True

    def stop_recording_data(self):
        self._record_data = False
        
    def save_data(self, h5_filename):
        self._mutex.acquire()
        if len(self._data_msgs) == 0:
            rospy.logerr("No data to save to H5 file")
        else:
            log_msgs_to_h5(self._data_msgs, h5_filename)
        self._mutex.release()

    def clear_data(self):
        self._mutex.acquire()
        self._data_msgs = []
        self._mutex.release()

    def _collect_babble_data(self):
        filenames = sorted([f for f in os.listdir(self._data_dir) if f.endswith('.h5')])
        n_collected = len(filenames)
        if n_collected > 0:
            rospy.loginfo(f"Already {n_collected} instances collected")
        if n_collected >= self._n_to_collect:
            rospy.loginfo(f"No instances to collect. Requested {self._n_to_collect} and "
                          f"there are already {n_collected} collected.")
            return

        rospy.loginfo(f"Collecting {self._n_to_collect - n_collected} instances")
        for i in tqdm(range(n_collected + 1, self._n_to_collect + 1)):
            h5_filename = os.path.join(self._data_dir, f'babble_{i:04d}.h5')
            self._collect_babble_instance(h5_filename)

    def _collect_babble_instance(self, h5_filename):
        if len(self._data_msgs) > 0:
            rospy.logerr("Data is already cached from previous recording. Need to clear data.")
            return False

        self._joint_cmd.position = self._joint_state.position
        self.start_recording_data()
        while not rospy.is_shutdown() and len(self._data_msgs) < self._n_steps:
            noise = np.random.multivariate_normal(np.zeros(7), np.eye(7)) * 0.15
            current = np.array(self._joint_cmd.position[:7])
            joints = np.clip(current + noise, self._joint_lows, self._joint_highs)
            gripper = np.random.randint(0, 2, size=(1,))
            rand_cmd = np.concatenate([joints, gripper]).tolist()
            self._joint_cmd.position = rand_cmd
            self._cmd_pub.publish(self._joint_cmd)
            self._rate.sleep()

        self.stop_recording_data()
        self.save_data(h5_filename)
        self.clear_data()
        
    def _data_cb(self, msg):
        self._received_data = True
        if self._record_data:
            self._mutex.acquire()
            self._data_msgs.append(msg)
            self._mutex.release()

    def _joint_state_cb(self, msg):
        self._joint_state = msg

    
if __name__ == '__main__':
    rospy.init_node('babble_data_collector')
    data_dir = rospy.get_param('~data_dir')
    n_to_collect = rospy.get_param('~n_to_collect')
    n_steps = rospy.get_param('~n_steps', 1000)
    rate = rospy.get_param('~rate', 1)
    collector = BabbleDataCollector(data_dir, n_to_collect, n_steps, rate=rate)
    collector.run()
