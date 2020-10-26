#!/usr/bin/env python
import os
import sys
from multiprocessing import Lock

import rospy
from std_srvs.srv import Trigger, TriggerResponse

from ll4ma_isaac.util.ros_util import log_msgs_to_h5
from ll4ma_isaac.srv import SaveData, SaveDataResponse
from ll4ma_isaac.msg import DataLog


class IsaacDataRecorder:

    def __init__(self, data_dir='', rate=100):
        self.data_dir = data_dir
        self.rate = rospy.Rate(rate)

        if self.data_dir:
            os.makedirs(self.data_dir, exist_ok=True)

        self._rate = rospy.Rate(rate)
        self._record_data = False
        self._msgs = []        
        self._mutex = Lock() # Safely interact with msg queue

        rospy.Subscriber("/isaac/data_log", DataLog, self._data_cb)
        rospy.Service("/data_collection/start_record_data", Trigger, self._start_record_srv)
        rospy.Service("/data_collection/stop_record_data", Trigger, self._stop_record_srv)
        rospy.Service("/data_collection/clear_data", Trigger, self._clear_data_srv)
        rospy.Service("/data_collection/save_data", SaveData, self._save_data_srv)

        self._received_data = False
        
    def run(self):
        rospy.loginfo("Waiting for data...")
        while not rospy.is_shutdown() and not self._received_data:
            self.rate.sleep()
        rospy.loginfo("Data received!")

        # Loop waiting for service calls
        while not rospy.is_shutdown():
            self.rate.sleep()

    def start_record_data(self):
        rospy.loginfo("Recording data...")
        self._record_data = True

    def stop_record_data(self):
        self._record_data = False
        rospy.loginfo("Data recording stopped")
        
    def save_data(self, h5_filename):
        self._mutex.acquire()
        if len(self._msgs) == 0:
            rospy.logerr("No data to save to H5 file")
            return False
        else:
            rospy.loginfo(f"Saving data to {h5_filename}...")
            log_msgs_to_h5(self._msgs, h5_filename)
            rospy.loginfo("Data saved successfully")
        self._mutex.release()

    def clear_data(self):
        self._mutex.acquire()
        self._msgs = []
        self._mutex.release()

    def _start_record_srv(self, req):
        if not self._received_data:
            return TriggerResponse(success=False,
                                   message="Cannot start recording, no data received yet")
        self.start_record_data()
        return TriggerResponse(success=True)

    def _stop_record_srv(self, req):
        self.stop_record_data()
        return TriggerResponse(success=True)

    def _save_data_srv(self, req):
        os.makedirs(req.save_dir, exist_ok=True)
        existing_files = [f for f in os.listdir(req.save_dir)
                          if f.endswith('.h5') and f.startswith(req.file_prefix)]
        idx = len(existing_files) + 1
        filename = os.path.join(req.save_dir, f'{req.file_prefix}_{idx:04d}.h5')

        if not self.save_data(filename):
            return SaveDataResponse(success=False, message="Could not save data")
        else:
            return SaveDataResponse(success=True)

    def _clear_data_srv(self, req):
        self.clear_data()
        return TriggerResponse(success=True)

    def _data_cb(self, msg):
        self._received_data = True
        if self._record_data:
            self._mutex.acquire()
            self._msgs.append(msg)
            self._mutex.release()
        

if __name__ == '__main__':
    rospy.init_node('isaac_data_recorder')
    recorder = IsaacDataRecorder()
    recorder.run()
