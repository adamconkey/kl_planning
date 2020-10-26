#!/usr/bin/env python
import os
import sys
import yaml
from time import time
from tqdm import tqdm
from multiprocessing import Lock

import rospy
import rospkg
rospack = rospkg.RosPack()
from std_srvs.srv import Trigger, TriggerRequest

from ll4ma_isaac.msg import DataLog, Object
from ll4ma_isaac.srv import ResetTask, ResetTaskRequest, CreateScene, CreateSceneRequest
from ll4ma_isaac.environments.common import Status
from ll4ma_isaac.util.ros_util import log_msgs_to_h5


DEFAULT_SCENE_PATH = os.path.join(rospack.get_path('ll4ma_isaac'), 'config', 'scenes')


class IsaacDataCollector:

    def __init__(self, scene_config, expert_data_dir='', learner_data_dir='',
                 n_expert_demos=0, n_learner_executions=0, timeout=45,
                 max_attempts=5, rate=100):
        self._scene_config = scene_config
        self._expert_data_dir = expert_data_dir
        self._learner_data_dir = learner_data_dir
        self._n_expert_demos = n_expert_demos
        self._timeout=timeout
        self._max_attempts = max_attempts
        self._rate = rospy.Rate(rate)

        if self._expert_data_dir:
            os.makedirs(self._expert_data_dir, exist_ok=True)
        if self._learner_data_dir:
            os.makedirs(self._learner_data_dir, exist_ok=True)

        rospy.Subscriber("/isaac/data_log", DataLog, self._data_cb)

        rospy.loginfo("Waiting for services...")
        rospy.wait_for_service("/isaac/create_scene")
        rospy.wait_for_service("/isaac/perform_task")
        rospy.wait_for_service("/isaac/reset_task")
        rospy.loginfo("Services are ready.")

        self._sim_status = None
        self._record_data = False

        # This will hold log msgs coming from Isaac, the idea is this will be cleared prior
        # to performing task, then it will accumulate while task is being performed, and at
        # end they can be converted as necessary and saved to disk.
        self._data_msgs = []
        self._mutex = Lock() # For safe handling of msg cache
    
    def run(self):
        self.create_scene(self._scene_config)

        rospy.loginfo("Waiting for simulator scene to be initialized...")
        if not self._wait_for_status(Status.READY):
            rospy.logerr("Simulation was not initialized. Scene creation is not working.")
            return
        rospy.loginfo("Simulator is ready!")

        self.reset_task()

        if self._expert_data_dir:
            rospy.loginfo(f"Collecting {self._n_expert_demos} expert demos...")
            self._collect_expert_demos()
            rospy.loginfo("Finished collecting expert demos.")

        if self._learner_data_dir:
            rospy.loginfo("Collecting learner policy executions...")
            self._collect_learner_executions()
            rospy.loginfo("Finished collecting learner policy executions.")
        
    def create_scene(self, config_filename, config_path=DEFAULT_SCENE_PATH):
        _config_filename = os.path.join(config_path, config_filename)
        if not os.path.exists(_config_filename):
            rospy.logerr(f"Scence config does not exist: {_config_filename}")
            return

        with open(_config_filename, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            
        req = CreateSceneRequest()
        req.task_name = config['task_name']
        req.reset_scene = True
        for obj_id, obj_data in config['objects'].items():
            obj = Object()
            obj.name = obj_id
            obj.body_type = obj_data['body_type']
            obj.semantic_label = obj_data['semantic_label']
            obj.length = obj_data['length']
            obj.width = obj_data['width']
            obj.height = obj_data['height']
            obj.units = obj_data['units']
            obj.mass = obj_data['mass']
            obj.color.r = obj_data['color'][0]
            obj.color.g = obj_data['color'][1]
            obj.color.a = obj_data['color'][2]
            req.objects.append(obj)
        _create_scene = rospy.ServiceProxy("/isaac/create_scene", CreateScene)
        try:
            _create_scene(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call to create scene failed: {e}")

    def reset_task(self):
        _reset_task = rospy.ServiceProxy("/isaac/reset_task", ResetTask)
        try:
            _reset_task(ResetTaskRequest())
        except rospy.ServiceException as e:
            rospy.logerr(f"Reset task service request failed: {e}")

    def reset_robot(self):
        _reset_robot = rospy.ServiceProxy("/isaac/reset_robot", Trigger)
        try:
            _reset_robot(TriggerRequest())
        except rospy.ServiceException as e:
            rospy.logerr(f"Reset robot service request failed: {e}")
            
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

    def undo_recorded_task(self, attempts):
        """
        Convenience function to make collection code more compact
        """
        self.stop_recording_data()
        self.clear_data()
        self.reset_task()
        return attempts + 1

    def _collect_expert_demos(self):
        expert_demo_filenames = [f for f in os.listdir(self._expert_data_dir) if f.endswith('.h5')]
        expert_demo_filenames = sorted(expert_demo_filenames)
        n_expert_collected = len(expert_demo_filenames)
        if n_expert_collected > 0:
            rospy.loginfo(f"Already {n_expert_collected} demos collected")
        if n_expert_collected >= self._n_expert_demos:
            rospy.loginfo(f"No expert demos to collect. Requested {self._n_expert_demos} and "
                          "there are already {n_expert_collected} demos collected.")
            return

        rospy.loginfo(f"Collecting {self._n_expert_demos - n_expert_collected} demos")
        for i in tqdm(range(n_expert_collected + 1, self._n_expert_demos + 1)):
            h5_filename = os.path.join(self._expert_data_dir, f'expert_demo_{i:04d}.h5')
            if not self._collect_expert_demo(h5_filename):
                rospy.logerr("Could not collect demo, something is probably wrong with sim")
                break

    def _collect_expert_demo(self, h5_filename):
        if len(self._data_msgs) > 0:
            rospy.logerr("Data is already cached from previous recording. Need to clear data "
                         "before collecting a new demo.")
            return False
        if not self._wait_for_status(Status.READY):
            rospy.logerr("Simulation is not ready. Something is wrong.")
            return False

        attempts = 0
        while attempts < self._max_attempts:
            # Wait until task scene is ready to be manipulated
            if not self._wait_for_status(Status.READY):
                attempts = self.undo_recorded_task(attempts)
                continue
            # Start recording and initiate the task
            self.start_recording_data()
            self._start_task()
            # Wait for task to actually start executing
            if not self._wait_for_status(Status.EXECUTING_TASK):
                attempts = self.undo_recorded_task(attempts)
                continue
            # Wait for task execution to finish (should be either success or failure)
            if not self._wait_for_different_status(Status.EXECUTING_TASK, timeout=40):
                attempts = self.undo_recorded_task(attempts)
                continue
            # Retract the robot to home position
            self.reset_robot()
            if not self._wait_for_status(Status.RESETTING_ROBOT):
                attempts = self.undo_recorded_task(attempts)
                continue
            # Wait for robot to actually be at the home position
            if not self._wait_for_status(Status.ROBOT_HOME):
                attempts = self.undo_recorded_task(attempts)
                continue
            # If we got this far it should have recorded the full demo
            self.stop_recording_data()
            break
        
        if attempts > self._max_attempts:
            rospy.logerr("Exceeded max number of attempts. Something is probably wrong in sim.")
            self.reset_task()
            return False

        self.save_data(h5_filename)
        self.clear_data()
        self.reset_task()
        return True

    def _collect_learner_executions(self):
        
        # TODO this should manage a directory whose name is the same as the model checkpoint loaded
        # and copy the checkpoint into that directory, can still count the number of executions
        # in it because it might fail and you can just pick up data recording where you left off.
        # Will then collect up to the number of exeuctions specified

        pass

    def _collect_learner_execution(self, h5_filename):
        """
        Runs one episode of the learner agent executing its policy in the environment, and
        recording the data to be logged to an H5 file.
        """
        attempts = 0
        while attempts < self._max_attempts:
            # Wait until task scene is ready to be manipulated
            if not self._wait_for_status(Status.READY):
                self.reset_task()
                attempts += 1
                continue
            self.start_recording_data()

            
            # TODO need to take loaded model (probably load in init), and use the MPC planner to
            # run executions taking live observations from the sim. This I think should just
            # execute up to some pre-defined horizon that seems suitable for the task. Maybe 30
            # seconds or something is sufficient to start with. Or some fixed number of actions
            # that should be commanded in the environment that should be sufficient if the task
            # were to be successfully performed.
            
            
            self.stop_recording_data()
                
            # If we got this far it should have been successful
            self.reset_task()
            break
        if attempts > self._max_attempts:
            rospy.logerr("Exceeded max number of attempts. Something is probably wrong in sim.")
            return False

        self.save_data(h5_filename)
        self.clear_data()
        return True
        
    def _data_cb(self, msg):
        if self._record_data:
            self._mutex.acquire()
            self._data_msgs.append(msg)
            self._mutex.release()
        self._sim_status = Status[msg.status]

    def _start_task(self):
        start_task = rospy.ServiceProxy("/isaac/perform_task", Trigger)
        try:
            start_task(TriggerRequest())
        except rospy.ServiceException as e:
            rospy.logerr(f"Perform task service request failed: {e}")

    def _wait_for_status(self, status, timeout=None):
        if timeout is None:
            timeout = self._timeout
        current_time = 0
        while self._sim_status != status and current_time <= timeout and not rospy.is_shutdown():
            start = time()
            self._rate.sleep()
            current_time += time() - start
        if current_time > timeout:
            rospy.logerr(f"Timed out ({timeout} seconds) waiting for {status.name}")
            return False
        return True

    def _wait_for_different_status(self, status, timeout=None):
        if timeout is None:
            timout = self._timeout
        current_time = 0
        while self._sim_status == status and current_time <= timeout and not rospy.is_shutdown():
            start = time()
            self._rate.sleep()
            current_time += time() - start
        if current_time > timeout:
            rospy.logerr(f"Timed out ({timeout} seconds) waiting for change from {status.name}")
            return False
        return True

    
if __name__ == '__main__':
    rospy.init_node('isaac_data_collector')
    scene_config = rospy.get_param('~scene_config')
    expert_data_dir = rospy.get_param('~expert_data_dir', '')
    learner_data_dir = rospy.get_param('~learner_data_dir', '')
    n_expert_demos = rospy.get_param('~n_expert_demos', 0)
    collector = IsaacDataCollector(scene_config, expert_data_dir, learner_data_dir, n_expert_demos)
    collector.run()
