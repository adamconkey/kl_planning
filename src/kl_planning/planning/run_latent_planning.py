#!/usr/bin/env python
import os
import sys
import rospy
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
import torch
from torch.distributions.normal import Normal
from multiprocessing import Lock

from sensor_msgs.msg import JointState, Image
from std_srvs.srv import Trigger, TriggerRequest
from ll4ma_isaac.srv import SaveData, SaveDataRequest

from kl_planning.learners import LatentPlanningLearner
from kl_planning.planning import Planner
from kl_planning.environments import LatentEnvironment
from kl_planning.util import ros_util, data_util, math_util
from kl_planning.common.config import default_config


class LatentPlanner:

    def __init__(self, checkpoint_filename, device, save_dir='', save_prefix='',
                 execution_rate=1, visualize=True):
        self.device = device
        self.save_dir = save_dir
        self.save_prefix = save_prefix
        self.visualize = visualize
        self.learner = LatentPlanningLearner(checkpoint_filename=checkpoint_filename, device=device)
        self.learner.set_models_to_eval()

        rospy.Subscriber("/panda/joint_states", JointState, self._joint_state_cb)
        rospy.Subscriber("/rgb", Image, self._rgb_cb)

        self.joint_command_pub = rospy.Publisher("/panda/robot_command", JointState, queue_size=1)
        self.rgb_belief_pub = rospy.Publisher("/rgb_belief", Image, queue_size=1)

        self.rgb = None
        self.joint_state = None
        self.joint_command = JointState()

        self.rate = rospy.Rate(execution_rate)

        self.planner = Planner()
        self.env = LatentEnvironment(self.learner)

        self.save_data = save_dir != ''

        if self.save_data:
            rospy.loginfo("Waiting for data recording services...")
            rospy.wait_for_service("/data_collection/start_record_data")
            rospy.wait_for_service("/data_collection/stop_record_data")
            rospy.wait_for_service("/data_collection/clear_data")
            rospy.wait_for_service("/data_collection/save_data")
            rospy.loginfo("Services are up.")

    def run(self, horizon, n_iters, n_candidates, n_elite, timeout):
        rospy.loginfo("Waiting for observations...")
        while not rospy.is_shutdown() and (self.rgb is None or self.joint_state is None):
            self.rate.sleep()
        rospy.loginfo("Observations received!")

        # Initialize joint command with current configuration
        self.joint_command.name = self.joint_state.name[:7] # Excluding gripper
        self.joint_command.position = list(self.joint_state.position)[:7]
        self.joint_command.velocity = [0.0] * len(self.joint_state.name)
        self.joint_command.effort = [0.0] * len(self.joint_state.name)
                
        act = {'delta_joint_positions': torch.zeros(1, 1, 7, dtype=torch.float32).to(self.device)}
                
        # TODO trying just getting a goal observation to encode to goal latent state for planning
        # with L2 cost. If this works should set this up as alternative as it is probably a
        # baseline
        h5_filename = '/home/adam/push_block_lr_data/expert_demo_0002.h5'
        idx = 150
        goal_obs = {}
        goal_act = {}
        with h5py.File(h5_filename, 'r') as h5_file:
            for m in self.learner.dataset.obs_modalities:
                goal_obs[m] = self.learner.dataset.process_data_in(
                    np.expand_dims(h5_file[m][idx], 0), m)
            for m in self.learner.dataset.act_modalities:
                if m == 'delta_joint_positions':
                    joint_pos = h5_file['joint_positions']
                    data = joint_pos[idx] - joint_pos[idx - self.learner.config.time_subsample]
                    data = np.expand_dims(data, 0)
                else:
                    data = np.expand_dims(h5_file[m][idx], 0)
                goal_act[m] = self.learner.dataset.process_data_in(data, m)
        goal_obs = {k: v.unsqueeze(1).to(self.device) for k, v in goal_obs.items()}
        goal_act = {k: v.unsqueeze(1).to(self.device) for k, v in goal_act.items()}
        goal_state = self.learner.compute_transition(goal_act, goal_obs).posterior_states[-1]
        decode = self.learner.decode_state(goal_state.unsqueeze(0))

        # plt.imshow(decode['rgb'].squeeze())
        # plt.show()
        # sys.exit()

        # TODO assuming Dirac distribution for now to test planner
        from kl_planning.distributions import DiracDelta
        goal_dist = DiracDelta(goal_state.repeat(n_candidates, 1), force_identity_precision=True)
        kl_divergence = lambda x, y: math_util.kl_dirac_mvn(x, y, self.device)

        joint_delta = 0.2
        min_act = torch.full((self.learner.config.action_size,), -joint_delta, device=self.device)
        max_act = torch.full((self.learner.config.action_size,), joint_delta, device=self.device)
        
        
        timed_out = False
        start = rospy.get_time()

        if self.save_data:
            self._start_record_data()

        while not rospy.is_shutdown() and not timed_out:
            obs = self._get_current_observations()
            rssm_out = self.learner.compute_transition(act, obs)

            current_state = rssm_out.posterior_states[-1]
            start_mean = rssm_out.posterior_means[-1].squeeze()
            start_std_dev = rssm_out.posterior_std_devs[-1].squeeze()
            start_dist = Normal(start_mean, start_std_dev)
            belief = rssm_out.beliefs[-1]

            deltas = self.planner.plan_cem(self.env, start_dist, goal_dist, min_act, max_act,
                                           self.device, horizon, n_iters, n_candidates, n_elite,
                                           kl_divergence=kl_divergence, belief=belief)

            
            deltas = self.learner.dataset.process_data_out(deltas, 'delta_joint_positions')

            # print("DELTAS", deltas)

            if self.visualize:
                decoded = self.learner.decode_state(current_state.unsqueeze(0))
                rgb_belief_msg = ros_util.rgb_to_msg(decoded['rgb'].squeeze())
                self.rgb_belief_pub.publish(rgb_belief_msg)

            for i in range(len(self.joint_command.position)):
                self.joint_command.position[i] += deltas[0][i]
            self.joint_command.header.stamp = rospy.Time.now()
            self.joint_command_pub.publish(self.joint_command)
                        
            self.rate.sleep()
            timed_out = rospy.get_time() - start > timeout
            
        if timed_out:
            rospy.loginfo(f"Timed out after {int(rospy.get_time() - start)} seconds")
        if self.save_data:
            self._stop_record_data()
            self._save_data()
            self._clear_data()

    def _get_current_observations(self):
        rgb = self.learner.dataset.process_data_in(np.expand_dims(self.rgb, 0), 'rgb')
        rgb = rgb.unsqueeze(0)
        joints = np.expand_dims(np.expand_dims(self.joint_state.position[:7], 0), 0)
        joints = self.learner.dataset.process_data_in(joints, 'joint_positions')
        obs = {'rgb': rgb.to(self.device), 'joint_positions': joints.to(self.device)}
        return obs
        
    def _joint_state_cb(self, msg):
        self.joint_state = msg
    
    def _rgb_cb(self, msg):
        self.rgb = ros_util.msg_to_rgb(msg)
            
    def _start_record_data(self):
        start_record = rospy.ServiceProxy("/data_collection/start_record_data", Trigger)
        try:
            start_record()
        except rospy.ServiceException as e:
            rospy.logerr("Could not call start data record service")

    def _stop_record_data(self):
        stop_record = rospy.ServiceProxy("/data_collection/stop_record_data", Trigger)
        try:
            stop_record()
        except rospy.ServiceException as e:
            rospy.logerr(f"Could not call stop data record service: {e}")

    def _save_data(self):
        save_data = rospy.ServiceProxy("/data_collection/save_data", SaveData)
        try:
            save_data(SaveDataRequest(save_dir=self.save_dir, file_prefix=self.save_prefix))
        except rospy.ServiceException as e:
            rospy.logerr(f"Could not call save data service: {e}")

    def _clear_data(self):
        clear_data = rospy.ServiceProxy("/data_collection/clear_data", Trigger)
        try:
            clear_data()
        except rospy.ServiceException as e:
            rospy.logerr(f"Could not call clear data service: {e}")
            

if __name__ == '__main__':
    rospy.init_node('mpc_planner')

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--n_iters', type=int, default=5)
    parser.add_argument('--n_candidates', type=int, default=100)
    parser.add_argument('--n_elite', type=int, default=5)
    parser.add_argument('--execution_rate', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=45)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--save_prefix', type=str, default='')
    args = parser.parse_args()
        
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    planner = LatentPlanner(args.checkpoint, device, args.save_dir, args.save_prefix,
                            args.execution_rate, args.visualize)
    planner.run(args.horizon, args.n_iters, args.n_candidates, args.n_elite, args.timeout)
