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

from kl_planning.learners import LatentPlanningLearner
from kl_planning.planning import Planner
from kl_planning.environments import LatentEnvironment
from kl_planning.util import ros_util, data_util, math_util
from kl_planning.common.config import default_config


class LatentPlanner:

    def __init__(self, checkpoint_filename, device, n_obs_hist=3, execution_rate=1,
                 visualize=True):
        self.device = device
        self.n_obs_hist = n_obs_hist
        self.visualize = visualize
        self.learner = LatentPlanningLearner(checkpoint_filename=checkpoint_filename, device=device)
        self.learner.set_models_to_eval()

        rospy.Subscriber("/panda/joint_states", JointState, self._joint_state_cb)
        rospy.Subscriber("/rgb", Image, self._rgb_cb)
        rospy.Subscriber("/depth", Image, self._depth_cb)

        self.joint_command_pub = rospy.Publisher("/panda/robot_command", JointState, queue_size=1)
        self.rgb_belief_pub = rospy.Publisher("/rgb_belief", Image, queue_size=1)
        self.depth_belief_pub = rospy.Publisher("/depth_belief", Image, queue_size=1)

        self.rgb = []
        self.depth = []
        self.joint_state = None
        self.joint_command = JointState()

        self.rate = rospy.Rate(execution_rate)

        self.planner = Planner()
        self.env = LatentEnvironment(self.learner)

        self._rgb_mutex = Lock()
        self._depth_mutex = Lock()

    def run(self, horizon, n_iters, n_candidates, n_elite, timeout):
        rospy.loginfo("Waiting for observations...")
        while not rospy.is_shutdown() and (len(self.rgb) < self.n_obs_hist or
                                           len(self.depth) < self.n_obs_hist or
                                           self.joint_state is None):
            self.rate.sleep()
        rospy.loginfo("Observations received!")

        # Initialize joint command with current configuration
        self.joint_command.name = self.joint_state.name[:7] # Excluding gripper
        self.joint_command.position = list(self.joint_state.position)[:7]
        self.joint_command.velocity = [0.0] * len(self.joint_state.name)
        self.joint_command.effort = [0.0] * len(self.joint_state.name)
                
        act = {'delta_joint_positions':
               torch.zeros(self.n_obs_hist, 1, 7, dtype=torch.float32).to(self.device)}
                
        # TODO trying just getting a goal observation to encode to goal latent state for planning
        # with L2 cost. If this works should set this up as alternative as it is probably a
        # baseline
        h5_filename = '/home/adam/push_blocks_data/expert_demo_0001.h5'
        idx = 60
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

        # TODO assuming Dirac distribution for now to test planner
        from kl_planning.distributions import DiracDelta
        goal_dist = DiracDelta(goal_state.repeat(n_candidates, 1), force_identity_precision=True)
        kl_divergence = lambda x, y: math_util.kl_dirac_mvn(x, y, self.device)

        joint_delta = 0.1
        min_act = torch.full((self.learner.config.action_size,), -joint_delta, device=self.device)
        max_act = torch.full((self.learner.config.action_size,), joint_delta, device=self.device)
        
        
        timed_out = False
        start = rospy.get_time()
        
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

    def _get_current_observations(self):
        self._rgb_mutex.acquire()
        rgb = [self.learner.dataset.process_data_in(np.expand_dims(d, 0), 'rgb') for d in self.rgb]
        self._rgb_mutex.release()
        obs = {'rgb': torch.stack(rgb).to(self.device)}
        return obs
        
    def _joint_state_cb(self, msg):
        self.joint_state = msg
    
    def _rgb_cb(self, msg):
        with self._rgb_mutex:
            if len(self.rgb) == self.n_obs_hist:
                self.rgb.pop(0)
            self.rgb.append(ros_util.msg_to_rgb(msg))

    def _depth_cb(self, msg):
        with self._depth_mutex:
            if len(self.depth) == self.n_obs_hist:
                self.depth.pop(0)
            self.depth.append(ros_util.msg_to_img(msg))


if __name__ == '__main__':
    rospy.init_node('mpc_planner')

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--n_iters', type=int, default=5)
    parser.add_argument('--n_candidates', type=int, default=25)
    parser.add_argument('--n_elite', type=int, default=5)
    parser.add_argument('--execution_rate', type=int, default=1)
    parser.add_argument('--n_obs_hist', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=45)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
        
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    planner = LatentPlanner(args.checkpoint, device, args.n_obs_hist, args.execution_rate,
                            args.visualize)
    planner.run(args.horizon, args.n_iters, args.n_candidates, args.n_elite, args.timeout)
