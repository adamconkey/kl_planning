import sys
import rospy
import torch
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence
from scipy.spatial.transform import Rotation as R
import numpy as np
from time import time

from kl_planning.util import file_util, math_util
from kl_planning.srv import SetPose, SetPoseRequest


class Navigation2DEnvironment:

    def __init__(self, config_filename):
        file_util.check_path_exists(config_filename, "Scene configuration file")
        scene_config = file_util.load_yaml(config_filename)
        self.object_config = scene_config['objects']
        self.agent_config = scene_config['agents']
        self.indicator_config = scene_config['indicators']

        self.wheel_radius = self.agent_config['agent']['wheel_radius']
        self.robot_length = self.agent_config['agent']['length']
        
        self._create_collision_objects()

        # These save some lookups/computations in collision check
        L = self.agent_config['agent']['length']
        W = self.agent_config['agent']['width']
        self.p0 = np.array([-L / 2., W / 2.])
        self.p1 = self.p0 + np.array([L, 0])
        self.p2 = self.p1 + np.array([0, -W])
        self.p3 = self.p2 + np.array([-L, 0])

    def set_agent_location(self, pose):
        req = SetPoseRequest()
        req.pose.position.x = pose[0]
        req.pose.position.y = pose[1]
        req.pose.position.z = 0.01 # TODO hard-code
        quat = R.from_euler('z', pose[2], degrees=False).as_quat()
        req.pose.orientation.x = quat[0]
        req.pose.orientation.y = quat[1]
        req.pose.orientation.z = quat[2]
        req.pose.orientation.w = quat[3]
        set_pose = rospy.ServiceProxy("/visualization/set_agent_location", SetPose)
        try:
            set_pose(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call to set agent location failed: {e}")
        
    def dynamics(self, start_pose, act, noise_gain=0.02):
        """
        start_pose (b, 3) x, y, theta
        act (b, 2) 
        """

        next_pose = start_pose.detach().clone()
        next_pose[:,-1] += act[:,-1]
        next_pose[:,-1].clamp_(min=-np.pi, max=np.pi)
        delta_x = act[:,0] * torch.cos(next_pose[:,-1])
        delta_y = act[:,0] * torch.sin(next_pose[:,-1])
        next_pose[:,0] += delta_x
        next_pose[:,1] += delta_y

        # Add in noise on resulting state to model stochastic transition
        next_pose += torch.randn_like(next_pose) * noise_gain
        
        # delta_x = self.wheel_radius * torch.cos(act[:,0] + act[:,1]) / 2.
        # delta_y = self.wheel_radius * torch.sin(act[:,0] + act[:,1]) / 2.
        # delta_theta = (self.wheel_radius / self.robot_length) * (act[:,0] - act[:,1])

        # TODO I think the dynamics are nonsense, need to try to fix this
        
        # delta_x = (self.wheel_radius / 2.) * (act[:,0] + act[:,1]) * torch.cos(start_pose[:,-1])
        # delta_y = (self.wheel_radius / 2.) * (act[:,0] + act[:,1]) * torch.sin(start_pose[:,-1])
        # delta_theta = (self.wheel_radius / self.robot_length) * (act[:,0] - act[:,1])
        # delta = torch.stack([delta_x, delta_y, delta_theta], dim=-1)
        # next_pose = start_pose + delta
        return next_pose

    def get_trajectory(self, start_state, actions):
        """
        Applies dynamics from start state with action sequence to get trajectories.
        """
        T = actions.size(0)
        n_trajs = actions.size(1)
        start_state = start_state.repeat(n_trajs, 1)
        trajs = torch.zeros(T+1, n_trajs, 3)
        trajs[0] = start_state
        for t in range(T):
            trajs[t+1] = self.dynamics(trajs[t], actions[t])
        return trajs

    def fk(self, q):
        """
        Assuming q is the centroid point of a rectangle
        """

        p0 = q + self.p0
        p1 = q + self.p1
        p2 = q + self.p2
        p3 = q + self.p3
        return [p0, p1, p2, p3, p0]

    def in_collision(self, q):
        """
        Based on separating axis theorem code here: 
            https://hackmd.io/@US4ofdv7Sq2GRdxti381_A/ryFmIZrsl
        """
        # TODO super hacked, need to at least get this from config
        return (q[:,0] > -0.7) * (q[:,0] < 0.7) * (q[:,1] > -0.7) * (q[:,1] < 0.7)

    def cost(self, act, start_mu, start_sigma, goal_mu, goal_sigma):
        mus = [start_mu]
        sigmas = [start_sigma]
        sigma_points = []

        for t in range(len(act)):
            act_t = act[t].unsqueeze(1).repeat(1, 2 * start_mu.size(-1) + 1, 1)
            act_t = act_t.view(act_t.size(0) * act_t.size(1), -1)
            g = lambda x: self.dynamics(x, act_t)
            mu_prime, sigma_prime, Y = math_util.unscented_transform(mus[-1], sigmas[-1], g)
            mus.append(mu_prime)
            sigmas.append(sigma_prime)
            sigma_points.append(Y)
            
        cost = 0

        # Compute KL cost from final distribution to goal distribution
        kl_cost = 0
        p_G = MultivariateNormal(goal_mu, goal_sigma)
        T = len(mus)
        for t in range(T):
            # Increasing contribution of KL cost as time increases
            lambda_ = (t + 1) / float(T)
            p_t = MultivariateNormal(mus[t], sigmas[t])
            kl_cost += lambda_ * kl_divergence(p_t, p_G)
        cost += kl_cost

        # print("KL COST", kl_cost)
            
        # Compute collision costs based on sigma points
        collision_cost = 0
        n_points = float(len(sigma_points))
        for i, Y in enumerate(sigma_points):
            # Decreasing contribution of collision cost as time increases
            lambda_ = (n_points - i) / n_points
            B = Y.size(0)
            n_sigma = Y.size(1)
            in_collision = self.in_collision(Y.view(B * n_sigma, -1)) * 1000.0
            in_collision = in_collision.view(B, n_sigma)
            collision_cost += lambda_ * in_collision.sum(dim=1)
        cost += collision_cost

        # print("COLLISION", collision_cost)

        return cost

    def _create_collision_objects(self):
        self.polygons = []
        for obj_id, obj_data in self.object_config.items():
            if obj_data['type'] == 'cube':
                origin = obj_data['position']
                L = obj_data['length']
                W = obj_data['width']
                p0 = np.array([-L / 2., W / 2.])
                p1 = p0 + np.array([L, 0])
                p2 = p1 + np.array([0, -W])
                p3 = p2 + np.array([-L, 0])
                polygon = [p0, p1, p2, p3, p0]
                self.polygons.append(polygon)
            else:
                print(f"Unknown object type for making collision object: {obj_data['type']}")    
