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
        
        self._create_collision_objects()

        # These save some lookups/computations in collision check
        L = self.agent_config['agent']['length']
        W = self.agent_config['agent']['width']
        self.p0 = np.array([-L / 2., W / 2.])
        self.p1 = self.p0 + np.array([L, 0])
        self.p2 = self.p1 + np.array([0, -W])
        self.p3 = self.p2 + np.array([-L, 0])

    def set_agent_location(self, position, angle):
        quat = R.from_euler('z', angle, degrees=False).as_quat()
        req = SetPoseRequest()
        req.pose.position.x = position[0]
        req.pose.position.y = position[1]
        req.pose.position.z = 0.01 # TODO hard-code
        req.pose.orientation.x = quat[0]
        req.pose.orientation.y = quat[1]
        req.pose.orientation.z = quat[2]
        req.pose.orientation.w = quat[3]
        set_pose = rospy.ServiceProxy("/visualization/set_agent_location", SetPose)
        try:
            set_pose(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call to set agent location failed: {e}")
        
    def dynamics(self, start_xy, act, repeat=True):
        """
        start_xy (b, 2) x, y
        act (b, 2) dist, angle
        """
        delta_x = act[:,0] * torch.cos(act[:,1])
        delta_y = act[:,0] * torch.sin(act[:,1])
        delta_xy = torch.stack([delta_x, delta_y], dim=-1)
        if repeat:
            # 2n+1 = 5 sigma points, just hard-coding for now:
            delta_xy = delta_xy.unsqueeze(1).repeat(1, 5, 1)
        next_xy = start_xy + delta_xy
        return next_xy

    def get_trajectory(self, start_state, actions):
        """
        Applies dynamics from start state with action sequence to get trajectories.
        """
        T = actions.size(0)
        n_trajs = actions.size(1)
        start_state = start_state.repeat(n_trajs, 1)
        trajs = torch.zeros(T+1, n_trajs, 2)
        trajs[0] = start_state
        for t in range(T):
            trajs[t+1] = self.dynamics(trajs[t], actions[t], False)
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
        # TODO mocking hard to see if collision checker is fucked
        return q[0] > -0.7 and q[0] < 0.7 and q[1] > -0.7 and q[1] < 0.7

    def cost(self, act, start_mu, start_sigma, goal_mu, goal_sigma):
        mus = [start_mu]
        sigmas = [start_sigma]
        sigma_points = []

        start = time()
        for t in range(len(act)):
            g = lambda x: self.dynamics(x, act[t])
            mu_prime, sigma_prime, Y = math_util.unscented_transform(mus[-1], sigmas[-1], g)
            mus.append(mu_prime)
            sigmas.append(sigma_prime)
            sigma_points.append(Y)
        print(f"UNSCENTED: {time() - start}")
            
        cost = 0

        start = time()
        # Compute KL cost from final distribution to goal distribution
        kl_cost = 0
        p_G = MultivariateNormal(goal_mu, goal_sigma)
        T = len(mus)
        for t in range(T):
            # Increasing contribution of KL cost as time increases
            lambda_ = (t + 1) / float(T)
            p_t = MultivariateNormal(mus[t], sigmas[t])
            kl_cost += lambda_ * kl_divergence(p_t, p_G)
        print(f"KL COST: {time() - start}")
        cost += kl_cost
            
        # Compute collision costs based on sigma points
        start = time()
        collision_cost = 0
        n_points = float(len(sigma_points))
        for i, Y in enumerate(sigma_points):
            # Decreasing contribution of collision cost as time increases
            lambda_ = (n_points - i) / n_points
            B = Y.size(0)
            n_sigma = Y.size(1)
            # TODO ideally you can batch compute the collisions, will need to rewrite the
            # collision check code though
            in_collision = torch.zeros(B, n_sigma)
            for j in range(B):
                for k in range(n_sigma):
                    in_collision[j, k] = float(self.in_collision(Y[j, k])) * 1000000.0
            collision_cost += lambda_ * in_collision.sum(dim=1)
        print(f"COLLISION COST: {time() - start}")
        cost += collision_cost

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
