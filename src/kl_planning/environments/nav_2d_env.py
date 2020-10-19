import sys
import rospy
import torch
from torch.distributions import MultivariateNormal
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
        self.start_config = scene_config['start']
        self.goal_config = scene_config['goals']

        self.wheel_radius = self.agent_config['agent']['wheel_radius']
        self.robot_length = self.agent_config['agent']['length']
        
        self._create_collision_checkers()

        # These save some lookups/computations in collision check
        L = self.agent_config['agent']['length']
        W = self.agent_config['agent']['width']
        self.p0 = np.array([-L / 2., W / 2.])
        self.p1 = self.p0 + np.array([L, 0])
        self.p2 = self.p1 + np.array([0, -W])
        self.p3 = self.p2 + np.array([-L, 0])

    def get_start_state(self):
        return self.start_config['state']

    def get_start_covariance(self):
        return self.start_config['covariance']

    def get_goal_states(self):
        return [v['state'] for v in self.goal_config.values()]

    def get_goal_covariances(self):
        return [v['covariance'] for v in self.goal_config.values()]

    def get_goal_weights(self):
        return [v['weight'] for v in self.goal_config.values()]
        
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

        Trying parameterization from CEMP where action at each timestep is a turn
        rate and duration for that action. A constant velocity is given such that
        if turn rate is zero then it goes in straight line at a constant velocity.
        """
        # Adding small value so it doesn't divide by zero, this avoids having to do boolean
        # check on all values, will get washed out in noisy dynamics anyways
        u = act[:,0] + 1e-5
        v = act[:,1]
        dt = act[:,2]
        dt_u = dt * u
        theta = start_pose[:,2]

        next_pose = start_pose.detach().clone()
        next_pose[:,0] += (v / u) * (torch.sin(theta + dt_u) - torch.sin(theta))
        next_pose[:,1] += (v / u) * (torch.cos(theta) - torch.cos(theta + dt_u))
        next_pose[:,2] += dt_u
        
        # Add in noise on resulting state to model stochastic transition
        next_pose += torch.randn_like(next_pose) * noise_gain
        
        return next_pose

    def get_trajectory(self, start_state, actions, noise_gain=0.0):
        """
        Applies dynamics from start state with action sequence to get trajectories.
        """
        T = actions.size(0)
        n_trajs = actions.size(1)
        start_state = start_state.repeat(n_trajs, 1)
        trajs = torch.zeros(T+1, n_trajs, 3)

        trajs[0] = start_state
        for t in range(T):
            trajs[t+1] = self.dynamics(trajs[t], actions[t], noise_gain=noise_gain)
        return trajs

    def in_collision(self, q):
        in_collision = torch.zeros(q.size(0), dtype=torch.bool)
        for obj_id, checker in self.collision_checkers.items():
            obj_in_collision = checker(q)

            # print(obj_id, obj_in_collision)
            
            in_collision += obj_in_collision  # Boolean OR operation
        # print("FINAL", in_collision)
        return in_collision

    def euclidean_cost(self, act, start_dist, goal_dist, lambda_=1.0, noise_gain=0.0):
        cost = 0
        trajs = self.get_trajectory(start_dist.loc, act, noise_gain)[1:] # Exclude start state
        for t in range(len(trajs)):
            # Euclidean distance cost to goal
            cost += 1 + lambda_ * (goal_dist.loc - trajs[t]).square().sum(dim=-1)
            # Collision cost
            cost += self.in_collision(trajs[t]) * 100.0
        return cost
    
    def kl_cost(self, act, start_dist, goal_dist, kl_divergence=None):
        if kl_divergence is None:
            kl_divergence = torch.distributions.kl.kl_divergence

        n_candidates = act.size(1)
        n_sigma = 2 * act.size(-1) + 1
        
        mus = [start_dist.loc.repeat(n_candidates, 1)]
        sigmas = [start_dist.covariance_matrix.repeat(n_candidates, 1, 1)]
        sigma_points = []

        state_size = start_dist.loc.size(-1)

        for t in range(len(act)):
            act_t = act[t].unsqueeze(1).repeat(1, 2 * state_size + 1, 1)
            act_t = act_t.view(act_t.size(0) * act_t.size(1), -1)
            g = lambda x: self.dynamics(x, act_t)
            mu_prime, sigma_prime, Y = math_util.unscented_transform(mus[-1], sigmas[-1], g)
            mus.append(mu_prime)
            sigmas.append(sigma_prime)
            sigma_points.append(Y)
            
        cost = 0

        # Compute KL cost from final distribution to goal distribution
        kl_cost = 0
        T = len(mus)
        for t in range(T):
            # Increasing contribution of KL cost as time increases
            lambda_ = (t + 1) / float(T)
            p_t = MultivariateNormal(mus[t], sigmas[t])
            kl_cost += lambda_ * kl_divergence(p_t, goal_dist, n_candidates) # I-projection
            # kl_cost += lambda_ * kl_divergence(goal_dist, p_t, n_candidates) # M-projection
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
            in_collision = self.in_collision(Y.view(B * n_sigma, -1)) * 100.0
            in_collision = in_collision.view(B, n_sigma)
            collision_cost += in_collision.sum(dim=1)
        cost += collision_cost

        # print("COST", cost)

        return cost

    def _create_collision_checkers(self, buffer_=0.1):
        self.collision_checkers = {}
        for obj_id, obj_data in self.object_config.items():
            if obj_data['type'] == 'cube':
                # TODO this is simplified for now to make it fast to check, assumes cubes are
                # axis aligned to world. Can make more general if necessary.
                origin = obj_data['position']
                L = obj_data['length']
                W = obj_data['width']
                x_l = origin[0] - (L / 2.) - buffer_
                x_h = origin[0] + (L / 2.) + buffer_
                y_l = origin[1] - (W / 2.) - buffer_
                y_h = origin[1] + (W / 2.) + buffer_
                self.collision_checkers[obj_id] = self._collision_checker_factory(x_l, x_h, y_l, y_h)
            else:
                print(f"Unknown object type for making collision object: {obj_data['type']}")    

    def _collision_checker_factory(self, x_l, x_h, y_l, y_h):
        # Multiplication on bool tensors is AND operator
        return lambda x: (x[:,0] > x_l) * (x[:,0] < x_h) * (x[:,1] > y_l) * (x[:,1] < y_h)
