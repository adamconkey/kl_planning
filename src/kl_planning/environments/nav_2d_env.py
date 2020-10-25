import sys
import rospy
import torch
from torch.distributions import MultivariateNormal, Normal
from scipy.spatial.transform import Rotation as R
import numpy as np
from time import time

from kl_planning.util import file_util, math_util
from kl_planning.srv import SetPose, SetPoseRequest


class Navigation2DEnvironment:

    def __init__(self, config, m_projection=False, belief_dynamics_noise=0.02):
        self.object_config = config['objects']
        self.agent_config = config['agent']
        self.indicator_config = config['indicators']
        self.start_config = config['start']
        self.goal_config = config['goals']
        self.state_size = len(self.start_config['state'])

        self.m_projection = m_projection
        self.belief_dynamics_noise = belief_dynamics_noise
        
        self._create_collision_checkers()

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

    def get_goal_low_high(self):
        """
        This is for uniform distribution.
        """
        return self.goal_config['goal']['low'], self.goal_config['goal']['high']
        
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
        u = act[:,0] + 1e-8
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
            in_collision += obj_in_collision  # Boolean OR operation
        return in_collision
    
    def cost(self, act, start_dist, goal_dist, kl_divergence=None):
        if kl_divergence is None:
            kl_divergence = torch.distributions.kl.kl_divergence

        n_candidates = act.size(1)
        n_state = start_dist.loc.size(-1)
        n_sigma = 2 * n_state + 1
        
        mus = [start_dist.loc.repeat(n_candidates, 1)]
        sigmas = [start_dist.covariance_matrix.repeat(n_candidates, 1, 1)]
        sigma_points = []

        for t in range(len(act)):
            act_t = act[t].unsqueeze(1).repeat(1, n_sigma, 1)
            act_t = act_t.view(act_t.size(0) * act_t.size(1), -1)
            g = lambda x: self.dynamics(x, act_t, self.belief_dynamics_noise)
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
            # TODO for now diagonalizing as there is no general MVN implementation of KL
            if isinstance(goal_dist, torch.distributions.uniform.Uniform):
                p_t = Normal(mus[t], torch.diagonal(sigmas[t], dim1=-2, dim2=-1))
            else:
                p_t = MultivariateNormal(mus[t], sigmas[t])

            if self.m_projection:
                kl_cost_t = lambda_ * kl_divergence(goal_dist, p_t)
                if isinstance(goal_dist, torch.distributions.uniform.Uniform):
                    # TODO scaling because uniform is huge, maybe parameterize this
                    kl_cost_t = kl_cost_t.sum(dim=-1) #  / 10.0
                kl_cost += kl_cost_t
            else:
                kl_cost += lambda_ * kl_divergence(p_t, goal_dist)
        cost += kl_cost
            
        # Compute collision costs based on sigma points
        collision_cost = 0
        n_points = float(len(sigma_points))
        for i, Y in enumerate(sigma_points):
            # # Decreasing contribution of collision cost as time increases
            # lambda_ = (n_points - i) / n_points
            B = Y.size(0)
            n_sigma = Y.size(1)
            in_collision = self.in_collision(Y.view(B * n_sigma, -1)) * 100.0
            in_collision = in_collision.view(B, n_sigma)
            collision_cost += in_collision.sum(dim=1)
        cost += collision_cost

        return cost

    def visualize_samples(self, start_state, samples, costs=None, size=0.03, sleep=0):
        elite_samples = self.get_trajectory(start_state, samples)
        vis_util.visualize_line_trajectory_samples(elite_samples, costs, size)
        if sleep > 0:
            rospy.sleep(sleep)

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
