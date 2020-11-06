import sys
import rospy
import torch
from torch.distributions import MultivariateNormal, Normal, Uniform
from scipy.spatial.transform import Rotation as R
import numpy as np

from kl_planning.environments import Environment
from kl_planning.util import file_util, math_util, vis_util
from kl_planning.srv import SetPose, SetPoseRequest


class Navigation2DEnvironment(Environment):
    """
    2D environment for Dubins car navigation.
    """

    def __init__(self, config, m_projection=False, belief_dynamics_noise=0.02,
                 device=torch.device('cuda')):
        super().__init__(config, m_projection, belief_dynamics_noise, device)        
        self._create_collision_checkers()
        
    def set_agent_location(self, state, interpolate=False, z_pos=0.01):
        """
        Set the cars 2D planar pose in the scene. Makes request to visualization service.

        Args:
            state (list): Planar pose to set as current location (x,y,theta)
            interpolate (bool): Interpolate between current and new state if True
            z_pos (float): Fixed z-position (height) above environment x-y plane
        """
        req = SetPoseRequest()
        req.interpolate = interpolate
        req.pose.position.x = state[0]
        req.pose.position.y = state[1]
        req.pose.position.z = z_pos
        quat = R.from_euler('z', state[2], degrees=False).as_quat()
        req.pose.orientation.x = quat[0]
        req.pose.orientation.y = quat[1]
        req.pose.orientation.z = quat[2]
        req.pose.orientation.w = quat[3]
        set_pose = rospy.ServiceProxy("/visualization/set_agent_location", SetPose)
        try:
            set_pose(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call to set agent location failed: {e}")
        
    def dynamics(self, state, act, noise_gain=0.02):
        """
        Dubins car dynamics.

        Args:
            state (Tensor): Start state of shape (n_batch, n_state)
            act (Tensor): Action to apply of shape (n_batch, n_act)
            noise_gain (float): Gain factor for additive Gaussian noise to dynamics
        Returns:
            next_state (Tensor): Next state after applying action on current state and feeding
                                 through nonlinear stochastic dynamics of shape (n_batch, n_state)
        """
        # Adding small value so it doesn't divide by zero, this avoids having to do boolean
        # check on all values, will get washed out in noisy dynamics anyways
        u = act[:,0] + 1e-8
        v = act[:,1]
        dt = act[:,2]
        dt_u = dt * u
        theta = state[:,2]

        next_state = state.detach().clone()
        next_state[:,0] += (v / u) * (torch.sin(theta + dt_u) - torch.sin(theta))
        next_state[:,1] += (v / u) * (torch.cos(theta) - torch.cos(theta + dt_u))
        next_state[:,2] += dt_u
        
        # Add in noise on resulting state to model stochastic transition
        next_state += torch.randn_like(next_state) * noise_gain
        
        return next_state

    def get_trajectory(self, start_state, actions, noise_gain=0.0):
        """
        Applies dynamics from start state with action sequence to get trajectories.

        Args:
            start_state (Tensor): Start to execute action trajectories from (n_state,)
            actions (Tensor): Action trajectories to apply (n_time, n_trajs, n_act)
            noise_gain (float): Gain factor on Gaussian dynamics noise
        Returns:
            trajs (Tensor): Computed trajectories from action sequences (n_time, n_trajs, n_state)
        """
        T = actions.size(0)
        n_trajs = actions.size(1)
        start_state = start_state.repeat(n_trajs, 1)
        trajs = torch.zeros(T+1, n_trajs, 3, device=self.device)

        trajs[0] = start_state
        for t in range(T):
            trajs[t+1] = self.dynamics(trajs[t], actions[t], noise_gain=noise_gain)
        return trajs

    def in_collision(self, state):
        """
        Checks (in batch) if a state is in collision.

        Args:
            state (Tensor): States to check for collision (n_batch, n_state)
        Returns:
            in_collision (Tensor): Boolean tensor True if in collision False otherwise (n_batch,)
        """
        in_collision = torch.zeros(state.size(0), dtype=torch.bool, device=self.device)
        for obj_id, checker in self.collision_checkers.items():
            obj_in_collision = checker(state)
            in_collision += obj_in_collision  # Boolean OR operation
        return in_collision
    
    def cost(self, act, start_dist, goal_dist, kl_divergence=None):
        """
        Cost function for ranking samples. Combined cost of KL divergence between start
        state distribution and goal distribution and collision cost.

        Args:
            act (Tensor): Actions to be applied of shape (horizon, n_candidates, n_act)
            start_dist (distribution): Start state distribution
            goal_dist (distribution): Goal state distribution
            kl_divergence (function): KL divergence function defined for state/goal distributions.
        Returns:
            cost (Tensor): Computed costs of shape (n_candidates,)
        """
        if kl_divergence is None:
            kl_divergence = torch.distributions.kl.kl_divergence

        n_candidates = act.size(1)
        n_state = start_dist.loc.size(-1)
        n_sigma = 2 * n_state + 1
        
        mus = [start_dist.loc.repeat(n_candidates, 1).to(self.device)]
        sigmas = [start_dist.covariance_matrix.repeat(n_candidates, 1, 1).to(self.device)]
        sigma_points = []

        # Apply action sequences through unscented transform to do uncertainty propagation
        for t in range(len(act)):
            act_t = act[t].unsqueeze(1).repeat(1, n_sigma, 1)
            act_t = act_t.view(act_t.size(0) * act_t.size(1), -1)
            g = lambda x: self.dynamics(x, act_t, self.belief_dynamics_noise)
            mu_prime, sigma_prime, Y = math_util.unscented_transform(mus[-1], sigmas[-1], g,
                                                                     device=self.device)
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
            if isinstance(goal_dist, Uniform):
                p_t = Normal(mus[t], torch.diagonal(sigmas[t], dim1=-2, dim2=-1).to(self.device))
            else:
                p_t = MultivariateNormal(mus[t], sigmas[t])

            if self.m_projection:
                kl_cost_t = lambda_ * kl_divergence(goal_dist, p_t)
                if isinstance(goal_dist, Uniform):
                    kl_cost_t = kl_cost_t.sum(dim=-1) / 10000.0 # Uniform KL is huge
                kl_cost += kl_cost_t
            else:
                kl_cost += lambda_ * kl_divergence(p_t, goal_dist)
        cost += kl_cost
            
        # Compute collision costs based on sigma points
        collision_cost = 0
        n_points = float(len(sigma_points))
        for i, Y in enumerate(sigma_points):
            B = Y.size(0)
            n_sigma = Y.size(1)
            in_collision = self.in_collision(Y.view(B * n_sigma, -1)) * 10000.0
            in_collision = in_collision.view(B, n_sigma)
            collision_cost += in_collision.sum(dim=1)
        cost += collision_cost

        return cost

    def purge_bad_samples(self, samples, start_dist):
        """
        This will discard any samples that have sigma points in collision. This is alternative
        to accumulating cost for collisions so that you only take collision-free samples and
        accumulate KL cost.

        Returns only samples that have no sigma points in collision.

        Note: I tried this, didn't seem to work well. I think it takes too long to find
              completely collision-free samples without a good action distribution initialization.
        """
        n_candidates = samples.size(1)
        n_state = start_dist.loc.size(-1)
        n_sigma = 2 * n_state + 1
        
        mus = [start_dist.loc.repeat(n_candidates, 1).to(self.device)]
        sigmas = [start_dist.covariance_matrix.repeat(n_candidates, 1, 1).to(self.device)]
        sigma_points = []
        
        for t in range(len(samples)):
            sample_t = samples[t].unsqueeze(1).repeat(1, n_sigma, 1)
            sample_t = sample_t.view(sample_t.size(0) * sample_t.size(1), -1)
            g = lambda x: self.dynamics(x, sample_t, self.belief_dynamics_noise)
            mu_prime, sigma_prime, Y = math_util.unscented_transform(mus[-1], sigmas[-1], g,
                                                                     device=self.device)
            mus.append(mu_prime)
            sigmas.append(sigma_prime)
            sigma_points.append(Y)

        collision = 0
        for i, Y in enumerate(sigma_points):
            B = Y.size(0)
            n_sigma = Y.size(1)
            in_collision = self.in_collision(Y.view(B * n_sigma, -1))
            in_collision = in_collision.view(B, n_sigma)
            collision += in_collision.sum(dim=-1)

        return samples[:,collision == 0]
        
    def visualize_samples(self, start_state=None, samples=None, costs=None,
                          colors=None, size=0.01, sleep=0):
        """
        Visualize trajectory samples in rviz.

        Args:
            start_state (Tensor): Start to execute action trajectories from (n_state,)
            samples (Tensor or array): Trajectory samples of shape (n_time, n_samples, n_state)
            costs (Tensor or array): Vector of costs associated with each sample
            colors (list): List of colors (RGBA tuples) to color samples, of length n_samples
            size (float): Width of lines to display in rviz
            sleep (float): Time to sleep after displaying samples
        """
        if start_state is not None and samples is not None:
            samples = self.get_trajectory(start_state, samples)
        vis_util.visualize_trajectory_samples(samples, costs=costs, colors=colors, size=size)
        if sleep > 0:
            rospy.sleep(sleep)

    def _create_collision_checkers(self, buffer_=0.1):
        """
        Creates collision checkers to determine if robot is in collision with environment.

        Simple polygonal checks to test if robot in collision, right now only defined 
        for cube obstacles and cube agent. Can add additional shapes as needed.

        Args:
            buffer_ (float): Buffer to pad collision determination from agent's body.
        """
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
        """
        Factory for creating collision check functions for cubes. Function will compute boolean
        test to determine if agent's body is in contact with (or penetrating) a cube obstacle.

        Args:
            x_l (float): Low x-dimension for boundary of obstacle
            x_h (float): High x-dimension for boundary of obstacle
            y_l (float): Low y-dimension for boundary of obstacle
            y_h (float): High y-dimension for boundary of obstacle
        """
        # Multiplication on bool tensors is AND operator
        return lambda x: (x[:,0] > x_l) * (x[:,0] < x_h) * (x[:,1] > y_l) * (x[:,1] < y_h)
