import sys
import rospy
import numpy as np
import pybullet
import torch
from torch.distributions import MultivariateNormal, Normal, Uniform
from time import time

from sensor_msgs.msg import JointState

from kl_planning.environments.resources import Panda, CollisionChecker
from kl_planning.util import file_util, math_util, vis_util


class ArmEnvironment:

    def __init__(self, config, m_projection=False, belief_dynamics_noise=0.02,
                 device=torch.device('cuda'), debug=False):
        self.object_config = config['objects']
        self.agent_config = config['agent']
        self.start_config = config['start']
        self.goal_config = config['goals']
        self.indicator_config = config['indicators']
        self.state_size = len(self.start_config['state'])

        # TODO taking one explicitly, this won't work if you have multiple goals
        self.desired_position = torch.tensor(self.goal_config['goal']['position'])
        self.desired_position = self.desired_position.unsqueeze(0).to(device)
        self.desired_orientation = torch.tensor(self.goal_config['goal']['orientation'])
        self.desired_orientation = self.desired_orientation.unsqueeze(0).to(device)
        
        self.m_projection = m_projection
        self.belief_dynamics_noise = belief_dynamics_noise
        self.device = device

        # Robot arm
        pybullet.connect(pybullet.GUI) if debug else pybullet.connect(pybullet.DIRECT)
        self.arm = Panda()
        for i in range(self.arm.num_joints):
            pybullet.resetJointState(self.arm.robot_id, i, self.start_config['state'][i])

        # Collision environment
        collision_objects = self._create_collision_objects()
        self.collision_checker = CollisionChecker(self.arm, collision_objects,
                                                  self.start_config['state'], debug)
        if debug:
            self.collision_checker.run_debug()

        # ROS
        self.joint_state = JointState()
        self.joint_state.name = [f'panda_joint{i}' for i in range(1,8)]
        self.joint_state.name += [f'panda_finger_joint{i}' for i in [1,2]]
        self.joint_state.position = [0] * len(self.joint_state.name)
        self.joint_state_pub = rospy.Publisher("/joint_states", JointState, queue_size=1)
        rospy.sleep(1)
        self.joint_state.header.stamp = rospy.Time.now()
        self.joint_state_pub.publish(self.joint_state)

    def get_start_state(self):
        return self.start_config['state']

    def get_goal_states(self):
        return [v['state'] for v in self.goal_config.values()]

    def get_goal_covariances(self):
        return [v['covariance'] for v in self.goal_config.values()]

    def set_agent_location(self, q):
        self.joint_state.position = q.tolist() + [0.04, 0.04] # Add gripper finger positions
        self.joint_state.header.stamp = rospy.Time.now()
        self.joint_state_pub.publish(self.joint_state)
    
    def dynamics(self, state, act, noise_gain=0.02):
        """
        Dynamics for arm are simply noisy versions of what is commanded.
        """
        new_state = state + act + torch.randn_like(state) * noise_gain
        return new_state

    def in_collision(self, q):
        collision = []
        n_samples = q.size(0)
        n_state = q.size(-1)
        for i in range(n_samples):
            for j in range(n_state):
                pybullet.resetJointState(self.arm.robot_id, j, q[i, j])
            collision.append(self.collision_checker.in_contact())            
        return torch.tensor(collision, device=self.device)

    def cost(self, act, start_dist, goal_dist, kl_divergence):
        n_candidates = act.size(1)
        n_state = start_dist.loc.size(-1)
        n_sigma = 2 * n_state + 1
        
        mus = [start_dist.loc.repeat(n_candidates, 1).to(self.device)]
        sigmas = [start_dist.covariance_matrix.repeat(n_candidates, 1, 1).to(self.device)]
        sigma_points = []

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
                    kl_cost_t = kl_cost_t.sum(dim=-1)
                kl_cost += kl_cost_t
            else:
                kl_cost += lambda_ * kl_divergence(p_t, goal_dist)

        # print("KL", kl_cost)
        
        cost += kl_cost
            
        # Compute collision costs based on sigma points
        collision_cost = 0
        n_points = float(len(sigma_points))
        for i, Y in enumerate(sigma_points):
            B = Y.size(0)
            n_sigma = Y.size(1)
            in_collision = self.in_collision(Y.view(B * n_sigma, -1)) # TODO parameterize
            in_collision = in_collision.view(B, n_sigma)
            collision_cost += in_collision.sum(dim=1)
        cost += collision_cost * 10000.0

        # TODO trying a cost on desired EE pose
        if self.desired_position is not None and self.desired_orientation is not None:
            ee_cost = 0
            for i, Y in enumerate(sigma_points):
                lambda_ = (i + 1) / float(len(sigma_points))
                B = Y.size(0)
                n_sigma = Y.size(1)
                ps, qs = self.arm.fk(Y.view(B * n_sigma, -1))
                ee_cost += lambda_ * self.ee_pose_error(ps, qs).view(B, n_sigma).sum(dim=1)

            cost += ee_cost * 10.0  # TODO figure out weighting

        # print("EE COST", ee_cost)

        return cost

    def ee_pose_error(self, ps, qs):
        """
        ps (B, 3)
        qs (B, 4)
        """
        B = ps.size(0)
        p_desired = self.desired_position.repeat(B, 1)
        q_desired = self.desired_orientation.repeat(B, 1)
        p_error = self.arm.position_error(p_desired, ps)
        # q_error = self.arm.orientation_error(q_desired, qs)
        error = p_error # + q_error
        return error

    def visualize_samples(self, start_state, samples, costs=None):
        pass

    def _create_collision_objects(self):
        objects = []
        for obj_id, obj_data in self.object_config.items():
            if obj_data['type'] == 'cylinder':
                objects.append((pybullet.GEOM_CYLINDER, obj_data['radius'],
                                obj_data['height'], obj_data['position']))
            else:
                raise ValueError(f"Unknown collision object type: {obj_data['type']}")
        return objects
