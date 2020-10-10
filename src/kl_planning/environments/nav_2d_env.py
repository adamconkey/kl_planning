import sys
import torch
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence

import numpy as np

from kl_planning.util import file_util, math_util


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
        
    def dynamics(self, start_xy, act, repeat=True):
        """
        start_xy (b, 2) x, y
        act (b, 2) dist, angle
        """
        delta_x = act[:,0] * torch.cos(act[:,1])
        delta_y = act[:,0] * torch.sin(act[:,1])
        delta_xy = torch.stack([delta_x, delta_y], dim=-1).unsqueeze(1)
        if repeat:
            delta_xy = delta_xy.repeat(1, 5, 1) # 2n+1 = 5 sigma points, just hard-coding for now
        next_xy = start_xy + delta_xy
        return next_xy

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
        robot_pts = self.fk(q)
        robot_edges = self._edges_of(robot_pts)
        for polygon in self.polygons:
            for o in [self._orthogonal(e) for e in robot_edges + self._edges_of(polygon)]:
                if self._is_separating_axis(o, robot_pts, polygon):
                    return False
        return True
            
    def _edges_of(self, vertices):
        """
        Return the vectors for the edges of the polygon p.
        
        p is a polygon.
        """
        edges = []
        N = len(vertices)
        for i in range(N):
            edge = vertices[(i + 1) % N] - vertices[i]
            edges.append(edge)
        return edges

    def _orthogonal(self, v):
        """
        Return a 90 degree clockwise rotation of the vector v.
        """
        return np.array([-v[1], v[0]])

    def _is_separating_axis(self, o, p1, p2):
        """
        Return True if o is a separating axis of p1 and p2.
        """
        min1, max1 = float('+inf'), float('-inf')
        min2, max2 = float('+inf'), float('-inf')
        for v in p1:
            projection = np.dot(v, o)
            min1 = min(min1, projection)
            max1 = max(max1, projection)
        for v in p2:
            projection = np.dot(v, o)
            min2 = min(min2, projection)
            max2 = max(max2, projection)    
        return max1 < min2 or max2 < min1

    def cost(self, act, start_mu, start_sigma, goal_mu, goal_sigma):
        mus = [start_mu]
        sigmas = [start_sigma]
        sigma_points = []
        
        for t in range(len(act)):
            g = lambda x: self.dynamics(x, act[t])
            mu_prime, sigma_prime, Y = math_util.unscented_transform(mus[-1], sigmas[-1], g)
            mus.append(mu_prime)
            sigmas.append(sigma_prime)
            sigma_points.append(Y)

        cost = 0
        
        # Compute KL cost from final distribution to goal distribution
        p_G = MultivariateNormal(goal_mu, goal_sigma)
        T = len(mus)
        for t in range(T):
            lambda_ = (t+1) / T
            p_t = MultivariateNormal(mus[t], sigmas[t])
            cost += lambda_ * kl_divergence(p_G, p_t)
            
        # Compute collision costs based on sigma points
        n_points = float(len(sigma_points))
        for i, Y in enumerate(sigma_points):
            lambda_ = (n_points - i) / n_points
            B = Y.size(0)
            n_sigma = Y.size(1)
            Y = Y.view(B * n_sigma, -1)
            # TODO ideally you can batch compute the collisions, will need to rewrite the
            # collision check code though
            in_collision = torch.zeros(B * n_sigma)
            for i in range(len(in_collision)):
                in_collision[i] = float(self.in_collision(Y[i])) * 100.0
            # TODO for now just trying a simple summing
            cost += lambda_ * in_collision.view(B, n_sigma).sum(dim=1)
        
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
