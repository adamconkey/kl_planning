import sys
import rospy
import torch

import matplotlib.pyplot as plt
from tqdm import tqdm

from kl_planning.util import vis_util


class Planner:

    def __init__(self):
        pass

    def plan_cem(self, env, start_dist, goal_dist, min_act, max_act,
                 horizon=10, n_iters=10, n_candidates=1000, n_elite=10,
                 visualize=False):
        act_size = min_act.size(-1)
        
        # start_mu = start_mu.repeat(n_candidates, 1)
        # start_sigma = start_sigma.repeat(n_candidates, 1, 1)
        
        act_mu = torch.zeros(horizon, 1, act_size)
        # TODO trying full time for euclidean distance case
        act_mu[:,:,-1] = max_act[-1]
        
        act_sigma = torch.ones(horizon, 1, act_size)
    
        best_costs = []
        worst_costs = []
        
        for _ in tqdm(range(n_iters)):
            # Generate action delta samples
            noise = torch.randn(horizon, n_candidates, act_size)
            act = act_mu + act_sigma * noise
            act = torch.max(torch.min(act, max_act), min_act)
            if visualize:
                trajs = env.get_trajectory(start_dist.loc, act)
                vis_util.visualize_trajectory_samples(trajs, size=0.005)
                # rospy.sleep(1)
                
            # Find top K low-cost action sequences
            costs = env.kl_cost(act, start_dist, goal_dist)
            # costs = env.euclidean_cost(act, start_dist, goal_dist, noise_gain=0.0)
            topk_costs, topk_indices = costs.topk(n_elite, dim=-1, largest=False, sorted=False)    
            elite = act[:, topk_indices]

            if visualize:
                trajs = env.get_trajectory(start_dist.loc, elite)
                vis_util.visualize_trajectory_samples(trajs, topk_costs)
                # rospy.sleep(1)
            
            # Update belief with new means and standard deviations
            act_mu = elite.mean(dim=1, keepdim=True)
            act_sigma = elite.std(dim=1, keepdim=True)
    
        return act_mu.squeeze()
