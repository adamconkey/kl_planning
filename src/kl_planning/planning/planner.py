import sys
import rospy
import torch

import matplotlib.pyplot as plt
from tqdm import tqdm

from kl_planning.distributions import GaussianMixture
from kl_planning.util import vis_util, ui_util


ACT_DISTRIBUTION_TYPES = ['gaussian', 'gmm']


class Planner:

    def plan_cem(self, env, start_dist, goal_dist, min_act, max_act,
                 horizon=10, n_iters=10, n_candidates=1000, n_elite=10,
                 n_components=2, kl_divergence=None, act_dist_type='gaussian',
                 visualize=False):
        """
        TODO: Introducing a distribution abstraction here would really clean things 
              up, instead of doing these switches everywhere on the distribution type.
        """
        if act_dist_type not in ACT_DISTRIBUTION_TYPES:
            ui_util.print_error(f"\nUnknown action distribution type for CEM: {act_dist_type}\n")
            return None

        # Initialize action distribution for sampling trajectories
        act_size = min_act.size(-1)
        if act_dist_type == 'gaussian':
            act_mu = torch.zeros(horizon, 1, act_size)
            act_mu[:,:,-1] = max_act[-1] # Initialize prior time durations as max possible
            act_sigma = torch.ones(horizon, 1, act_size)
        elif act_dist_type == 'gmm':
            act_mu = torch.zeros(horizon, act_size)
            act_mu[:,-1] = max_act[-1] # Initialize prior time durations as max possible
            act_mu = act_mu.view(1, 1, horizon * act_size).repeat(1, n_components, 1)
            act_sigma = torch.ones(1, n_components, horizon * act_size)
            act_dist = GaussianMixture(n_components, horizon * act_size, act_mu, act_sigma)
            
        for _ in tqdm(range(n_iters), desc='CEM'):
            # Generate action delta samples
            if act_dist_type == 'gaussian':
                noise = torch.randn(horizon, n_candidates, act_size)
                act = act_mu + act_sigma * noise
            elif act_dist_type == 'gmm':
                act = act_dist.sample(n_candidates)
                act = act.view(horizon, n_candidates, act_size)
            act = torch.max(torch.min(act, max_act), min_act)
                
            if visualize:
                trajs = env.get_trajectory(start_dist.loc, act)
                vis_util.visualize_trajectory_samples(trajs, size=0.005)
                # rospy.sleep(1)
                
            # Find top K low-cost action sequences
            costs = env.kl_cost(act, start_dist, goal_dist, kl_divergence)
            topk_costs, topk_indices = costs.topk(n_elite, dim=-1, largest=False, sorted=False)    
            elite = act[:, topk_indices]

            if visualize:
                trajs = env.get_trajectory(start_dist.loc, elite)
                vis_util.visualize_trajectory_samples(trajs, topk_costs)
                # rospy.sleep(1)
                
            # Update belief with new means and standard deviations
            if act_dist_type == 'gaussian':
                act_mu = elite.mean(dim=1, keepdim=True)
                act_sigma = elite.std(dim=1, keepdim=True)
            elif act_dist_type == 'gmm':
                elite = elite.view(n_elite, horizon * act_size)
                act_dist.fit(elite, delta=1.0, n_iter=10)
                # TODO just taking most likely for now
                best_idx = torch.argmax(act_dist.pi.flatten()).item()
                act_mu = act_dist.mu.squeeze()[best_idx].view(horizon, 1, act_size)
                
    
        return act_mu.squeeze()
