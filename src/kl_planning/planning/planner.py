import sys
import rospy
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.mixture import GaussianMixture

# from kl_planning.distributions import GaussianMixture
from kl_planning.util import vis_util, ui_util


ACT_DISTRIBUTION_TYPES = ['gaussian', 'gmm']


class Planner:

    def plan_cem(self, env, start_dist, goal_dist, min_act, max_act, device,
                 horizon=10, n_iters=10, n_candidates=1000, n_elite=10,
                 n_components=2, kl_divergence=None, act_dist_type='gaussian',
                 visualize=False, **kwargs):
        """
        TODO: Introducing a distribution abstraction here would really clean things 
              up, instead of doing these switches everywhere on the distribution type.
              Can then also return just the end distribution instead of these
              conditional return types I currently have.
        """
        if act_dist_type not in ACT_DISTRIBUTION_TYPES:
            ui_util.print_error(f"\nUnknown action distribution type for CEM: {act_dist_type}\n")
            return None

        # Initialize action distribution for sampling trajectories
        act_size = min_act.size(-1)
        if act_dist_type == 'gaussian':
            act_mu = torch.zeros(horizon, 1, act_size, device=device)
            act_mu[:,:,-1] = max_act[-1] # Initialize prior time durations as max possible
            act_sigma = torch.ones(horizon, 1, act_size, device=device)
        elif act_dist_type == 'gmm':
            act_mu = torch.zeros(horizon, act_size)
            act_mu[:,-1] = max_act[-1] # Initialize prior time durations as max possible
            act_mu = act_mu.view(1, horizon * act_size).repeat(n_components, 1)
            act_sigma = torch.ones(n_components, horizon * act_size)
            # act_mu = act_mu.view(1, 1, horizon * act_size).repeat(1, n_components, 1)
            # act_sigma = torch.ones(1, n_components, horizon * act_size)
            # act_dist = GaussianMixture(n_components, horizon * act_size, act_mu, act_sigma)
            act_dist = GaussianMixture(n_components,
                                       covariance_type='diag',
                                       init_params='kmeans',
                                       max_iter=10)
            act_dist.means_ = act_mu.numpy()
            act_dist.covariances_ = act_sigma.numpy()
            act_dist.weights_ = np.ones(n_components) / float(n_components)
            act_dist.precisions_cholesky_ = None
            
        for _ in tqdm(range(n_iters), desc='CEM'):
            # Generate action delta samples
            if act_dist_type == 'gaussian':
                noise = torch.randn(horizon, n_candidates, act_size, device=device)
                act = act_mu + act_sigma * noise                
            elif act_dist_type == 'gmm':
                act, component_labels = act_dist.sample(n_candidates)
                act = torch.from_numpy(act).float()
                act = act.view(horizon, n_candidates, act_size)
            act = torch.max(torch.min(act, max_act), min_act) # (horizon, candidates, act)
            # print("ACT", act)
                
            # Find top K low-cost action sequences
            costs = env.cost(act, start_dist, goal_dist, kl_divergence, device=device, **kwargs)
            topk_costs, topk_indices = costs.topk(n_elite, dim=-1, largest=False, sorted=False)    
            elite = act[:, topk_indices]

            # print("ELITE", elite)

            if visualize:
                # means = torch.from_numpy(act_dist.means_).view(horizon, 2, -1)
                # env.visualize_samples(start_dist.loc, means, size=0.07)
                # env.visualize_samples(start_dist.loc, act, size=0.005)
                env.visualize_samples(start_dist.loc, elite, topk_costs)
                
            # Update belief with new means and standard deviations
            if act_dist_type == 'gaussian':
                act_mu = elite.mean(dim=1, keepdim=True)
                act_sigma = elite.std(dim=1, keepdim=True)
                plan_return = act_mu.squeeze()
                # print("MEAN", act_mu)
                # print("PLAN RETURN", plan_return)
            elif act_dist_type == 'gmm':
                elite = elite.view(n_elite, horizon * act_size).numpy()
                act_dist.fit(elite)
                plan_return = act_dist
                
        return plan_return
