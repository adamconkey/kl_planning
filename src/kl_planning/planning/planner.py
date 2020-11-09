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
                 visualize=False, purge_bad_samples=False, **kwargs):
        """
        Runs CEM optimization to generate a plan action sequence for minimizing costs
        between a current state distribution and a goal distribution.
        
        Args:
            env (Environment): Environment object that has cost function defined
            start_dist (distribution): Start state distribution
            goal_dist (distribution): Goal state distribution
            min_act (Tensor): Minimum action values to clamp to
            max_act (Tensor): Maximum action values to clamp to
            device (device): Torch device to perform computations on
            horizon (int): Planning horizon
            n_iters (int): Number of CEM iterations to run
            n_candidates (int): Number of candidate samples to evaluate
            n_elite (int): Number of elite samples to fit for the next iteration
            n_components (int): Number of GMM components in planner (only used if planner is GMM)
            kl_divergence (func): KL divergence function to compute between distributions
            act_dist_type (str): Type of planner distribution (gaussian or gmm)
            visualize (bool): Samples are visualized if True
            purge_bad_samples (bool): Removes bad samples (e.g. in collision) if True. Note this
                                      was experimented with but proved to take too long.
           
        Returns:
            plan_return: Returns mean (Tensor) if planner is gaussian, and the full action
                         distribution object if gmm (stores component weights and parameters)

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
            act_dist = GaussianMixture(n_components, covariance_type='diag',
                                       init_params='kmeans', max_iter=10)
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
                if purge_bad_samples:
                    act = torch.zeros(horizon, 0, act_size, device=device)
                    while act.size(1) < n_candidates:
                        samples, labels = act_dist.sample(n_candidates)
                        samples = torch.from_numpy(samples).float().to(device)
                        samples = samples.view(horizon, n_candidates, act_size)
                        good_samples = env.purge_bad_samples(samples, start_dist)
                        act = torch.cat([act, good_samples], dim=1)
                    if act.size(1) > n_candidates:
                        act = act[:,:n_candidates,:]
                else:
                    act, labels = act_dist.sample(n_candidates)
                    act = torch.from_numpy(act).float().to(device)
                    act = act.view(horizon, n_candidates, act_size)
            act = torch.max(torch.min(act, max_act), min_act) # (horizon, candidates, act)
                
            # Find top K low-cost action sequences
            costs = env.cost(act, start_dist, goal_dist, kl_divergence)
            topk_costs, topk_indices = costs.topk(n_elite, dim=-1, largest=False, sorted=True)
            elite = act[:, topk_indices]

            if visualize:
                if act_dist_type == 'gmm':
                    colors = []
                    for label in labels[topk_indices.cpu().numpy()]:
                        # TODO need to fix this
                        colors.append([0.44, 0.7, 0.96, 1])                        
                        # if label == 0:
                        #     colors.append([0.44, 0.7, 0.96, 1])
                        # else:
                        #     colors.append([1, 0.7, 0.13, 1])
                    env.visualize_samples(start_dist.loc, elite, colors=colors)
                else:
                    env.visualize_samples(start_dist.loc, elite)
                
            # Update belief with new means and standard deviations
            if act_dist_type == 'gaussian':
                act_mu = elite.mean(dim=1, keepdim=True)
                act_sigma = elite.std(dim=1, keepdim=True)
                plan_return = act_mu.squeeze()
            elif act_dist_type == 'gmm':
                elite = elite.view(n_elite, horizon * act_size).cpu().numpy()
                act_dist.fit(elite)
                plan_return = act_dist
                
        return plan_return
