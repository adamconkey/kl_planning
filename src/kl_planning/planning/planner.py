import sys
import torch

import matplotlib.pyplot as plt
from tqdm import tqdm

from kl_planning.util import vis_util


class Planner:

    def __init__(self):
        pass

    def plan_cem(self, start_mu, start_sigma, goal_mu, goal_sigma, cost_func,
                 min_act, max_act, horizon=10, n_iters=20, n_candidates=1000,
                 n_elite=100, visualize=False, action_size=2):
    
        start_mu = start_mu.repeat(n_candidates, 1)
        start_sigma = start_sigma.repeat(n_candidates, 1, 1)
        
        act_mu = torch.zeros(horizon, 1, action_size)
        act_sigma = torch.ones(horizon, 1, action_size) * 3
    
        best_costs = []
        worst_costs = []
        
        for _ in tqdm(range(n_iters)):
            # Generate action delta samples
            noise = torch.randn(horizon, n_candidates, action_size)
            act = act_mu + act_sigma * noise
            act = torch.max(torch.min(act, max_act), min_act)
            
            # Find top K low-cost action sequences
            costs = cost_func(act, start_mu, start_sigma, goal_mu, goal_sigma)
            topk_costs, topk_indices = costs.topk(n_elite, dim=-1, largest=False, sorted=True)    
            topk_costs = topk_costs.squeeze()
            best_costs.append(topk_costs[0].item())
            worst_costs.append(topk_costs[-1].item())
            elite = act[:, topk_indices]

            if visualize:
                vis_util.visualize_trajectory_samples(elite, topk_costs)
            
            # Update belief with new means and standard deviations
            act_mu = elite.mean(dim=1, keepdim=True)
            act_sigma = elite.std(dim=1, unbiased=False, keepdim=True)
    
        # if visualize:
        #     plt.title("Elite Costs")
        #     plt.plot(best_costs, label='Best')
        #     plt.plot(worst_costs, label='Worst')
        #     plt.legend()
        #     plt.show()
    
        return act_mu.squeeze()
