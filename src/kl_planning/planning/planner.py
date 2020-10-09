import torch
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence

import matplotlib.pyplot as plt
from tqdm import tqdm


class Planner:

    def __init__(self):
        pass

    def plan_cem(self, start_mu, start_sigma, goal_mu, goal_sigma, cost_func,
                 dynamics_func, min_act, max_act, horizon=10, n_iters=10,
                 n_candidates=1000, n_elite=100, visualize=False, action_size=2):
    
        start_mu = start_mu.repeat(n_candidates, 1)
        start_sigma = start_sigma.repeat(n_candidates, 1, 1)
        
        act_mu = torch.zeros(horizon, 1, action_size)
        act_sigma = torch.ones(horizon, 1, action_size)
    
        best_costs = []
        worst_costs = []
        
        for _ in tqdm(range(n_iters)):
            # Generate action delta samples
            noise = torch.randn(horizon, n_candidates, action_size)
            act = act_mu + act_sigma * noise
            act = torch.max(torch.min(act, max_act), min_act)
            
            # Find top K low-cost action sequences
            costs = cost_func(act, start_mu, start_sigma, goal_mu, goal_sigma, dynamics_func)
            topk_values, topk_indices = costs.topk(n_elite, dim=-1, largest=False, sorted=True)    
            topk_values = topk_values.squeeze()
            best_costs.append(topk_values[0].item())
            worst_costs.append(topk_values[-1].item())
            elite = act[:, topk_indices]
            
            # Update belief with new means and standard deviations
            act_mu = elite.mean(dim=1, keepdim=True)
            act_sigma = elite.std(dim=1, unbiased=False, keepdim=True)
    
        if visualize:
            plt.title("Elite Costs")
            plt.plot(best_costs, label='Best')
            plt.plot(worst_costs, label='Worst')
            plt.legend()
            plt.show()
    
        return act_mu.squeeze()

    def cost(self, act, start_mu, start_sigma, goal_mu, goal_sigma, dynamics_func):
        mus = [start_mu]
        sigmas = [start_sigma]
        
        for t in range(len(act)):
            g = lambda x: dynamics_func(x, act[t])
            mu_prime, sigma_prime = self.unscented_transform(mus[-1], sigmas[-1], g)
            mus.append(mu_prime)
            sigmas.append(sigma_prime)
            
        p_T = MultivariateNormal(mus[-1], sigmas[-1])
        p_G = MultivariateNormal(goal_mu, goal_sigma)
        cost = kl_divergence(p_G, p_T)
        return cost
    
    def unscented_transform(self, mu, sigma, g, alpha=1, beta=2, kappa=1):
        """
        beta=2 optimal for Gaussians
    
        Not sure about the alpha/kappa defaults
        """
        B = mu.size(0)
        n = mu.size(-1)
        _lambda = alpha**2 * (n + kappa) - n
    
        Q = torch.diag_embed(torch.rand(B, n), dim1=-2, dim2=-1) * 0.2
    
        w_m = torch.full((B, 2*n + 1, 1), 1. / (2 * (n + _lambda)))
        w_c = torch.full((B, 2*n + 1, 1, 1), 1. / (2 * (n + _lambda)))
        w_m[:,0] = _lambda / (n + _lambda)
        w_c[:,0] = _lambda / (n + _lambda) + (1 - alpha**2 + beta)
        
        # No built-in matrix sqrt in torch, and no batch version in scipy, having to homebrew
        M = torch.zeros_like(sigma)
        for i in range(len(M)):
            # M[i] = torch.from_numpy(sqrtm((n + _lambda) * sigma[i].numpy()))
            # M[i] = torch.from_numpy(cholesky((n + _lambda) * sigma[i].numpy()))
            M[i] = torch.cholesky((n + _lambda) * sigma[i])
            
            
        X = mu.unsqueeze(1).repeat(1, 2*n + 1, 1)  # (B, 2n+1, n)
        for i in range(n):
            X[:,i+1,:] += M[:,i,:]
            X[:,n+i+1,:] -= M[:,i,:]
            
        Y = g(X) # (B, 2n+1, state_size)
    
        mu_prime = torch.sum(w_m * Y, dim=1)
        sigma_prime = torch.zeros(B, n, n)
        for i in range(2*n+1):
            y = Y[:,i,:] - mu_prime
            sigma_prime += w_c[:,i] * y.unsqueeze(2) * y.unsqueeze(1) # outer product
        sigma_prime += Q
            
        return mu_prime, sigma_prime
