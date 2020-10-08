#!/usr/bin/env python
import sys
import torch
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm, cholesky
from tqdm import tqdm


def dynamics(start_xy, act, repeat=True):
    """
    start_xy (b, 2) x, y
    act (b, 2) dist, angle
    """
    delta_x = act[:,0] * torch.cos(act[:,1])
    delta_y = act[:,0] * torch.sin(act[:,1])
    delta_xy = torch.stack([delta_x, delta_y], dim=-1).unsqueeze(1)
    if repeat:
        delta_xy = delta_xy.repeat(1, 5, 1) # 2n+1 = 5, just hard-coding for now
    next_xy = start_xy + delta_xy
    return next_xy


def unscented_transform(mu, sigma, g, alpha=1, beta=2, kappa=1):
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
        M[i] = torch.from_numpy(cholesky((n + _lambda) * sigma[i].numpy()))
        
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
    

def cost(act, start_mu, start_sigma, goal_mu, goal_sigma):
    mus = [start_mu]
    sigmas = [start_sigma]
    
    for t in range(len(act)):
        g = lambda x: dynamics(x, act[t])
        mu_prime, sigma_prime = unscented_transform(mus[-1], sigmas[-1], g)
        mus.append(mu_prime)
        sigmas.append(sigma_prime)
        
    p_T = MultivariateNormal(mus[-1], sigmas[-1])
    p_G = MultivariateNormal(goal_mu, goal_sigma)
    cost = kl_divergence(p_G, p_T)
    return cost


def plan_cem(start_mu, start_sigma, goal_mu, goal_sigma, cost_func,
             horizon=10, n_iters=10, n_candidates=1000, n_elite=100,
             visualize=False, action_size=2):

    start_mu = start_mu.repeat(n_candidates, 1)
    start_sigma = start_sigma.repeat(n_candidates, 1, 1)
    
    act_mu = torch.zeros(horizon, 1, action_size)
    act_sigma = torch.ones(horizon, 1, action_size)

    best_costs = []
    worst_costs = []

    min_act = torch.tensor([0, -np.pi])
    max_act = torch.tensor([2, np.pi])

    for _ in tqdm(range(n_iters)):
        # Generate action delta samples
        noise = torch.randn(horizon, n_candidates, action_size)
        act = act_mu + act_sigma * noise
        act = torch.max(torch.min(act, max_act), min_act)
        
        # Find top K low-cost action sequences
        costs = cost_func(act, start_mu, start_sigma, goal_mu, goal_sigma)
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


def plot(mus, sigmas):
    x = np.linspace(-10, 10, 500)
    y = np.linspace(-10, 10, 500)
    X, Y = np.meshgrid(x, y)
    pos = np.array([X.flatten(), Y.flatten()]).T

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(20, 10)
    for mu, sigma in zip(mus, sigmas):
        rv = multivariate_normal(mu.squeeze(), sigma.squeeze())
        axes[0].contour(rv.pdf(pos).reshape(500,500))
    return fig, axes
    

if __name__ == '__main__':
    start_mu = torch.tensor([-8, 8], dtype=torch.float32)
    start_sigma = torch.diag(torch.tensor([0.01, 0.01], dtype=torch.float32))

    goal_mu = torch.tensor([8, -8], dtype=torch.float32)
    goal_sigma = torch.diag(torch.tensor([0.3, 0.3], dtype=torch.float32))
    
    act_seq = plan_cem(start_mu, start_sigma, goal_mu, goal_sigma, cost,
                       visualize=True)

    mus = [start_mu.unsqueeze(0)]
    sigmas = [start_sigma.unsqueeze(0)]
    xy = start_mu.unsqueeze(0)
    xys = [xy.numpy().squeeze()]
    
    for t in range(len(act_seq)):
        g = lambda x: dynamics(x, act_seq[t].unsqueeze(0))
        mu_prime, sigma_prime = unscented_transform(mus[-1], sigmas[-1], g)
        mus.append(mu_prime)
        sigmas.append(sigma_prime)
    
        xy = dynamics(xy, act_seq[t].unsqueeze(0), repeat=False)
        xys.append(xy.numpy().squeeze())

    xs = [p[0] for p in xys]
    ys = [p[1] for p in xys]
    
    mus.append(goal_mu)
    sigmas.append(goal_sigma)

    fig, axes = plot(mus, sigmas)
    axes[1].scatter(xs, ys, color=cm.rainbow(np.linspace(0, 1, len(ys))))
    axes[1].set_xlim(-10, 10)
    axes[1].set_ylim(-10, 10)
    plt.tight_layout()
    plt.show()
    
