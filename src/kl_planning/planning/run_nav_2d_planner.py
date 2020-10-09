#!/usr/bin/env python
import torch
from math import pi

from kl_planning.planning import Planner
from kl_planning.environments import Navigation2DEnvironment

if __name__ == '__main__':
    planner = Planner()
    env = Navigation2DEnvironment()

    
    # TODO for now this is just hard-coding some stuff to get running, will want
    # to make this all configurable
    
    start_mu = torch.tensor([-8, 8], dtype=torch.float32)
    start_sigma = torch.diag(torch.tensor([0.01, 0.01], dtype=torch.float32))

    goal_mu = torch.tensor([8, -8], dtype=torch.float32)
    goal_sigma = torch.diag(torch.tensor([0.3, 0.3], dtype=torch.float32))

    min_act = torch.tensor([0, -pi])
    max_act = torch.tensor([2, pi])
    
    act_seq = planner.plan_cem(start_mu, start_sigma, goal_mu, goal_sigma,
                               planner.cost, env.dynamics, min_act, max_act, visualize=True)

    # mus = [start_mu.unsqueeze(0)]
    # sigmas = [start_sigma.unsqueeze(0)]
    # xy = start_mu.unsqueeze(0)
    # xys = [xy.numpy().squeeze()]
    
    # for t in range(len(act_seq)):
    #     g = lambda x: dynamics(x, act_seq[t].unsqueeze(0))
    #     mu_prime, sigma_prime = unscented_transform(mus[-1], sigmas[-1], g)
    #     mus.append(mu_prime)
    #     sigmas.append(sigma_prime)
    
    #     xy = dynamics(xy, act_seq[t].unsqueeze(0), repeat=False)
    #     xys.append(xy.numpy().squeeze())

    # xs = [p[0] for p in xys]
    # ys = [p[1] for p in xys]
    
    # mus.append(goal_mu)
    # sigmas.append(goal_sigma)

