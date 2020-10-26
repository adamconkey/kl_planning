import sys
import rospy
import torch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

from kl_planning.util import math_util


class LatentEnvironment:

    def __init__(self, learner, m_projection=True):
        self.learner = learner
        self.m_projection = m_projection


    def dynamics(self, start_state, act, belief):
        # TODO hacking for single action modality
        act = {'delta_joint_positions': act.unsqueeze(0)}
        rssm_out = self.learner.compute_transition(act, init_belief=belief, init_state=start_state)
        next_state = rssm_out.prior_states[-1]
        return next_state


    def cost(self, act, start_dist, goal_dist, kl_divergence=None,
             device=torch.device('cuda'), belief=None):
        if belief is None:
            raise ValueError("Must pass in belief")
        if kl_divergence is None:
            kl_diverence = torch.distributions.kl.kl_divergence

        n_candidates = act.size(1)
        n_state = start_dist.loc.size(-1)
        n_sigma = 2 * n_state + 1

        mus = [start_dist.loc.repeat(n_candidates, 1)]
        sigmas = [torch.diag_embed(start_dist.scale, dim1=-2, dim2=-1).repeat(n_candidates, 1, 1)]
        belief = belief.repeat(n_candidates * n_sigma, 1)
        sigma_points = []

        for t in range(len(act)):
            act_t = act[t].unsqueeze(1).repeat(1, n_sigma, 1)
            act_t = act_t.view(act_t.size(0) * act_t.size(1), -1)
            g = lambda x: self.dynamics(x, act_t, belief)
            mu = mus[-1]
            sigma = sigmas[-1]
            mu_prime, sigma_prime, Y = math_util.unscented_transform(mu, sigma, g, device=device)
            mus.append(mu_prime)
            sigmas.append(sigma_prime)
            sigma_points.append(Y)
            # Need to compute dynamics again in order to get updated belief
            act_t = {'delta_joint_positions': act_t.unsqueeze(0)}
            rssm_out = self.learner.compute_transition(act_t, init_belief=belief,
                                                       init_state=mu.repeat(n_sigma, 1))
            belief = rssm_out.beliefs[-1]
            
        cost = 0

        # Compute KL cost from final distribution to goal distribution
        kl_cost = 0
        T = len(mus)
        for t in range(T):
            # Increasing contribution of KL cost as time increases
            lambda_ = (t + 1) / float(T)
            # p_t = Normal(mus[t], torch.diagonal(sigmas[t], dim1=-2, dim2=-1))
            p_t = MultivariateNormal(mus[2], sigmas[t])
            
            if self.m_projection:
                kl_cost_t = lambda_ * kl_divergence(goal_dist, p_t)
                if isinstance(goal_dist, torch.distributions.uniform.Uniform):
                    # TODO scaling because uniform is huge, maybe parameterize this
                    kl_cost_t = kl_cost_t.sum(dim=-1) #  / 10.0
                kl_cost += kl_cost_t
            else:
                kl_cost += lambda_ * kl_divergence(p_t, goal_dist)
        cost += kl_cost

        return cost
