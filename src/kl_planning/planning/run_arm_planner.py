#!/usr/bin/env python
import os
import sys
import rospy
import rospkg
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import argparse

from kl_planning.planning import Planner
from kl_planning.environments import ArmEnvironment
from kl_planning.util import math_util, ui_util, file_util, vis_util


if __name__ == '__main__':
    rospy.init_node('run_arm_planner')
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dynamics_noise', type=float, default=0.02)
    parser.add_argument('--belief_dynamics_noise', type=float, default=0.02)
    parser.add_argument('--belief_observation_noise', type=float, nargs='+', default=[0.001])
    parser.add_argument('--cem_distribution', type=str, default='gaussian', choices=['gaussian'])
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--n_iters', type=int, default=3)
    parser.add_argument('--n_candidates', type=int, default=40)
    parser.add_argument('--n_elite', type=int, default=5)
    parser.add_argument('--m_projection', action='store_true')
    parser.add_argument('--force_dirac_identity_precision', action='store_true')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--n_mpc_runs', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # Load config that stores all scene and distribution information
    scene = rospy.get_param("scene")
    r = rospkg.RosPack()
    path = r.get_path('kl_planning')
    config_path = os.path.join(path, 'config', 'scenes', 'arm', f"{scene}.yaml")
    file_util.check_path_exists(config_path, "Scene configuration file")
    config = file_util.load_yaml(config_path)

    m_projection = config['m_projection'] if 'm_projection' in config else args.m_projection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    planner = Planner()
    env = ArmEnvironment(config, m_projection, args.belief_dynamics_noise, device, args.debug)
    start_state = torch.tensor(env.get_start_state(), dtype=torch.float32, device=device)
    state_size = start_state.size(-1)
    
    if len(args.belief_observation_noise) == 1:
        belief_observation_noise = torch.ones(env.state_size) * args.belief_observation_noise[0]
    else:
        belief_observation_noise = torch.tensor(args.belief_observation_noise)

    kl_divergence = torch.distributions.kl.kl_divergence
    if config['goal_distribution'] == 'gaussian':
        goal_states = env.get_goal_states()
        goal_covs = env.get_goal_covariances()
        if len(goal_states) > 1:
            ui_util.print_error("\nMore than one goal found in env config, "
                                "but Gaussian only takes one goal.\n")
            sys.exit(1)
        goal_mu = torch.tensor(goal_states[0], dtype=torch.float32, device=device)
        goal_sigma = torch.diag(torch.tensor(goal_covs[0], dtype=torch.float32, device=device))
        goal_dist = torch.distributions.MultivariateNormal(goal_mu, goal_sigma)
    elif config['goal_distribution'] == 'gmm':
        from kl_planning.distributions import GaussianMixture
        goal_states = env.get_goal_states()
        goal_covs = env.get_goal_covariances()
        goal_weights = env.get_goal_weights()
        mus = torch.tensor(goal_states).unsqueeze(0)
        sigmas = torch.tensor(goal_covs).unsqueeze(0)
        n_components = mus.size(1)
        n_features = mus.size(2)
        goal_dist = GaussianMixture(n_components, n_features, mus, sigmas).to(device)
        goal_dist.pi.data = torch.tensor(goal_weights).view(1, n_components, 1).to(device)
        kl_divergence = math_util.kl_gmm_gmm
    elif config['goal_distribution'] == 'uniform':
        from torch.distributions.uniform import Uniform
        lows, highs = env.get_goal_low_high()
        goal_dist = Uniform(torch.tensor(lows).to(device), torch.tensor(highs).to(device))
    elif config['goal_distribution'] == 'dirac_delta':
        from kl_planning.distributions import DiracDelta
        goal_state = torch.tensor(env.get_goal_states()[0])
        goal_state = goal_state.unsqueeze(0).repeat(args.n_candidates, 1).to(device)
        goal_dist = DiracDelta(goal_state, args.force_dirac_identity_precision)
        kl_divergence = math_util.kl_dirac_mvn
        # Creating one also for checking one state for logging
        log_goal_dist = DiracDelta(goal_dist.state[0].unsqueeze(0),
                                   args.force_dirac_identity_precision)
    else:
        ui_util.print_error(f"Unknown goal distribution type: {config['goal_distribution']}")
        sys.exit(0)
    
    min_act = torch.full((7,), -config['agent']['max_delta'], device=device)
    max_act = torch.full((7,), config['agent']['max_delta'], device=device)

    log = {'states': [], 'kl_divergence': []}
    for _ in range(args.n_mpc_runs):
        start_mu = start_state.to(device)
        start_sigma = torch.diag(belief_observation_noise).to(device)
        start_dist = torch.distributions.MultivariateNormal(start_mu, start_sigma)
    
        log_states = []
        log_kl_divergence = []
        true_state = start_state
        
        for k in range(200):
            if rospy.is_shutdown():
                sys.exit(0)
            env.set_agent_location(true_state.cpu().numpy())
            log_states.append(true_state.cpu().numpy())
            # Log KL divergence
            if args.save_path:
                s_t = MultivariateNormal(start_dist.loc.unsqueeze(0),
                                         start_dist.covariance_matrix.unsqueeze(0))
                if m_projection:
                    if config['goal_distribution'] == 'dirac_delta':
                        kl = kl_divergence(log_goal_dist, s_t)
                    else:
                        kl = kl_divergence(goal_dist, s_t)
                else:
                    kl = kl_divergence(s_t, goal_dist)
                log_kl_divergence.append(kl.item())
            
            act = planner.plan_cem(env, start_dist, goal_dist, min_act, max_act, device,
                                   args.horizon, args.n_iters, args.n_candidates,
                                   args.n_elite, kl_divergence=kl_divergence)

            # Update current position for next planning step
            true_state = env.dynamics(true_state.unsqueeze(0), act[0].unsqueeze(0),
                                      noise_gain=args.real_dynamics_noise).squeeze()        
            start_dist.loc = true_state

        log['states'].append(np.stack(log_states))
        log['kl_divergence'].append(log_kl_divergence)

    if args.save_path:
        file_util.save_pickle(log, args.save_path)
