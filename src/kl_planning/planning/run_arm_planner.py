#!/usr/bin/env python
import os
import sys
import rospy
import rospkg
rospack = rospkg.RosPack()
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import argparse

from kl_planning.planning import Planner
from kl_planning.environments import ArmEnvironment
from kl_planning.util import math_util, ui_util, file_util, vis_util


# TODO for now limited support for other distribution types
CEM_DISTRIBUTIONS = ['gaussian']
GOAL_DISTRIBUTIONS = ['gaussian']


def process_args(args, config):
    ui_util.assign_arg('real_dynamics_noise', args, config)
    ui_util.assign_arg('belief_dynamics_noise', args, config)
    ui_util.assign_arg('belief_observation_noise', args, config)
    ui_util.assign_arg('cem_distribution', args, config, CEM_DISTRIBUTIONS)
    ui_util.assign_arg('goal_distribution', args, config, GOAL_DISTRIBUTIONS)
    ui_util.assign_arg('max_plan_steps', args, config)
    ui_util.assign_arg('horizon', args, config)
    ui_util.assign_arg('n_iters', args, config)
    ui_util.assign_arg('n_candidates', args, config)
    ui_util.assign_arg('n_elite', args, config)
    ui_util.assign_arg('n_mpc_runs', args, config)
    if 'm_projection' in config:
        args.m_projection = config['m_projection']


if __name__ == '__main__':
    """
    Main script for running the arm environment plans.
    """
    rospy.init_node('run_arm_planner')
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dynamics_noise', type=float,
                        help="Gaussian noise gain on actual robot dynamics")
    parser.add_argument('--belief_dynamics_noise', type=float,
                        help="Gaussian noise gain on agent's belief of dynamics")
    parser.add_argument('--belief_observation_noise', type=float, nargs='+',
                        help="Gaussian noise gain on agent's state estimate")
    parser.add_argument('--cem_distribution', type=str, choices=CEM_DISTRIBUTIONS + [None],
                        help="Distribution type used in generating CEM action trajectory samples")
    parser.add_argument('--goal_distribution', type=str, choices=GOAL_DISTRIBUTIONS + [None],
                        help="Goal distribution type")
    parser.add_argument('--max_plan_steps', type=int, help="Max number of iterations to run planning")
    parser.add_argument('--horizon', type=int, help="Planning horizon")
    parser.add_argument('--n_iters', type=int, help="Number of iterations to run CEM")
    parser.add_argument('--n_candidates', type=int, help="Number of candidates to generate in CEM")
    parser.add_argument('--n_elite', type=int, help="Number of elite samples to fit in CEM")
    parser.add_argument('--n_mpc_runs', type=int, help="Number of full MPC executions to perform")
    parser.add_argument('--save_path', type=str,
                        help="Absolute path to save pickle data to (must provide to log data)")
    parser.add_argument('--m_projection', action='store_true',
                        help="Use M-projection in KL divergence, otherwise I-projection")
    parser.add_argument('--cpu', action='store_true', help="Use CPU instead of GPU")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # Load config that stores all scene and distribution information
    scene = rospy.get_param("scene")
    path = rospack.get_path('kl_planning')
    config_path = os.path.join(path, 'config', 'scenes', 'arm', f"{scene}.yaml")
    file_util.check_path_exists(config_path, "Scene configuration file")
    config = file_util.load_yaml(config_path)

    # Set args as defaults from config, and override with anything from CL
    process_args(args, config)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    planner = Planner()
    env = ArmEnvironment(config, args.m_projection, args.belief_dynamics_noise, device, args.debug)
    start_state = torch.tensor(env.get_start_state(), dtype=torch.float32, device=device)
    
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
