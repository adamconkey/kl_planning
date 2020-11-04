#!/usr/bin/env python
import os
import sys
import rospy
import rospkg
rospack = rospkg.RosPack()
import torch
import numpy as np
import argparse

from kl_planning.planning import Planner
from kl_planning.environments import Navigation2DEnvironment
from kl_planning.util import math_util, ui_util, file_util, vis_util

CEM_DISTRIBUTIONS = ['gaussian', 'gmm']
GOAL_DISTRIBUTIONS = ['gaussian', 'gmm', 'dirac_delta', 'uniform']


def assign_arg(key, args, config, choices=None):
    if not getattr(args, key):
        if key not in config:
            ui_util.print_error(f"\nMust specify value for '{key}' either in YAML config "
                                f"or as a command line arg like:\n  --{key} VALUE\n")
            sys.exit(1)
        setattr(args, key, config[key])
    if choices and getattr(args, key) not in choices:
        ui_util.print_error(f"\nInvalid value for {key}: {getattr(args, key)}\nChoices: {choices}\n")
        sys.exit(1)

        
def process_args(args, config):
    assign_arg('real_dynamics_noise', args, config)
    assign_arg('belief_dynamics_noise', args, config)
    assign_arg('belief_observation_noise', args, config)
    assign_arg('cem_distribution', args, config, CEM_DISTRIBUTIONS)
    assign_arg('goal_distribution', args, config, GOAL_DISTRIBUTIONS)
    assign_arg('max_plan_steps', args, config)
    assign_arg('horizon', args, config)
    assign_arg('n_iters', args, config)
    assign_arg('n_candidates', args, config)
    assign_arg('n_elite', args, config)
    assign_arg('n_mpc_runs', args, config)
    if args.cem_distribution == 'gmm':
        assign_arg('n_cem_gmm_components', args, config)
    if 'm_projection' in config:
        args.m_projection = config['m_projection']
    if args.goal_distribution == 'dirac_delta' and 'force_dirac_identity_precision' in config:
        args.force_dirac_identity_precision = config['force_dirac_identity_precision']


if __name__ == '__main__':
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
    parser.add_argument('--n_cem_gmm_components', type=int,
                        help="Number of GMM components to use in GMM-CEM planner")
    parser.add_argument('--save_path', type=str,
                        help="Absolute path to save pickle data to (must provide to log data)")
    parser.add_argument('--m_projection', action='store_true',
                        help="Use M-projection in KL divergence, otherwise I-projection")
    parser.add_argument('--force_dirac_identity_precision', action='store_true',
                        help="Use identity precision for Dirac-delta goal (Euclidean distance)")
    parser.add_argument('--visualize_png_overlay', action='store_true',
                        help="Project PNG images in rviz to visualize paths")
    parser.add_argument('--purge_bad_samples', action='store_true',
                        help="Reject 'bad' samples in CEM (e.g. in collision)")
    parser.add_argument('--cpu', action='store_true', help="Use CPU instead of GPU")
    args = parser.parse_args()

    # Load config that stores scene configuration and defaults for CL args
    scene = rospy.get_param('scene')
    path = rospack.get_path('kl_planning')
    config_path = os.path.join(path, 'config', 'scenes', 'nav_2d', f"{scene}.yaml")
    file_util.check_path_exists(config_path, "Scene configuration file")
    config = file_util.load_yaml(config_path)
    
    # Set args as defaults from config, and override with anything from CL
    process_args(args, config)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    planner = Planner()
    env = Navigation2DEnvironment(config, args.m_projection, args.belief_dynamics_noise, device)
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
    else:
        ui_util.print_error(f"Unknown goal distribution type: {config['goal_distribution']}")
        sys.exit(0)
    
    min_act = torch.tensor([-np.tan(config['agent']['max_phi']).astype(np.float32),
                            config['agent']['min_v'], config['agent']['min_time']]).to(device)
    max_act = torch.tensor([np.tan(config['agent']['max_phi']).astype(np.float32),
                            config['agent']['max_v'], config['agent']['max_time']]).to(device)

    log = {'states': [], 'kl_divergence': []}
    for _ in range(args.n_mpc_runs):
        start_mu = start_state.to(device)
        start_sigma = torch.diag(belief_observation_noise).to(device)
        start_dist = torch.distributions.MultivariateNormal(start_mu, start_sigma)
    
        log_states = []
        log_kl_divergence = []
        true_state = start_state

        env.set_agent_location(true_state.cpu().numpy(), False)

        if config['goal_distribution'] == 'gaussian':
            vis_util.visualize_gmm_goals([goal_dist.loc], [goal_dist.covariance_matrix])
        
        for k in range(args.max_plan_steps):
            env.set_agent_location(true_state.cpu().numpy())
            log_states.append(true_state.cpu().numpy())
            # Log KL divergence
            if args.m_projection:
                kl = kl_divergence(goal_dist, start_dist)
            else:
                kl = kl_divergence(start_dist, goal_dist)
            log_kl_divergence.append(kl.item())
            
            plan_return = planner.plan_cem(env, start_dist, goal_dist, min_act, max_act, device,
                                           args.horizon, args.n_iters, args.n_candidates,
                                           args.n_elite, args.n_cem_gmm_components,
                                           kl_divergence, args.cem_distribution, visualize=True,
                                           purge_bad_samples=args.purge_bad_samples)

            if args.cem_distribution == 'gmm':
                # TODO just taking most likely for now
                best_idx = np.argmax(plan_return.weights_)
                act = plan_return.means_[best_idx]
                act_size = len(min_act)
                act = torch.from_numpy(act).view(args.horizon, act_size).to(device)
                if args.visualize_png_overlay:
                    mu = start_dist.loc.unsqueeze(0).to(device)
                    sigma = start_dist.covariance_matrix.unsqueeze(0).to(device)
                    if config['goal_distribution'] == 'gaussian':
                        goal_mus = [goal_dist.loc]
                        goal_sigmas = [goal_dist.covariance_matrix]
                        # TODO these don't look right yet
                        # vis_util.visualize_gmm_plan(mu, sigma, plan_return, env, args.horizon,
                        #                             act_size, goal_mus, goal_sigmas)
                        vis_util.visualize_gmm_goals(goal_mus, goal_sigmas)
                    elif config['goal_distribution'] == 'gmm':
                        goal_mus = mus.squeeze()
                        goal_sigmas = torch.diag_embed(sigmas.squeeze(), dim1=-2, dim2=-1)
                        vis_util.visualize_gmm_goals(goal_mus, goal_sigmas)
                    else:
                        ui_util.print_error(f"\nUnsupported goal distribution for GMM-CEM: "
                                            f"{config['goal_distribution']}")
                        sys.exit(1)
            else:
                act = plan_return
                if args.visualize_png_overlay:
                    mu = start_dist.loc.unsqueeze(0).to(device)
                    sigma = start_dist.covariance_matrix.unsqueeze(0).to(device)
                    if config['goal_distribution'] == 'gaussian':
                        goal_mus = [goal_dist.loc]
                        goal_sigmas = [goal_dist.covariance_matrix]
                        vis_util.visualize_gaussian_plan(mu, sigma, act, env, goal_mus, goal_sigmas)
                    elif config['goal_distribution'] == 'gmm':
                        goal_mus = mus.squeeze()
                        goal_sigmas = torch.diag_embed(sigmas.squeeze(), dim1=-2, dim2=-1)
                        vis_util.visualize_gaussian_plan(mu, sigma, act, env, goal_mus, goal_sigmas)
                    elif config['goal_distribution'] == 'uniform':
                        vis_util.visualize_gaussian_plan(mu, sigma, act, env, uniform_lows=lows,
                                                         uniform_highs=highs)
                    else:
                        vis_util.visualize_gaussian_plan(mu, sigma, act, env, state_size)
                    rospy.sleep(1)
                        
            # Update current position for next planning step
            true_state = env.dynamics(true_state.unsqueeze(0), act[0].unsqueeze(0),
                                      noise_gain=args.real_dynamics_noise).squeeze()
        
            start_dist.loc = true_state

        log['states'].append(np.stack(log_states))
        log['kl_divergence'].append(log_kl_divergence)

    if args.save_path:
        file_util.save_pickle(log, args.save_path)
