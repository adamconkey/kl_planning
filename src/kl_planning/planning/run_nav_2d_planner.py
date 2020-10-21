#!/usr/bin/env python
import os
import sys
import rospy
import rospkg
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import argparse
import cv2
from cv_bridge import CvBridge
from scipy.stats import multivariate_normal
from math import pi

from kl_planning.planning import Planner
from kl_planning.environments import Navigation2DEnvironment
from kl_planning.util import ros_util, math_util, ui_util, file_util
from kl_planning.srv import DisplayImage, DisplayImageRequest


def plot(mus, sigmas):
    x = np.linspace(-2, 2, 500)
    y = np.linspace(-2, 2, 500)
    X, Y = np.meshgrid(x, y)
    pos = np.array([X.flatten(), Y.flatten()]).T

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10)
    for mu, sigma in zip(mus, sigmas):
        rv = multivariate_normal(mu.squeeze()[:2], sigma.squeeze()[:2,:2])
        ax.contour(rv.pdf(pos).reshape(500,500))
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('/tmp/kl_img.png', bbox_inches='tight', pad_inches=0) # , transparent=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dynamics_noise', type=float, default=0.02)
    parser.add_argument('--real_observation_noise', type=float, nargs='+', default=[0.001])
    parser.add_argument('--belief_dynamics_noise', type=float, default=0.02)
    parser.add_argument('--belief_observation_noise', type=float, nargs='+', default=[0.001])
    parser.add_argument('--cem_distribution', type=str, default='gaussian',
                        choices=['gaussian', 'gmm'])
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--n_iters', type=int, default=10)
    parser.add_argument('--n_candidates', type=int, default=200)
    parser.add_argument('--n_elite', type=int, default=10)
    parser.add_argument('--n_cem_gmm_components', type=int, default=2)
    parser.add_argument('--m_projection', action='store_true')
    parser.add_argument('--force_dirac_identity_precision', action='store_true')
    args = parser.parse_args()

    # Load config that stores all scene and distribution information
    scene = rospy.get_param("scene")
    r = rospkg.RosPack()
    path = r.get_path('kl_planning')
    config_path = os.path.join(path, 'config', 'scenes', 'nav_2d', f"{scene}.yaml")
    file_util.check_path_exists(config_path, "Scene configuration file")
    config = file_util.load_yaml(config_path)

    m_projection = config['m_projection'] if 'm_projection' in config else args.m_projection
    
    planner = Planner()
    env = Navigation2DEnvironment(config, m_projection, args.belief_dynamics_noise)
    
    kl_divergence = None

    # TODO real observation noise isn't being used yet, but can add in if you use a
    # real observation model to give agent noisy samples of the true state
    # if len(args.real_observation_noise) == 1:
    #     real_observation_noise = torch.ones(env.state_size) * args.real_observation_noise[0]
    # else:
    #     real_observation_noise = torch.tensor(args.real_observation_noise)

    if len(args.belief_observation_noise) == 1:
        belief_observation_noise = torch.ones(env.state_size) * args.belief_observation_noise[0]
    else:
        belief_observation_noise = torch.tensor(args.belief_observation_noise)

    true_state = torch.tensor(env.get_start_state(), dtype=torch.float32)

    # TODO for now just taking true state as mean, can explore noisy sample instead later
    start_mu = true_state
    start_sigma = torch.diag(belief_observation_noise)
    start_dist = torch.distributions.MultivariateNormal(start_mu, start_sigma)
    
    if config['goal_distribution'] == 'gaussian':
        goal_states = env.get_goal_states()
        goal_covs = env.get_goal_covariances()
        if len(goal_states) > 1:
            ui_util.print_error("\nMore than one goal found in env config, "
                                "but Gaussian only takes one goal.\n")
            sys.exit(1)
        goal_mu = torch.tensor(goal_states[0], dtype=torch.float32)
        goal_sigma = torch.diag(torch.tensor(goal_covs[0], dtype=torch.float32))
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
        goal_dist = GaussianMixture(n_components, n_features, mus, sigmas)
        goal_dist.pi.data = torch.tensor(goal_weights).view(1, n_components, 1)
        kl_divergence = math_util.kl_gmm_gmm
    elif config['goal_distribution'] == 'uniform':
        from torch.distributions.uniform import Uniform
        lows, highs = env.get_goal_low_high()
        goal_dist = Uniform(torch.tensor(lows), torch.tensor(highs))
    elif config['goal_distribution'] == 'dirac_delta':
        from kl_planning.distributions import DiracDelta
        goal_state = torch.tensor(env.get_goal_states()[0])
        goal_state = goal_state.unsqueeze(0).repeat(args.n_candidates, 1)
        goal_dist = DiracDelta(goal_state, args.force_dirac_identity_precision)
        kl_divergence = math_util.kl_dirac_mvn
    else:
        ui_util.print_error(f"Unknown goal distribution type: {args.goal_distribution}")
        sys.exit(0)
            
    
    min_act = torch.tensor([-np.tan(config['agent']['max_phi']).astype(np.float32),
                            config['agent']['min_v'], config['agent']['min_time']])
    max_act = torch.tensor([np.tan(config['agent']['max_phi']).astype(np.float32),
                            config['agent']['max_v'], config['agent']['max_time']])

    state_size = start_mu.size(-1)
    
    for k in range(100):
        env.set_agent_location(true_state)
        
        act = planner.plan_cem(env, start_dist, goal_dist, min_act, max_act,
                               args.horizon, args.n_iters, args.n_candidates,
                               args.n_elite, args.n_cem_gmm_components,
                               kl_divergence, args.cem_distribution, visualize=True)
        
        # mus = [start_dist.loc.unsqueeze(0)]
        # sigmas = [start_dist.covariance_matrix.unsqueeze(0)]
        
        # for t in range(len(act)):
        #     act_t = act[t].unsqueeze(0).unsqueeze(0).repeat(1, 2 * state_size + 1, 1)
        #     act_t = act_t.view(act_t.size(0) * act_t.size(1), -1)
        #     g = lambda x: env.dynamics(x, act_t)
        #     mu_prime, sigma_prime, _ = math_util.unscented_transform(mus[-1], sigmas[-1], g)
        #     mus.append(mu_prime)
        #     sigmas.append(sigma_prime)


        # TODO need to update dynamics so that you separately update belief and true dynamics
        # by passing them through both functions with respective noise. I think you want to add
        # yet another noise parameter so you can maintain observation noise to mimic noisy sensor
        # for initial state distribution. Can do no-op action on current state 

        
        # Update current position for next planning step
        true_state = env.dynamics(true_state.unsqueeze(0), act[0].unsqueeze(0),
                                  noise_gain=args.real_dynamics_noise).squeeze()
        
        # TODO for now just taking true state as mean, probably more correct to have a noisy
        # observation function and then let the agent use an observation model in planning.
        start_dist.loc = true_state
        
        # mus.append(goal_mu)
        # sigmas.append(goal_sigma)

        # plot(mus, sigmas)

        # img = cv2.imread('/tmp/kl_img.png', cv2.IMREAD_COLOR)
        # img_msg = CvBridge().cv2_to_imgmsg(img, "bgr8")
    
        # display_img = rospy.ServiceProxy("/display_image", DisplayImage)
        # try:
        #     display_img(DisplayImageRequest(img_msg))
        # except rospy.ServiceException as e:
        #     rospy.logerr(f"Service request to display image failed: {e}")

        # plt.close('all') # Free up memory
