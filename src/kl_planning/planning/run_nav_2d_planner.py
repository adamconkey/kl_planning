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
from kl_planning.util import ros_util, math_util, ui_util
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
    parser.add_argument('--dynamics_noise', type=float, default=0.02)
    parser.add_argument('--goal_distribution', type=str, default='gaussian',
                        choices=['gaussian', 'gmm', 'uniform'])
    parser.add_argument('--cem_distribution', type=str, default='gaussian',
                        choices=['gaussian', 'gmm'])
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--n_iters', type=int, default=10)
    parser.add_argument('--n_candidates', type=int, default=100)
    parser.add_argument('--n_elite', type=int, default=10)
    parser.add_argument('--n_cem_gmm_components', type=int, default=2)
    args = parser.parse_args()

    scene = rospy.get_param("scene")
    r = rospkg.RosPack()
    path = r.get_path('kl_planning')
    config_path = os.path.join(path, 'config', 'scenes', f"{scene}.yaml")
    
    
    planner = Planner()
    env = Navigation2DEnvironment(config_path)
    
    kl_divergence = None
    
    start_mu = torch.tensor(env.get_start_state(), dtype=torch.float32)
    start_sigma = torch.diag(torch.tensor(env.get_start_covariance(), dtype=torch.float32))
    start_dist = torch.distributions.MultivariateNormal(start_mu, start_sigma)
    
    if args.goal_distribution == 'gaussian':
        goal_states = env.get_goal_states()
        goal_covs = env.get_goal_covariances()
        if len(goal_states) > 1:
            ui_util.print_error("\nMore than one goal found in env config, "
                                "but Gaussian only takes one goal.\n")
            sys.exit(1)
        goal_mu = torch.tensor(goal_states[0], dtype=torch.float32)
        goal_sigma = torch.diag(torch.tensor(goal_covs[0], dtype=torch.float32))
        goal_dist = torch.distributions.MultivariateNormal(goal_mu, goal_sigma)
    elif args.goal_distribution == 'gmm':
        from kl_planning.models import GaussianMixture
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
    elif args.goal_distribution == 'uniform':
        from torch.distributions.uniform import Uniform
        lows, highs = env.get_goal_low_high()
        goal_dist = Uniform(torch.tensor(lows), torch.tensor(highs))
    else:
        ui_util.print_error(f"Unknown goal distribution type: {args.goal_distribution}")
        sys.exit(0)
            

    max_phi = 0.7  # Angular turn rate
    min_v = 0.0    # Min linear velocity
    max_v = 0.5    # Max linear velocity
    min_time = 0.
    max_time = 1.
    
    min_act = torch.tensor([-np.tan(max_phi).astype(np.float32), min_v, min_time])
    max_act = torch.tensor([np.tan(max_phi).astype(np.float32), max_v, max_time])

    state_size = start_mu.size(-1)
    
    env.set_agent_location(start_mu)
    
    for k in range(100):
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

        # Update current position for next planning step
        start_dist.loc = env.dynamics(start_dist.loc.unsqueeze(0), act[0].unsqueeze(0),
                                      noise_gain=args.dynamics_noise).squeeze()
        env.set_agent_location(start_dist.loc)
    
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
