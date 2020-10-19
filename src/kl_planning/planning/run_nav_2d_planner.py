#!/usr/bin/env python
import os
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
from kl_planning.util import ros_util, math_util
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
    args = parser.parse_args()

    scene = rospy.get_param("scene")
    r = rospkg.RosPack()
    path = r.get_path('kl_planning')
    config_path = os.path.join(path, 'config', 'scenes', f"{scene}.yaml")
    
    
    planner = Planner()
    env = Navigation2DEnvironment(config_path)
    
    # TODO for now this is just hard-coding some stuff to get running, will want
    # to make this all configurable

    start_pos = env.indicator_config['start']['position']
    start_mu = torch.tensor([start_pos[0], start_pos[1], 0.0], dtype=torch.float32)
    start_sigma = torch.diag(torch.tensor([0.001, 0.001, 0.001], dtype=torch.float32))
    start_dist = torch.distributions.MultivariateNormal(start_mu, start_sigma)

    goal_pos = env.indicator_config['goal2']['position']
    goal_mu = torch.tensor([goal_pos[0], goal_pos[1], 0.0], dtype=torch.float32)
    goal_sigma = torch.diag(torch.tensor([0.03, 0.03, 1.0], dtype=torch.float32))
    goal_dist = torch.distributions.MultivariateNormal(goal_mu, goal_sigma)

    # Actions are wheel rotations which then induce delta x, y, theta
    max_phi = 0.7
    min_time = 0.
    max_time = 1.
    
    min_act = torch.tensor([-np.tan(max_phi).astype(np.float32), min_time])
    max_act = torch.tensor([np.tan(max_phi).astype(np.float32), max_time])

    state_size = goal_mu.size(-1)
    
    env.set_agent_location(start_mu)
    
    for k in range(100):
        act = planner.plan_cem(env, start_dist, goal_dist, min_act, max_act, visualize=True)
        
        mus = [start_dist.loc.unsqueeze(0)]
        sigmas = [start_dist.covariance_matrix.unsqueeze(0)]
        
        for t in range(len(act)):
            act_t = act[t].unsqueeze(0).unsqueeze(0).repeat(1, 2 * state_size + 1, 1)
            act_t = act_t.view(act_t.size(0) * act_t.size(1), -1)
            g = lambda x: env.dynamics(x, act_t)
            mu_prime, sigma_prime, _ = math_util.unscented_transform(mus[-1], sigmas[-1], g)
            mus.append(mu_prime)
            sigmas.append(sigma_prime)

        # Update current position for next planning step
        start_dist.loc = env.dynamics(start_dist.loc.unsqueeze(0), act[0].unsqueeze(0),
                                      noise_gain=args.dynamics_noise).squeeze()
        env.set_agent_location(start_dist.loc)
    
        mus.append(goal_mu)
        sigmas.append(goal_sigma)

        # plot(mus, sigmas)

        # img = cv2.imread('/tmp/kl_img.png', cv2.IMREAD_COLOR)
        # img_msg = CvBridge().cv2_to_imgmsg(img, "bgr8")
    
        # display_img = rospy.ServiceProxy("/display_image", DisplayImage)
        # try:
        #     display_img(DisplayImageRequest(img_msg))
        # except rospy.ServiceException as e:
        #     rospy.logerr(f"Service request to display image failed: {e}")

        # plt.close('all') # Free up memory
