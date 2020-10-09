#!/usr/bin/env python
import rospy
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2
from cv_bridge import CvBridge
from scipy.stats import multivariate_normal
from math import pi

from kl_planning.planning import Planner
from kl_planning.environments import Navigation2DEnvironment
from kl_planning.util import ros_util
from kl_planning.srv import DisplayImage, DisplayImageRequest


def plot(mus, sigmas):
    x = np.linspace(-2, 2, 500)
    y = np.linspace(-2, 2, 500)
    X, Y = np.meshgrid(x, y)
    pos = np.array([X.flatten(), Y.flatten()]).T

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10)
    for mu, sigma in zip(mus, sigmas):
        rv = multivariate_normal(mu.squeeze(), sigma.squeeze())
        ax.contour(rv.pdf(pos).reshape(500,500))
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('/tmp/kl_img.png', bbox_inches='tight', pad_inches=0) # , transparent=True)


if __name__ == '__main__':
    planner = Planner()
    env = Navigation2DEnvironment()

    
    # TODO for now this is just hard-coding some stuff to get running, will want
    # to make this all configurable
    
    start_mu = torch.tensor([-1.5, 1.5], dtype=torch.float32)
    start_sigma = torch.diag(torch.tensor([0.001, 0.001], dtype=torch.float32))

    goal_mu = torch.tensor([1.5, -1.5], dtype=torch.float32)
    goal_sigma = torch.diag(torch.tensor([0.03, 0.03], dtype=torch.float32))

    min_act = torch.tensor([0, -pi])
    max_act = torch.tensor([0.25, pi])
    
    act_seq = planner.plan_cem(start_mu, start_sigma, goal_mu, goal_sigma,
                               planner.cost, env.dynamics, min_act, max_act, visualize=False)

    mus = [start_mu.unsqueeze(0)]
    sigmas = [start_sigma.unsqueeze(0)]
    xy = start_mu.unsqueeze(0)
    xys = [xy.numpy().squeeze()]
    
    for t in range(len(act_seq)):
        g = lambda x: env.dynamics(x, act_seq[t].unsqueeze(0))
        mu_prime, sigma_prime = planner.unscented_transform(mus[-1], sigmas[-1], g)
        mus.append(mu_prime)
        sigmas.append(sigma_prime)
    
        xy = env.dynamics(xy, act_seq[t].unsqueeze(0), repeat=False)
        xys.append(xy.numpy().squeeze())

    xs = [p[0] for p in xys]
    ys = [p[1] for p in xys]
    
    mus.append(goal_mu)
    sigmas.append(goal_sigma)

    plot(mus, sigmas)

    img = cv2.imread('/tmp/kl_img.png', cv2.IMREAD_COLOR)
    img_msg = CvBridge().cv2_to_imgmsg(img, "bgr8")
    
    display_img = rospy.ServiceProxy("/display_image", DisplayImage)
    try:
        display_img(DisplayImageRequest(img_msg))
    except rospy.ServiceException as e:
        rospy.logerr(f"Service request to display image failed: {e}")
                