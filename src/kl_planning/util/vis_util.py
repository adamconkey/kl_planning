import torch
import rospy
import random
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import cv2
from cv_bridge import CvBridge
from scipy.stats import multivariate_normal

from kl_planning.srv import DisplayImage, DisplayImageRequest
from kl_planning.srv import VisualizeTrajectorySamples, VisualizeTrajectorySamplesRequest
from kl_planning.util import math_util


def visualize_trajectory_samples(samples, costs=None, size=0.03):
    """
    Serializes trajectory samples and makes service request to visualize them.
    Samples are visualized as lines in rviz colored based on cost.

    Args:
        samples (Tensor or array): Trajectory samples of shape (time, samples, state)
        costs (Tensor or array): Vector of costs associated with each sample
    """
    if torch.is_tensor(samples):
        samples = samples.numpy()
    if costs is not None and torch.is_tensor(costs):
        costs = costs.numpy()

    vis_request = VisualizeTrajectorySamplesRequest()
    vis_request.samples = samples.flatten().tolist()
    vis_request.shape = list(samples.shape)
    if costs is not None:
        vis_request.costs = costs.tolist()
    vis_request.size = size

    visualize = rospy.ServiceProxy("/visualization/visualize_trajectory_samples",
                                   VisualizeTrajectorySamples)
    try:
        visualize(vis_request)
    except rospy.ServiceException as e:
        rospy.logerr(f"Service request to visualize trajectory samples failed: {e}")


def display_rviz_img(load_path):
    img = cv2.imread(load_path, cv2.IMREAD_COLOR)
    img_msg = CvBridge().cv2_to_imgmsg(img, "bgr8")
    
    display_img = rospy.ServiceProxy("/visualization/display_image", DisplayImage)
    try:
        display_img(DisplayImageRequest(img_msg))
    except rospy.ServiceException as e:
        rospy.logerr(f"Service request to display image failed: {e}")
        

def get_color_sequence(n_colors, palette='hls', shuffle=False):
    colors = list(sns.color_palette(palette, n_colors))
    if shuffle:
        random.shuffle(colors)
    return colors


def plot_2d_gaussians(mus, sigmas, save_path='/tmp/img.png'):
    x = np.linspace(-2, 2, 500)
    y = np.linspace(-2, 2, 500)
    X, Y = np.meshgrid(x, y)
    pos = np.array([X.flatten(), Y.flatten()]).T

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10)
    Z = np.zeros((500, 500))
    for mu, sigma in zip(mus, sigmas):
        mvn = multivariate_normal(mu.squeeze()[:2], sigma.squeeze()[:2,:2])
        Z += mvn.pdf(pos).reshape(500,500)
    ax.contourf(X, Y, Z, 20, cmap='magma_r')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close('all') # Free up memory
    

def visualize_gmm_plan(start_mu, start_sigma, plan_dist, env, state_size, horizon,
                       act_size, temp_img_filename='/tmp/kl_img.png'):
    mus = [[start_mu] for _ in range(plan_dist.n_components)]
    sigmas = [[start_sigma] for _ in range(plan_dist.n_components)]

    for i, act in enumerate(plan_dist.means_):
        act = torch.from_numpy(act).view(horizon, act_size)
        for t in range(len(act)):
            act_t = act[t].unsqueeze(0).unsqueeze(0).repeat(1, 2 * state_size + 1, 1)
            act_t = act_t.view(act_t.size(0) * act_t.size(1), -1)
            g = lambda x: env.dynamics(x, act_t)
            mu_prime, sigma_prime, _ = math_util.unscented_transform(mus[i][-1], sigmas[i][-1], g)
            mus[i].append(mu_prime)
            sigmas[i].append(sigma_prime)

    all_mus = [m for mus_i in mus for m in mus_i]
    all_sigmas = [s for sigmas_i in sigmas for s in sigmas_i]
    plot_2d_gaussians(all_mus, all_sigmas, temp_img_filename)
    display_rviz_img(temp_img_filename)
        
        
def visualize_gaussian_plan(start_mu, start_sigma, act, env, state_size,
                            temp_img_filename='/tmp/kl_img.png'):
    mus = [start_mu]
    sigmas = [start_sigma]
    
    for t in range(len(act)):
        act_t = act[t].unsqueeze(0).unsqueeze(0).repeat(1, 2 * state_size + 1, 1)
        act_t = act_t.view(act_t.size(0) * act_t.size(1), -1)
        g = lambda x: env.dynamics(x, act_t)
        mu_prime, sigma_prime, _ = math_util.unscented_transform(mus[-1], sigmas[-1], g)
        mus.append(mu_prime)
        sigmas.append(sigma_prime)

    plot_2d_gaussians(mus, sigmas, temp_img_filename)
    display_rviz_img(temp_img_filename)

    
if __name__ == '__main__':
    rospy.init_node('test_plot')
    mus = [np.zeros(2), np.ones(2), -np.ones(2)]
    sigmas = [np.eye(2) * 0.03, np.eye(2) * 0.03, np.eye(2) * 0.03]
    
    plot(mus, sigmas)
    display_rviz_img()
