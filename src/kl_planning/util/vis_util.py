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
    

if __name__ == '__main__':
    rospy.init_node('test_plot')
    mus = [np.zeros(2), np.ones(2), -np.ones(2)]
    sigmas = [np.eye(2) * 0.03, np.eye(2) * 0.03, np.eye(2) * 0.03]
    
    plot(mus, sigmas)
    display_rviz_img()
