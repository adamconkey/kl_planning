import torch
import rospy
import random
import seaborn as sns

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


def get_color_sequence(n_colors, palette='hls', shuffle=False):
    colors = list(sns.color_palette(palette, n_colors))
    if shuffle:
        random.shuffle(colors)
    return colors
