#!/usr/bin/env python
import os
import sys
import rospy
import rospkg
import argparse
import numpy as np
import torch

from kl_planning.util import file_util, vis_util
from kl_planning.environments import Navigation2DEnvironment


if __name__ == '__main__':
    rospy.init_node('visualize_gmm_runs')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True)
    args = parser.parse_args()

    file_util.check_path_exists(args.data_file, "Data file")
    data = file_util.load_pickle(args.data_file)
    data = [t[:,:2] for t in data] # Ignore orientation
    
    mode_idxs = [int(t[-1,-1] < 0) for t in data] # Distinguish left/right goals
    # TODO just handling 2 for now
    colors = []
    for idx in mode_idxs:
        if idx == 0:
            colors.append([0.44, 0.7, 0.96, 1])
        else:
            colors.append([1, 0.7, 0.13, 1])
    
    data = np.array(data).transpose(1, 0, 2) # Swap time/sample dims
    vis_util.visualize_line_trajectory_samples(data, colors=colors, size=0.03)


    # Load config that stores all scene and distribution information
    scene = rospy.get_param("scene")
    r = rospkg.RosPack()
    path = r.get_path('kl_planning')
    config_path = os.path.join(path, 'config', 'scenes', 'nav_2d', f"{scene}.yaml")
    file_util.check_path_exists(config_path, "Scene configuration file")
    config = file_util.load_yaml(config_path)
    env = Navigation2DEnvironment(config)
    
    goal_states = env.get_goal_states()
    goal_covs = env.get_goal_covariances()
    goal_weights = env.get_goal_weights()
    goal_mus = torch.tensor(goal_states)
    goal_sigmas = torch.diag_embed(torch.tensor(goal_covs), dim1=-2, dim2=-1)

    device = torch.device('cuda')

    vis_util.visualize_gmm_goals(goal_mus, goal_sigmas)


    
    print("\nCounts (L/R):", np.unique(mode_idxs, return_counts=True)[-1])
