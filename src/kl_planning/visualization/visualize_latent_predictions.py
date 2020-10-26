#!/usr/bin/env python
import os
import sys
import h5py
import argparse
import numpy as np
import cv2
import torch

import rospy
from sensor_msgs.msg import JointState, Image
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray

from kl_planning.util import data_util, file_util, ros_util
from kl_planning.learners import LatentPlanningLearner
from kl_planning.datasets import LatentPlanningDataset


def generate_predictions(data, checkpoint, device):
    learner = LatentPlanningLearner(checkpoint_filename=checkpoint)
    learner.set_models_to_eval()

    # Initial obs/prev action at start of prediction sequence
    init_obs = {m: learner.dataset.process_data_in(data[m][:1], m)
                for m in learner.config.obs_modalities}
    init_act = {m: learner.dataset.process_data_in(data[m][:1], m)
                for m in learner.config.act_modalities}
    # Actions to propagate through latent space
    act = {m: learner.dataset.process_data_in(data[m][1:], m)
           for m in learner.config.act_modalities}

    # Add batch dims to everything and send to device
    init_obs = {k: v.to(device).unsqueeze(1) for k, v in init_obs.items()}
    init_act = {k: v.to(device).unsqueeze(1) for k, v in init_act.items()}
    act = {k: v.to(device).unsqueeze(1) for k, v in act.items()}
    
    decoded = learner.predict_outputs(init_obs, init_act, act)
    return decoded

def visualize_data(h5_filename, cp_filename, start_idx, device, time_subsample=1, rate=30):
    rospy.loginfo("Loading H5 data...")
    with h5py.File(h5_filename, 'r') as h5_file:
        rgb = np.array(h5_file['rgb'][start_idx:None:time_subsample])
        joint_pos = np.array(h5_file['joint_positions'][start_idx:None:time_subsample])
        delta_joint_pos = joint_pos[1:] - joint_pos[:-1]
        delta_joint_pos = np.concatenate([np.zeros((1, joint_pos.shape[-1])), delta_joint_pos], axis=0)
        gripper_pos = np.array(h5_file['gripper_joint_positions'][start_idx:None:time_subsample])
        tf = {}
        for obj_id, obj_data in h5_file['tf'].items():
            tf[obj_id] = {
                'parent_frame': obj_data.attrs['parent_frame'],
                'position': np.array(obj_data['position'][start_idx:None:time_subsample]),
                'orientation': np.array(obj_data['orientation'][start_idx:None:time_subsample])
            }
        objects = {}
        for obj_id in h5_file['objects'].keys():
            obj_data = h5_file['objects'][obj_id].attrs
            objects[obj_id] = {
                # Extent/position converted from cm to m
                'length': obj_data['length'] / 100.,
                'width': obj_data['width'] / 100.,
                'height': obj_data['height'] / 100.,
                'color': np.array(obj_data['color']),
            }
        task_goals = [s.decode('utf-8') for s in h5_file.attrs['task_goals']]

        if cp_filename:
            data = {
                'rgb': rgb,
                'joint_positions': joint_pos,
                'delta_joint_positions': delta_joint_pos,
                'gripper_joint_positions': gripper_pos
            }
            decoded = generate_predictions(data, cp_filename, device)
        else:
            decoded = {}

    rospy.loginfo("H5 data loaded.")

    rospy.loginfo("Creating ROS messages...")
    n_timesteps = len(rgb)

    colors = ["windows blue", "amber", "faded green", "dusty purple", "light red",
              "deep blue", "blue green", "pumpkin", "periwinkle blue",
              "lemon yellow", "mocha", "greeny yellow", "grey/blue", "dark fuchsia",
              "greyish teal", "eggplant purple", "strong blue"]
        
    joint_state_msgs = []
    rgb_msgs = []
    rgb_pred_msgs = []
    tf_msgs = []
    for i in range(1, n_timesteps):
        joint_state_msgs.append(ros_util.get_joint_state_msg(joint_pos[i]))
        rgb_msgs.append(ros_util.rgb_to_msg(rgb[i].squeeze()))
        tf_msgs.append(ros_util.get_tf_msg(tf, i))

    if decoded:
        for i in range(n_timesteps - 1):
            rgb_pred_msgs.append(ros_util.rgb_to_msg(decoded['rgb'][i]))
        
    rospy.loginfo("ROS messages created.")
        
    joint_state_pub = rospy.Publisher('/panda/joint_states', JointState, queue_size=1)
    rgb_pub = rospy.Publisher('/rgb', Image, queue_size=1)
    tf_pub = rospy.Publisher('/tf', TFMessage, queue_size=1)
    
    if decoded:
        rgb_pred_pub = rospy.Publisher('/rgb_prediction', Image, queue_size=1)

    rospy.loginfo("Publishing messages...")
    rate = rospy.Rate(rate)
    while not rospy.is_shutdown():
        for i in range(n_timesteps - 1):
            ros_util.publish_msg(rgb_msgs[i], rgb_pub)
            if decoded:
                ros_util.publish_msg(rgb_pred_msgs[i], rgb_pred_pub)
            ros_util.publish_tf_msg(tf_msgs[i], tf_pub)
            rate.sleep()
        rospy.sleep(3)
        

if __name__ == '__main__':
    rospy.init_node('visualize_latent_predictions')

    h5_filename = rospy.get_param("~h5")
    cp_filename = rospy.get_param("~checkpoint", "")
    use_cpu = rospy.get_param("~cpu", False)
    start_idx = rospy.get_param("~start_idx", 0)
    rate = rospy.get_param("~rate", 30)
    
    file_util.check_path_exists(h5_filename, "H5 file")
    if cp_filename:
        file_util.check_path_exists(cp_filename, "Model checkpoint file")
        learner = LatentPlanningLearner(checkpoint_filename=cp_filename)
        time_subsample = learner.config.time_subsample
    else:
        time_subsample = 1
        
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
    
    visualize_data(h5_filename, cp_filename, start_idx, device, time_subsample, rate)
