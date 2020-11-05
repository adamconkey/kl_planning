#!/usr/bin/env python
import os
import sys
import pickle
import argparse
import rospy
import rospkg
import numpy as np
from scipy import interpolate
from matplotlib import colors
from moveit_msgs.msg import DisplayRobotState, DisplayTrajectory, RobotTrajectory, ObjectColor
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import ColorRGBA

from kl_planning.util import file_util, vis_util


PANDA_LINKS = [f"panda_link{i}" for i in range(8)] + \
              ['panda_hand', 'panda_leftfinger', 'panda_rightfinger']
PANDA_JOINTS = [f"panda_joint{i}" for i in range(1,8)] + [f"panda_finger_joint{i}" for i in [1,2]]
DEFAULT_GRIPPER_JOINTS = [0.04, 0.04]


def get_display_robot_msg(joints, color=None):
    msg = DisplayRobotState()
    msg.state.joint_state.name = PANDA_JOINTS
    msg.state.joint_state.position = joints + DEFAULT_GRIPPER_JOINTS
    if color:
        msg.highlight_links = [ObjectColor(id=l, color=get_color(color)) for l in PANDA_LINKS]
    return msg


def get_trajectory_msg(trajectory):
    robot_traj = RobotTrajectory()
    robot_traj.joint_trajectory.joint_names = PANDA_JOINTS
    for i in range(len(trajectory)):
        point = JointTrajectoryPoint(positions=trajectory[i].tolist() + DEFAULT_GRIPPER_JOINTS)
        robot_traj.joint_trajectory.points.append(point)
    msg = DisplayTrajectory(trajectory=[robot_traj])
    return msg
    
    
def get_color(color_id):
    """
    color can be any name from this page:
    http://matplotlib.org/mpl_examples/color/named_colors.hires.png
    """
    if isinstance(color_id, str):
        converter = colors.ColorConverter()
        c = converter.to_rgba(colors.cnames[color_id])
    elif len(color_id) == 3:
        c = list(color_id) + [1]
    else:
        c = color_id
    return ColorRGBA(*c)


if __name__ == '__main__':
    rospy.init_node('visualize_arm_execution')
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle', type=str, required=True)
    parser.add_argument('--scene', type=str, required=True)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--subsample', type=int, default=1)
    parser.add_argument('--interpolate', type=int, default=0)
    args = parser.parse_args()

    r = rospkg.RosPack()
    path = r.get_path('kl_planning')
    config_path = os.path.join(path, 'config', 'scenes', 'arm', f"{args.scene}.yaml")
    file_util.check_path_exists(config_path, "Scene configuration file")
    config = file_util.load_yaml(config_path)

    rate = rospy.Rate(1)
    goal_pub = rospy.Publisher("/visualization/goal_joints", DisplayRobotState, queue_size=1)
    start_pub = rospy.Publisher("/visualization/start_joints", DisplayRobotState, queue_size=1)
    trajectory_pub = rospy.Publisher("/visualization/trajectory", DisplayTrajectory, queue_size=1)
    
    goal_joints = config['goals']['goal']['state']
    start_joints = config['start']['state']
    with open(args.pickle, 'rb') as f:
        data = pickle.load(f)
    trajectory = data['states'][0][:args.end_idx:args.subsample] # (t, 7)
    n_steps = len(trajectory)
    traj_pubs = [rospy.Publisher(f"/visualization/path_joints_{i}", DisplayRobotState, queue_size=1)
                 for i in range(n_steps)]
    
    rate.sleep()
    goal_pub.publish(get_display_robot_msg(goal_joints, 'indianred'))
    start_pub.publish(get_display_robot_msg(start_joints))
    
    if args.interpolate > 0:
        dims = []
        n_time, n_state = trajectory.shape
        nominal_ts = np.linspace(0, 1, n_time)
        interp_ts = np.linspace(0, 1, n_time * args.interpolate)
        for i in range(n_state):
            f = interpolate.interp1d(nominal_ts, trajectory[:,i])
            dims.append(f(interp_ts))
        trajectory = np.stack(dims, axis=-1)            
            
    trajectory_pub.publish(get_trajectory_msg(trajectory))

    
    # Display trajectory as color gradient
    colors = vis_util.get_color_sequence(n_steps, palette="YlGnBu")
    for i in range(n_steps):
        traj_pubs[i].publish(get_display_robot_msg(trajectory[i].tolist(), colors[i]))
        

