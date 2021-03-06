#!/usr/bin/env python
import os
import sys
import yaml

import rospy
import rospkg
from visualization_msgs.msg import MarkerArray

from kl_planning.environments import Navigation2DEnvironment, ArmEnvironment
from kl_planning.util import ros_util, file_util, math_util
from kl_planning.srv import SetPose, SetPoseResponse


class SceneManager:
    """
    Scene manager for rendering the environment environment. Displays the agent,
    obstacles, goal/start indicators, etc..
    """

    def __init__(self, rate=500, n_interpolation_points=100):
        """
        Args:
            rate (float): ROS rate to run at
            n_interpolation_points (int): Number of points to interpolate with if being used
        """
        self.rate = rospy.Rate(rate)
        self.n_interpolation_points = n_interpolation_points
        self.scene_msg = None
        self.start_goal_msg = None
        self.agent_msg = None
        
        self.scene_pub = rospy.Publisher("/scene", MarkerArray, queue_size=1)
        self.start_goal_pub = rospy.Publisher("/start_goal", MarkerArray, queue_size=1)
        self.agent_pub = rospy.Publisher("/agent", MarkerArray, queue_size=1)

        rospy.Service("/visualization/set_agent_location", SetPose, self._update_agent_location)

    def create_scene(self, env):
        """
        Creates the visualizations for everything specified.

        Args:
            env (Environment): Environment object with configs specifying what's to be visualized.
        """
        self.env = env
        self.scene_msg = ros_util.get_marker_array_msg(env.object_config)
        if env.indicator_config:
            self.start_goal_msg = ros_util.get_marker_array_msg(env.indicator_config)
        self._add_goal_text_markers()
        if 'type' in env.agent_config:
            self.agent_msg = ros_util.get_marker_array_msg({'agent': env.agent_config})
                
    def run(self):
        """
        Main run function that enables visualization.
        """
        rospy.loginfo("Visualizing scene")
        while not rospy.is_shutdown():
            if self.scene_msg:
                self.scene_pub.publish(self.scene_msg)
            if self.start_goal_msg:
                self.start_goal_pub.publish(self.start_goal_msg)
            if self.agent_msg:
                self.agent_pub.publish(self.agent_msg)
            self.rate.sleep()

    def shutdown(self):
        rospy.loginfo("Exiting")

    def _update_agent_location(self, req):
        """
        Service offered that updates the agent's location in the scene.
        """
        if req.interpolate:
            poses = math_util.interpolate_poses(self.agent_msg.markers[0].pose, req.pose,
                                                self.n_interpolation_points)
            for pose in poses:
                self.agent_msg.markers[0].pose = pose
                self.rate.sleep()
        else:
            self.agent_msg.markers[0].pose = req.pose
        return SetPoseResponse(success=True)

    def _add_goal_text_markers(self):
        """
        Visualizes text above goals in rviz.
        """
        marker_id = 1234
        for goal_id, goal_data in self.env.goal_config.items():
            if 'weight' not in goal_data:
                continue
            text_data = {
                'type': 'text',
                'text': str(goal_data['weight']),
                'position': [goal_data['state'][0], goal_data['state'][1], 0.9],
                'orientation': [0, 0, 0, 1],
                'parent_frame': 'world',
                'color': [1, 1, 1, 1],
                'length': 0.3,
                'width': 0.3,
                'height': 0.3
            }
            self.start_goal_msg.markers.append(ros_util.get_marker_msg(text_data, marker_id))
            marker_id += 1            


if __name__ == '__main__':
    rospy.init_node('create_scene')
    config_filename = rospy.get_param('~config_filename')
    env_type = rospy.get_param('~env_type')
    r = rospkg.RosPack()
    path = r.get_path('kl_planning')
    config_path = os.path.join(path, 'config', 'scenes', config_filename)
    file_util.check_path_exists(config_path, "Scene configuration file")
    config = file_util.load_yaml(config_path)

    manager = SceneManager()
    if env_type == 'nav_2d':
        env = Navigation2DEnvironment(config)
    elif env_type == 'arm':
        env = ArmEnvironment(config)
    else:
        rospy.logerr(f"Unknown environment type: {env_type}")
        sys.exit(1)
    rospy.on_shutdown(manager.shutdown)
    manager.create_scene(env)
    manager.run()
