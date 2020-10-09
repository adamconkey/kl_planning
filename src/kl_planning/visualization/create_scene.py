#!/usr/bin/env python
import os
import sys
import yaml

import rospy
from visualization_msgs.msg import MarkerArray

from kl_planning.util import file_util, ros_util


class SceneManager:

    def __init__(self, rate=100):
        self.rate = rospy.Rate(rate)
        self.scene_msg = None
        self.start_goal_msg = None
        self.agent_msg = None
        
        self.scene_pub = rospy.Publisher("/scene", MarkerArray, queue_size=1)
        self.start_goal_pub = rospy.Publisher("/start_goal", MarkerArray, queue_size=1)
        self.agent_pub = rospy.Publisher("/agent", MarkerArray, queue_size=1)

    def create_scene(self, config_filename):
        file_util.check_path_exists(config_filename, "Scene configuration file")
        self.scene_config = file_util.load_yaml(config_filename)
        self.scene_msg = ros_util.get_marker_array_msg(self.scene_config['objects'])
        self.start_goal_msg = ros_util.get_marker_array_msg(self.scene_config['indicators'])
        self.agent_msg = ros_util.get_marker_array_msg(self.scene_config['agents'])
                
    def run(self):
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


if __name__ == '__main__':
    rospy.init_node('create_scene')
    config_filename = rospy.get_param('~config_filename')
    manager = SceneManager()
    rospy.on_shutdown(manager.shutdown)
    manager.create_scene(config_filename)
    manager.run()
