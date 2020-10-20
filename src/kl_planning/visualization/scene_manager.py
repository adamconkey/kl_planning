#!/usr/bin/env python
import os
import sys
import yaml

import rospy
import rospkg
from visualization_msgs.msg import MarkerArray

from kl_planning.environments import Navigation2DEnvironment
from kl_planning.util import ros_util
from kl_planning.srv import SetPose, SetPoseResponse


class SceneManager:

    def __init__(self, rate=100):
        self.rate = rospy.Rate(rate)
        self.scene_msg = None
        self.start_goal_msg = None
        self.agent_msg = None
        
        self.scene_pub = rospy.Publisher("/scene", MarkerArray, queue_size=1)
        self.start_goal_pub = rospy.Publisher("/start_goal", MarkerArray, queue_size=1)
        self.agent_pub = rospy.Publisher("/agent", MarkerArray, queue_size=1)

        rospy.Service("/visualization/set_agent_location", SetPose, self._update_agent_location)

    def create_scene(self, env):
        self.env = env
        self.scene_msg = ros_util.get_marker_array_msg(env.object_config)
        self.start_goal_msg = ros_util.get_marker_array_msg(env.indicator_config)
        self._add_goal_text_markers()
        self.agent_msg = ros_util.get_marker_array_msg(env.agent_config)
                
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

    def _update_agent_location(self, req):
        self.agent_msg.markers[0].pose = req.pose
        return SetPoseResponse(success=True)

    def _add_goal_text_markers(self):
        marker_id = 1234
        for goal_id, goal_data in self.env.goal_config.items():
            if 'weight' not in goal_data:
                continue
            text_data = {
                'type': 'text',
                'text': str(goal_data['weight']),
                'position': [goal_data['state'][0], goal_data['state'][1], 0.75],
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
    r = rospkg.RosPack()
    path = r.get_path('kl_planning')
    config_path = os.path.join(path, 'config', 'scenes', config_filename)
    
    manager = SceneManager()
    env = Navigation2DEnvironment(config_path)
    rospy.on_shutdown(manager.shutdown)
    manager.create_scene(env)
    manager.run()
