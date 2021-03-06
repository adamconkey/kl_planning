#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from kl_planning.util import vis_util
from kl_planning.srv import VisualizeTrajectorySamples, VisualizeTrajectorySamplesResponse

class TrajectorySampleVisualizer:

    def __init__(self, rate=100, debug=False):
        self.rate = rospy.Rate(rate)
        self.debug = debug
        
        self.samples_msg = None
        self.samples_pub = rospy.Publisher("/visualization/trajectory_samples",
                                           MarkerArray, queue_size=1)
        rospy.Service("/visualization/visualize_trajectory_samples",
                      VisualizeTrajectorySamples, self._visualize_samples)

    def run(self):
        rospy.loginfo("Ready to visualize trajectory samples")
        while not rospy.is_shutdown():
            if self.samples_msg:
                self.samples_pub.publish(self.samples_msg)
            self.rate.sleep()

    def _visualize_samples(self, req):
        if req.samples:
            samples = np.array(req.samples).reshape(req.shape)
        costs = list(req.costs)
        size = req.size
        if self.debug:
            self._debug_log(f"Samples shape: {samples.shape}")
            self._debug_log(f"Costs length: {len(costs)}")
            self._debug_log(f"Size: {size}")

        if costs:
            colors = vis_util.get_color_sequence(len(costs), 'coolwarm')
            colors = [list(c) + [1] for c in colors] # Add alpha
        elif req.colors:
            colors = [[c.r, c.g, c.b, c.a] for c in req.colors]
        else:
            colors = [[0.44, 0.7, 0.96, 1] for _ in range(samples.shape[1])]
        self.samples_msg = MarkerArray([self._get_marker(samples[:,i,:], colors[i], i, size)
                                        for i in range(samples.shape[1])])
        return VisualizeTrajectorySamplesResponse(success=True)

    def _get_marker(self, points, color, marker_id=0, size=0.01, frame_id='world', z=0.05):
        m = Marker()
        m.header.frame_id = frame_id
        m.id = marker_id
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = size
        m.pose.orientation.w = 1
        m.color = ColorRGBA(*color)
        for i in range(len(points)):
            m.points.append(Point(points[i, 0], points[i, 1], z))
        return m
    
    def _debug_log(self, msg):
        rospy.logwarn(f"[{type(self).__name__}] {msg}")

    def _debug(self):
        self._debug_log("DEBUG")
        from kl_planning.util.vis_util import visualize_trajectory_samples
        rospy.wait_for_service("/visualization/visualize_trajectory_samples")
        samples = np.random.rand(10, 5, 2)
        costs = np.random.rand(5)
        self._debug_log("Calling visualize samples")
        visualize_trajectory_samples(samples, costs)
        self._debug_log("Samples should be visualized")
    
        
if __name__ == '__main__':
    rospy.init_node('trajectory_sample_visualizer')
    rate = rospy.get_param("~rate", 100)
    debug = rospy.get_param("~debug", False)
    visualizer = TrajectorySampleVisualizer(rate, debug)

    if debug:
        import multiprocessing
        p = multiprocessing.Process(target=visualizer._debug)
        p.start()
    
    visualizer.run()
