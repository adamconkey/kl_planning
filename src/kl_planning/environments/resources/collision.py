import sys
import torch
import pybullet
import pybullet_data
import numpy as np

from kl_planning.environments.resources import Panda
from kl_planning.util import pybullet_util, math_util


class CollisionChecker:

    def __init__(self, arm=None, env=[], start_arm_config=[], debug=False):
        """
        Pybullet collision checker.

        Args:
            arm (Arm): Arm instance for computing kinematics and visualization.
            debug (bool): Set true to run basic debug functionality.
        """
        self.arm = arm
        if debug:
            self.slider_ids = []
            for i in range(arm.num_joints):
                if start_arm_config:
                    slider_id = pybullet_util.add_slider("Joint {}".format(i), -np.pi, np.pi,
                                                         start_arm_config[i])
                else:
                    slider_id = pybullet_util.add_slider("Joint {}".format(i), -np.pi, np.pi, 0)
                self.slider_ids.append(slider_id)

        self.collision_objects = []
        self.add_collision_objects(env)

        if debug:
            self._run_debug()

    def add_collision_objects(self, env, client=pybullet):
        for config in env:
            if config[0] == pybullet.GEOM_SPHERE:
                if len(config) != 3:
                    raise ValueError("Sphere config should be tuple (GEOM, radius, [x,y,z])")
                geom, radius, origin = config
                self.add_sphere(radius, origin, client)
            elif config[0] == pybullet.GEOM_CYLINDER:
                if len(config) != 4:
                    raise ValueError("Cylinder config should be tuple (GEOM, radius, height, [x,y,z]")
                geom, radius, height, origin = config
                self.add_cylinder(radius, height, origin)
            else:
                raise ValueError("Supported GEOM types: [GEOM_SPHERE, GEOM_CYLINDER]")

    def add_sphere(self, radius, origin=[0, 0, 0], client=pybullet):
        """
        Adds a sphere collision object to the environment and registers a signed distance function.

        Registered SDF is a function over X,Y,Z positions and computes Euclidean distance between
        the query point and the origin of the sphere minus the sphere's radius.

        Args:
            radius (float): Radius of sphere.
            origin (List): X,Y,Z position coordinates for the origin of the sphere.
        """
        # TODO maybe you want to track shape ID also? Not sure if it's needed yet
        shape_id, obj_id = pybullet_util.add_sphere(radius, origin, client)
        self.collision_objects.append(obj_id)

    def add_cylinder(self, radius, height, origin=[0, 0, 0], client=pybullet):
        shape_id, obj_id = pybullet_util.add_cylinder(radius, height, origin, client)
        self.collision_objects.append(obj_id)

    def in_contact(self, epsilon=0.0):
        """
        Tests if robot is in collision with an object.

        Args:
            epsilon (float): Collision tolerance, makes contact check more conservative by
                             reporting contact if any point on robot is within distance
                             epsilon of an object.
        Returns:
            in_contact (bool): True if robot is in contact with any object, False otherwise.
        """
        in_contact = False
        for obj_id in self.collision_objects:
            points = pybullet.getClosestPoints(obj_id, self.arm.robot_id, epsilon)
            in_contact = in_contact or len(points) > 0
        return in_contact

    def dist_to_nearest_obstacle(self, max_dist=100.0):
        """
        Computes minimum signed distance between the robot and any obstacle.

        Args:
            max_dist (float): Cutoff distance for including "close" points. Set high to compute
                              for all points, lower if you want to limit search to a certain
                              proximity to objects.
        Returns:
            min_dist (float): Minimum distance between any point on the robot and any obstacle.
        """
        min_dist = sys.maxsize
        for obj_id in self.collision_objects:
            points = pybullet.getClosestPoints(obj_id, self.arm.robot_id, max_dist)
            obj_min_dist = min([p[8] for p in points])  # Index 8 is distance in point structure
            min_dist = min(min_dist, obj_min_dist)
        return min_dist
            
    def _run_debug(self):
        """
        Simple debug test of collision/distance functionality.

        You will be able to see registered collision objects, and move the arm 
        around with sliders to see it colliding with stuff.
        """
        while True:
            # Update joint state based on slider values
            for i in range(self.arm.num_joints):
                theta = pybullet.readUserDebugParameter(self.slider_ids[i])
                pybullet.resetJointState(self.arm.robot_id, i, theta)
            print('-' * 40)
            print("IN CONTACT", self.in_contact())
            print("DIST", self.dist_to_nearest_obstacle())


if __name__ == '__main__':
    arm = Panda()
    env = [(pybullet.GEOM_SPHERE, 0.2, [0.4, 0, 0.8])]
    checker = CollisionChecker(arm, env, debug=True)
