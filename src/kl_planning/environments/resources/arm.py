import os
import time
import torch
import pybullet
import pybullet_data
import numpy as np

from kl_planning.util import pybullet_util, math_util

BULLET_DATA_PATH = pybullet_data.getDataPath()


class Arm:
    """
    Base class implementing functionality of a robot arm with differentiable forward kinematics.
    """
    def __init__(self, alphas, Ds, As, urdf_path, ee_index=-1, modified_DH=False):
        """
        Args:
            alphas: List of 'alpha' DH parameters.
            Ds: List of 'd' DH parameters.
            As: List of 'a' DH parameters.

        Note: Assuming all revolute joints, so Ds are always fixed and thetas always variable.
        """
        self.alphas = torch.DoubleTensor(alphas).view(-1, 1)
        self.Ds = torch.DoubleTensor(Ds).view(-1, 1)
        self.As = torch.DoubleTensor(As).view(-1, 1)
        self.num_joints = len(alphas)
        self.urdf_path = urdf_path
        self.ee_index = ee_index
        self.modified_DH = modified_DH

        self.robot_id = pybullet_util.load_urdf(urdf_path)
        for i in range(self.num_joints):
            pybullet.resetJointState(self.robot_id, i, 0)

    def fk(self, thetas):
        """
        Computes forward kinematics.

        Args:
            thetas (Tensor): Joint angles
        Returns:
            T: Homogeneous TF matrix representing end-effector 3D pose.
        """
        return self.fk_links(thetas)[-1]

    def fk_links(self, thetas):
        """
        Computes forward kinematics for all links.

        Args:
            thetas (Tensor): Joint angles
        Returns:
            Ts (list): List of homogeneous TF matrices, one for each link in kinematic chain.
        """
        Ts = []
        T = torch.eye(4, dtype=torch.double)
        for i in range(len(thetas)):
            T_i = self._compute_T(self.alphas[i], thetas[i], self.Ds[i], self.As[i])
            T = torch.mm(T, T_i)
            Ts.append(T)
        return Ts
    
    def get_link_positions(self, thetas):
        """
        Computes the position of each link in Cartesian space w.r.t. to robot base.
        """
        Ts = self.fk_links(thetas)
        positions = [math_util.pos_from_homogeneous(T) for T in Ts]
        return positions
        
    def pose_error(self, T_desired, T_actual):
        position_error = self.ee_position_error(T_desired, T_actual)
        orientation_error = self.ee_orientation_error(T_desired, T_actual)
        # TODO add weighting option so you can weight position/orientation error differently,
        # or weight errors on different axes differently
        error = position_error + orientation_error
        return error

    def ee_position_error(self, T_desired, T_actual, sum_squares=True):
        """
        Computes sum of squares error between position components of two homogeneous TF matrices.
        """
        p_desired = math_util.pos_from_homogeneous(T_desired)
        p_actual = math_util.pos_from_homogeneous(T_actual)
        error = p_desired - p_actual
        if sum_squares:
            error = torch.sum(error**2)
        return error

    def ee_orientation_error(self, T_desired, T_actual):
        """
        Computes orientation error as Frobenius norm on difference of rotation matrices.

        TODO: Will want to accommodate other orientation measures, particularly over different
              orientation representations. Can either add a flag on this function or just break
              out into separate functions for each.
        """
        R_desired = math_util.rot_from_homogeneous(T_desired)
        R_actual = math_util.rot_from_homogeneous(T_actual)
        R_diff = torch.mm(R_actual.t(), R_desired)
        # TODO maybe want to compute this by hand since it's computing a sqrt and squaring it
        error = torch.norm(R_diff - torch.eye(3, dtype=torch.double), p='fro')**2
        return error
        
    def _compute_T(self, alpha, theta, d, a):
        """
        Computes homogeneous transformation matrix.

        If self.modified_DH=True, uses modified DH convention:
        https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters#Modified_DH_parameters
        """
        ct = torch.cos(theta)
        st = torch.sin(theta)
        ca = torch.cos(alpha)
        sa = torch.sin(alpha)
        # These are a little silly, but necessary for autograd not to yell at you
        one = torch.DoubleTensor([1.0])
        zero = torch.DoubleTensor([0.0])
        
        if self.modified_DH:
            T = torch.stack([
                torch.stack([     ct,      -st, zero,       a]),
                torch.stack([st * ca,  ct * ca,  -sa, -d * sa]),
                torch.stack([st * sa,  ct * sa,   ca,  d * ca]),
                torch.stack([   zero,     zero, zero,     one])
            ])
        else:
            T = torch.stack([
                torch.stack([  ct, -st * ca,  st * sa, a * ct]),
                torch.stack([  st,  ct * ca, -ct * sa, a * st]),
                torch.stack([zero,       sa,       ca,      d]),
                torch.stack([zero,     zero,     zero,    one])
            ])
            
        return T.squeeze()
        

class Panda(Arm):

    def __init__(self, urdf_path=os.path.join(BULLET_DATA_PATH, "franka_panda/panda.urdf")):
        """
        DH parameters from: 
          https://frankaemika.github.io/docs/control_parameters.html#denavithartenberg-parameters

        Note: These are in the Modified DH convention, the FK function accounts for this.
        """
        alphas = [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2]
        Ds = [0.333, 0, 0.316, 0, 0.384, 0, 0]
        As = [0, 0, 0, 0.0825, -0.0825, 0, 0.088]
        ee_index = 6  # TODO right now this is flange, need to make hand once transform is computed
        super().__init__(alphas, Ds, As, urdf_path, ee_index, True)

    def fk(self, thetas):
        T = super().fk(thetas)
        # Adding one more fixed transform for the flange
        alpha = torch.DoubleTensor([0.0])
        theta = torch.DoubleTensor([0.0])
        d = torch.DoubleTensor([0.107])
        a = torch.DoubleTensor([0.0])
        T = torch.mm(T, self._compute_T(alpha, theta, d, a))
        # TODO there's one more transform to add from flange to hand, need to manually find it
        return T
        
