import os
import sys
import time
import torch
import pybullet
import pybullet_data
import numpy as np

from kl_planning.util import pybullet_util, math_util

BULLET_DATA_PATH = pybullet_data.getDataPath()
PANDA_URDF_PATH = os.path.join(BULLET_DATA_PATH, "franka_panda/panda.urdf")


class Panda:

    def __init__(self, urdf_path=PANDA_URDF_PATH, ee_index=8, device=torch.device('cuda')):
        self.urdf_path = urdf_path
        self.ee_index = ee_index  # 8 is panda_hand
        self.num_joints = 7
        self.device = device
        
        self.robot_id = pybullet_util.load_urdf(urdf_path)
        for i in range(self.num_joints):
            pybullet.resetJointState(self.robot_id, i, 0)
        
    def fk(self, thetas):
        """
        Computes forward kinematics.

        Args:
            thetas (Tensor): Joint anles (B, n_theta)
        Returns:
            ps (Tensor): EE positions (B, 3)
            qs (Tensor): EE orientations as quaterions (B, 4)
        """
        ps = []
        qs = []
        for i in range(len(thetas)):
            for j in range(thetas.size(-1)):
                pybullet.resetJointState(self.robot_id, j, thetas[i,j])
            _, _, _, _, p, q = pybullet.getLinkState(self.robot_id, self.ee_index,
                                                     computeForwardKinematics=True)
            ps.append(torch.tensor(p).to(self.device))
            qs.append(torch.tensor(q).to(self.device))
        return torch.stack(ps), torch.stack(qs)
        
    def position_error(self, p_desired, p_actual, sum_squares=True):
        """
        Computes sum of squares error between 3D positions.

        (B, 3) each
        """
        error = p_desired - p_actual
        if sum_squares:
            error = (error**2).sum(dim=-1)
        return error

    def orientation_error(self, q_desired, q_actual, device=torch.device('cuda')):
        B = q_desired.size(0)
        R_desired = math_util.rot_from_quat(q_desired)
        R_actual = math_util.rot_from_quat(q_actual)
        
        R_diff = torch.bmm(torch.transpose(R_actual, -2, -1), R_desired)
        I = torch.eye(3, dtype=torch.double, device=device).unsqueeze(0).repeat(B, 1, 1)
        error = torch.norm(R_diff - I, p='fro')**2
        return error
