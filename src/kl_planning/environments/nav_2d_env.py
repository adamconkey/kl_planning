import torch


class Navigation2DEnvironment:

    def __init__(self):
        pass

    def dynamics(self, start_xy, act, repeat=True):
        """
        start_xy (b, 2) x, y
        act (b, 2) dist, angle
        """
        delta_x = act[:,0] * torch.cos(act[:,1])
        delta_y = act[:,0] * torch.sin(act[:,1])
        delta_xy = torch.stack([delta_x, delta_y], dim=-1).unsqueeze(1)
        if repeat:
            delta_xy = delta_xy.repeat(1, 5, 1) # 2n+1 = 5 sigma points, just hard-coding for now
        next_xy = start_xy + delta_xy
        return next_xy
