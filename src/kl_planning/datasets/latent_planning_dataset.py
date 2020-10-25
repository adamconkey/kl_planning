import sys
import h5py
import numpy as np
from math import inf
from tqdm import tqdm

import torch
import torch.nn.functional as F

from kl_planning.util.data_util import scale_min_max


class LatentPlanningDataset(torch.utils.data.Dataset):

    def __init__(self, obs_modalities=[], act_modalities=[], filenames=[], chunk_size=25,
                 time_subsample=1, network_in_min=-1, network_in_max=1):
        super().__init__()
        self.obs_modalities = obs_modalities
        self.act_modalities = act_modalities
        self.filenames = filenames
        self.chunk_size = chunk_size
        self.time_subsample = time_subsample
        self.network_in_min = network_in_min
        self.network_in_max = network_in_max
        self.goal_weights = [] # For cross entropy loss weighting
        self.sample_options = []

        # Track range of modality values for min/max scaling on inputs. Some start with priors
        self.data_ranges = {
            'gripper_joint_positions': [0.0, 0.04],
            'joint_positions': [-np.pi, np.pi],
            'delta_joint_positions': [inf, -inf]
        }

        if self.filenames:
            self.process_filenames()

    def __len__(self):
        return len(self.sample_options)

    def __getitem__(self, idx):
        """
        Note relative time shifting of actions/observations for causality is handled 
        in loss computation, so they are kept time-aligned here.
        """
        filename, start_idx = self.sample_options[idx]
        end_idx = start_idx + (self.time_subsample * self.chunk_size)
        with h5py.File(filename, 'r') as h5_data:
            obs = {}
            act = {}
            init_obs = {}
            init_act = {}
            for m in self.obs_modalities:
                obs[m] = self.process_data_in(h5_data[m][start_idx:end_idx:self.time_subsample], m)
                init_obs[m] = self.process_data_in(h5_data[m][:1], m)
            for m in self.act_modalities:
                if m == 'delta_joint_positions':
                    data = h5_data['joint_positions'][start_idx:end_idx:self.time_subsample]
                    if start_idx < self.time_subsample:
                        first = data[0]
                    else:
                        first = h5_data['joint_positions'][start_idx - self.time_subsample]
                    data = np.concatenate([np.expand_dims(first, 0), data], axis=0)
                    act[m] = self.process_data_in(data[1:] - data[:-1], m)
                    init_act[m] = self.process_data_in(np.zeros((1, act[m].shape[-1])), m)
                else:
                    act[m] = self.process_data_in(h5_data[m][start_idx:end_idx:self.time_subsample], m)
                    init_act[m] = self.process_data_in(h5_data[m][:1], m)
                            
            goal = self._get_goal(h5_data, start_idx, end_idx)
            
            sample = {'obs': obs, 'act': act,
                      'init_obs': init_obs, 'init_act': init_act,
                      'goal': goal}
        return sample

    def process_data_in(self, data, modality):
        if modality == 'rgb':
            data = data.transpose(0, 3, 1, 2) # (t, 3, h, w)
            data = torch.tensor(data, dtype=torch.float32)
            # TODO parameterize image size
            data = F.interpolate(data, (64, 64), mode='bilinear', align_corners=False) # (t, 3, 64, 64)
            data.div_(255) # Normalize to range [0, 1]
        elif modality == 'depth':
            data = np.expand_dims(data, 1) # Add channel: (t, 1, h, w)
            data = torch.tensor(data, dtype=torch.float32)
            data = F.interpolate(data, (64, 64), mode='bilinear', align_corners=False) # (t, 1, 64, 64)
        elif modality in self.data_ranges:
            data = scale_min_max(data,
                                 self.data_ranges[modality][0],
                                 self.data_ranges[modality][1],
                                 self.network_in_min,
                                 self.network_in_max)
            data = torch.tensor(data, dtype=torch.float32)
        else:
            print(f"Unknown modality for processing data into network: {modality}")
            data = None
        return data

    def process_data_out(self, data, modality):
        data = data.squeeze(dim=1) # TODO if you do batches then really need to view t,b in one dim
        if modality == 'rgb':
            # TODO can parameterize img size
            data = F.interpolate(data, (512, 512), mode='bilinear', align_corners=False)
            data = data.cpu().detach().numpy()
            data *= 255
            data = data.astype(np.uint8)
            data = data.transpose(0, 2, 3, 1) # (t, h, w, 3)
        elif modality in self.data_ranges:
            data = data.cpu().detach().numpy().squeeze()
            data = scale_min_max(data,
                                 self.network_in_min,
                                 self.network_in_max,
                                 self.data_ranges[modality][0],
                                 self.data_ranges[modality][1])
        else:
            print(f"Unknown modality for processing data out of network: {modality}")
            data = None
        return data

    def process_filenames(self):
        """
        Generates all sample options (filename and start index) and determines 
        goal weights for computing weighted cross entropy loss.
        """
        # Figure out which goals there are (every H5 stores all possible goals)
        with h5py.File(self.filenames[0], 'r') as h5_file:
            goals = h5_file.attrs['task_goals'].tolist()
            goal_counts = [0] * len(goals)
        
        # Generate all possible sample options (file and start idx)
        self.sample_options = []
        for filename in self.filenames:
            with h5py.File(filename, 'r') as h5_file:
                n_steps = min([len(v) for v in h5_file.values() if isinstance(v, h5py.Dataset)])
                idxs = list(range(n_steps - (self.chunk_size * self.time_subsample) - 1))
                for start_idx in idxs:
                    self.sample_options.append((filename, start_idx))
        
                # Count all the goals for weighting cross entropy loss
                for i, goal in enumerate(h5_file.attrs['task_goals']):
                    goal_counts[goals.index(goal)] += np.sum(h5_file['goal_status'][:,i])
                
                # Check min/max for data scaling
                for modality in self.data_ranges:
                    if modality == 'delta_joint_positions':
                        data = h5_file['joint_positions'][::self.time_subsample]
                        deltas = data[1:] - data[:-1]
                        h5_min = np.max(deltas)
                        h5_max = np.max(deltas)
                    else:
                        h5_min = np.min(h5_file[modality])
                        h5_max = np.max(h5_file[modality])
                    if h5_min < self.data_ranges[modality][0]:
                        self.data_ranges[modality][0] = h5_min
                    if h5_max > self.data_ranges[modality][1]:
                        self.data_ranges[modality][1] = h5_max
        
        total_count = float(sum(goal_counts))
        self.goal_weights = [1. - (count / total_count) for count in goal_counts]

    def state_dict(self):
        return vars(self)

    def load_state_dict(self, _dict):
        for k, v in _dict.items():
            setattr(self, k, v)
        
    def _get_goal(self, h5_data, start_idx, end_idx):
        goal = np.array(h5_data['goal_status'][start_idx:end_idx:self.time_subsample])
        # Give K-level index to each non-zero goal bit going across time
        goal = goal * np.arange(goal.shape[-1])
        # Flatten along goal axis to get the final K-level targets across time
        goal = np.sum(goal, axis=1)
        return goal
