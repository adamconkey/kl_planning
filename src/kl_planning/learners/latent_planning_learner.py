#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import h5py
import cv2
import numpy as np
from tqdm import tqdm
from time import time

import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from kl_planning.util import data_util, file_util, ui_util
from kl_planning.models import torch_helper
from kl_planning.datasets import LatentPlanningDataset
from kl_planning.common.config import default_config


class LatentPlanningLearner:

    def __init__(self, params={}, checkpoint_filename='', device=torch.device('cuda')):
        self.config = default_config()
        self.device = device

        self.checkpoint = None
        if checkpoint_filename:
            self._load_checkpoint(checkpoint_filename)
        elif params:
            self.config.update_with_dict(params, True) # Override anything passed from command line

        self.dataset = LatentPlanningDataset(self.config.obs_modalities,
                                             self.config.act_modalities,
                                             chunk_size=self.config.chunk_size,
                                             time_subsample=self.config.time_subsample)
        if self.checkpoint:
            self.dataset.load_state_dict(self.checkpoint['dataset'])
        
        self.models = self.config.models
        self.send_models_to_device()
        
        # Global prior N(0, I)
        mean = torch.zeros(self.config.batch_size, self.config.state_size, device=self.device)
        cov = torch.ones(self.config.batch_size, self.config.state_size, device=self.device)
        self.global_prior = Normal(mean, cov)
        # Allowed deviation in KL divergence
        self.free_nats = torch.full((1, ), self.config.free_nats, dtype=torch.float32, device=device)

        # Useful to see config at console for catching configuration mistakes
        print(f"\n{'-'*20} CONFIG {'-'*20}")
        for k, v in self.config.to_dict().items():
            print(f"{k:>20}: {v}")
        print("-" * 48)
        
    def train(self, episode, h5_filenames, checkpoint_dir):
        print(f"\nLearning from Episode {episode} data "
              f"({self.config.n_epochs_per_episode} epochs)...")

        self.parameters = [p for m in self.models.values() for p in m.parameters()]
            
        self.optimizer = Adam(self.parameters, lr=self.config.lr, eps=self.config.adam_epsilon)
        self.scheduler = StepLR(self.optimizer, step_size=self.config.lr_step)
        
        self.dataset.filenames = h5_filenames
        self.dataset.process_filenames()
        data_loader = DataLoader(self.dataset, self.config.batch_size, shuffle=True,
                                 num_workers=1, pin_memory=True, drop_last=False)
        
        metrics = {'observation_loss': [], 'kl_loss': []}

        self.set_models_to_train()
        
        for epoch in range(1, self.config.n_epochs_per_episode + 1):
            pbar = tqdm(total=len(data_loader.dataset), file=sys.stdout)
            obs_losses = []
            kl_losses = []
            for batch in data_loader:
                torch_helper.move_batch_to_device(batch, self.device)
                obs_loss, kl_loss = self._compute_loss(batch)
                obs_losses.append(obs_loss)
                kl_losses.append(kl_loss)
                # TODO probably want more meaningful way to track loss, rolling window average?
                desc = f'  Epoch {epoch}: ' \
                       f'obs_loss={np.mean(obs_losses):.4f}, ' \
                       f'kl_loss={np.mean(kl_losses):.4f}'
                pbar.set_description(desc)
                pbar.update(self.config.batch_size)
            pbar.close()

            # TODO maybe compute this differently, e.g. over window?
            metrics['observation_loss'].append(np.mean(obs_losses))
            metrics['kl_loss'].append(np.mean(kl_losses))

            file_util.save_pickle(metrics, os.path.join(checkpoint_dir, 'metrics.pickle'))

            if epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(episode, epoch, checkpoint_dir)

            self.scheduler.step()
                
        ui_util.print_happy(f"\nEpisode {episode} learning complete.\n")
        
    def compute_transition(self, act, obs=None, init_belief=None, init_state=None):
        # Last batch can differ if dataset len not divisible by batch size
        batch_size = next(iter(act.values())).size(1)

        if init_belief is None:
            init_belief = torch.zeros(batch_size, self.config.belief_size, device=self.device)
        if init_state is None:
            init_state = torch.zeros(batch_size, self.config.state_size, device=self.device)

        act = torch.cat([act[modality] for modality in self.config.act_modalities], dim=-1)
        
        # Encode all observations (additionally use multisensory encoder if multiple modalities)
        if obs is not None:
            encoded = []
            for modality in self.config.obs_modalities:
                encoded.append(data_util.process_time_batch(self.models[f'{modality}_encoder'],
                                                            obs[modality]))
            encoded = torch.cat(encoded, dim=-1)
            if len(self.config.obs_modalities) > 1:
                encoded = data_util.process_time_batch(self.models['multisensory_encoder'], encoded)
        else:
             encoded = None

        return self.models['dynamics_model'](init_state, act, init_belief, encoded)

        
    def predict_outputs(self, init_obs, init_act, act):
        """
        Generates forward predictions of observations.
        """
        current_rssm_out = self.compute_transition(init_act, init_obs)
        belief = current_rssm_out.beliefs[-1]
        state = current_rssm_out.posterior_states[-1]
        rssm_out = self.compute_transition(act, init_state=state, init_belief=belief)
        decoded = self.decode_state(rssm_out.prior_states)
        # # Get latent state at beginning of task for goal classifier
        # start_rssm_out = self.compute_transition(task_start_act, task_start_obs)
        # start_state = start_rssm_out.posterior_states[-1]
        # plt.imshow(self.decode_state(start_state.unsqueeze(0))['rgb'].squeeze())
        # plt.show()
        return decoded
    
    def decode_state(self, state):
        with torch.no_grad():
            decodes = {}
            for modality in self.config.dec_modalities:
                decoded = data_util.process_time_batch(self.models[f'{modality}_decoder'], state)
                decodes[modality] = self.dataset.process_data_out(decoded, modality)
        return decodes

    def send_models_to_device(self):
        for model in self.models.values():
            model.to(self.device)
    
    def set_models_to_train(self):
        for model in self.models.values():
            model.train()

    def set_models_to_eval(self):
        for model in self.models.values():
            model.eval()
    
    def _compute_loss(self, batch):
        # NOTE: There is a tricky thing here with obs/act alignment that will be confusing
        # especially if you look at PlaNet code. Actions need to be aligned a step ahead of
        # observations to account for causality. BUT, my data is recorded agnostic to an
        # action modality, it's all time-aligned observations and I am just interpreting
        # some modality as the action. Normally you would want to shift actions ahead of
        # observations. But the transition function operates on previous action and current
        # observation. This all amounts to NOT shifting either the obs or action.
        # Think in joint angles, if both obs and act and joint angles, at the last timestep
        # I took action q1, my current observation should be q1 (assuming perfect control in
        # that timestep).

        # Swap batch and time indices so they are (t, b, *channels)
        obs = {k: v.transpose(0, 1) for k, v in batch['obs'].items()}
        act = {k: v.transpose(0, 1) for k, v in batch['act'].items()}
        init_obs = {k: v.transpose(0, 1) for k, v in batch['init_obs'].items()}
        init_act = {k: v.transpose(0, 1) for k, v in batch['init_act'].items()}

        # # Get state from start of demo, used as context to the goal classifier
        # init_rssm_out = self.compute_transition(init_act, init_obs)
        # init_state = init_rssm_out.posterior_states[-1]

        # Get priors/posteriors over time
        rssm_out = self.compute_transition(act, obs)
        
        # Compute all losses
        obs_loss = self._compute_observation_loss(rssm_out, obs)
        kl_loss = self._compute_kl_loss(rssm_out)
        loss = obs_loss + kl_loss
        
        # Update model parameters
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, self.config.grad_clip_norm, norm_type=2)
        self.optimizer.step()

        obs_loss = obs_loss.item()
        kl_loss = kl_loss.item()
        return obs_loss, kl_loss
    
    def _compute_observation_loss(self, rssm_out, obs):
        # Compute observation likelihood; sum over final dims, average over batch and time
        # TODO not sure if decoders should be over beliefs+state or just state. PlaNet (pytorch)
        # does the former, and that is expressed also in paper. But Google code does only state,
        # and state is a function of belief so in paper might just be implicitly dependent on
        # both. Trying just state for now since it's a little simpler.

        observation_loss = 0
        for modality, target in obs.items():
            decoded = data_util.process_time_batch(self.models[f'{modality}_decoder'],
                                                   rssm_out.posterior_states)
            modality_loss = F.mse_loss(decoded, target, reduction='none')
            sum_dims = list(range(modality_loss.dim()))[2:] # Exclude time/batch dims
            modality_loss = modality_loss.sum(dim=sum_dims) # Sum over channel dims
            observation_loss += modality_loss
        observation_loss = observation_loss.mean(dim=(0, 1)) # Mean over time/batch dims
        return observation_loss

    def _compute_kl_loss(self, rssm_out):
        # Compute KL loss (again sum final dims and average over batch and time)
        prior_state = Normal(rssm_out.prior_means, rssm_out.prior_std_devs)
        posterior_state = Normal(rssm_out.posterior_means, rssm_out.posterior_std_devs)
        kl_div = kl_divergence(posterior_state, prior_state).sum(dim=2)
        kl_loss = torch.max(kl_div, self.free_nats).mean(dim=(0, 1))
        if self.config.global_kl_beta != 0:
            global_kl_div = kl_divergence(posterior_state, self.global_prior).sum(dim=2)
            global_kl_div = global_kl_div.mean(dim=(0, 1))
            kl_loss += self.config.global_kl_beta * global_kl_div
        return kl_loss
    
    def _save_checkpoint(self, episode, epoch, checkpoint_dir):
        checkpoint = self.config.to_dict()
        for name, model in self.models.items():
            checkpoint[name] = model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint['scheduler'] = self.scheduler.state_dict()
        checkpoint['dataset'] = self.dataset.state_dict()
        filename = os.path.join(checkpoint_dir, f'episode_{episode:03d}_epoch_{epoch:03d}.pt')
        torch.save(checkpoint, filename)
        print(f"\nCheckpoint saved: {filename}\n")

    def _load_checkpoint(self, checkpoint_filename):
        file_util.check_path_exists(checkpoint_filename)
        print(f"\nUsing checkpoint {checkpoint_filename}")
        self.checkpoint = torch.load(checkpoint_filename)
        self.config.update_with_dict(self.checkpoint)
        for name, model in self.config.models.items():
            model.load_state_dict(self.checkpoint[name])

    def _preprocess_depth(self, depth):
        depth = depth.copy() # Needed to avoid writable warning from torch, not sure why
        # TODO Seems depth from ROS bridge is in different units, need to sort that out
        depth /= 1000.
        depth = np.expand_dims(depth, 0) # Add channel
        depth = np.expand_dims(depth, 0) # Add batch
        depth = torch.from_numpy(depth)
        depth = F.interpolate(depth, size=(64, 64), mode='bilinear', align_corners=False)
        depth = depth.to(self.device)
        return depth

    def _postprocess_depth(self, depth):
        depth = depth.cpu().detach().numpy().squeeze()
        depth = cv2.resize(depth, (512, 512), interpolation=cv2.INTER_LINEAR)
        return depth

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Directory for checkpoints')
    parser.add_argument('--checkpoint', type=str, default='', help="Checkpoint file to load")
    parser.add_argument('--expert_h5_root', type=str, required=True,
                        help='Root directory of expert demo H5 files')
    parser.add_argument('--learner_h5_root', type=str,
                        help='Root directory of learner execution H5 files')
    parser.add_argument('--seed', type=int, default=1, help='Seed for torch/numpy randomization')
    parser.add_argument('--use_cpu', action='store_true', help='Force run on CPU')
    parser.add_argument('--params', type=str, default='{}',
                        help='YAML-formatted dictionary for overriding default parameters')
    parser.add_argument('--episode', type=int, default=1) # TODO infer?
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.use_cpu else 'cpu')

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Initialize with expert demos, then accumulate any learner episode execution data
    file_util.check_path_exists(args.expert_h5_root)
    h5_filenames = [f for f in file_util.list_dir(args.expert_h5_root) if f.endswith('.h5')]
    if args.learner_h5_root:
        file_util.check_path_exists(args.learner_h5_root)
        h5_filenames += [f for f in file_util.list_dir(args.learner_h5_root) if f.endswith('.h5')]
        
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    params = yaml.safe_load(args.params)

    learner = LatentPlanningLearner(params, args.checkpoint, device)
    learner.train(args.episode, h5_filenames, args.checkpoint_dir)
