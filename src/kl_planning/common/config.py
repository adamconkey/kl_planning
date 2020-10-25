import torch
from dataclasses import dataclass, fields
from typing import List, Dict

from kl_planning.models import (
    RSSM, ImageEncoder, ImageDecoder, MultisensoryEncoder,
    MulticlassClassifier, VectorEncoder, VectorDecoder
)


# These configs are based on planet's AttrDict, but here using dataclasses as they
# achieve the same thing but are proper Python built-ins (as of 3.7, and can be
# back-installed to earlier python versions)

@dataclass
class Config:
    """
    Empty data class that will hold all learning config values, e.g. hyperparameters,
    training session configuration, model configurations, etc. Makes it simple to pass
    around a single object to setup models, checkpoint, compute losses, etc.

    TODO can make this more structured by having instances of other dataclasses that
    organize by type (e.g. ModelConfig, TrainConfig, etc.). I don't want to take the
    time to manage that right now though and will just garbage dump everything in 
    a flat structure for now.
    """
    belief_size: int = None
    state_size: int = None
    hidden_size: int = None
    embedding_size: int = None
    vec_embedding_size: int = None
    img_embedding_size: int = None
    action_size: int = None
    activation: str = None
    min_std_dev: float = None
    me_hidden_sizes: List[int] = None
    gc_hidden_sizes: List[int] = None
    vec_hidden_sizes: List[int] = None

    seed: int = None
    lr: float = None
    lr_step: float = None
    adam_epsilon: float = None
    free_nats: int = None
    grad_clip_norm: int = None
    batch_size: int = None
    n_epochs_per_episode: int = None
    checkpoint_interval: int = None
    global_kl_beta: float = None

    time_subsample: int = None
    chunk_size: int = None
    n_goals: int = None
    data_ranges: Dict[str, List[int]] = None
    obs_modalities: List[str] = None
    act_modalities: List[str] = None
    dec_modalities: List[str] = None
    
    def to_dict(self):
        _dict = {}
        for field in fields(self):
            value = getattr(self, field.name)
            # TODO for now skipping models as they are being done separately, not sure if there's
            # a good way to load from dict. Easy to save models to state_dict but then loading you
            # need to know what kind of model it is.
            if value is None or field.type == torch.nn.Module:
                continue
            _dict[field.name] = value
        return _dict

    def update_with_dict(self, _dict, update_models=True):
        """
        Override fields with values provided by dictionary. Note update_models=True
        will create new models with the updated values, set to False if loading from
        a checkpoint.
        """
        for field in fields(self):
            # TODO also skipping models here, need to decide if you want to try loading them here
            if field.name not in _dict or field.type == torch.nn.Module:
                continue
            setattr(self, field.name, _dict[field.name])
        if update_models:
            _set_models(self)


def default_config(override_params={}):
    config = Config()
    _process_data_params(config, override_params)
    _process_model_params(config, override_params)
    _process_learning_params(config, override_params)
    _set_models(config)
    return config


def _process_model_params(config, params):
    config.belief_size = params.get('belief_size', 256)
    config.state_size = params.get('state_size', 256)
    config.hidden_size = params.get('hidden_size', 512)
    config.embedding_size = params.get('embedding_size', 256)
    config.vec_embedding_size = params.get('vec_embedding_size', 8)
    config.img_embedding_size = params.get('img_embedding_size', 256)
    config.action_size = params.get('action_size', 7)  # TODO should get from some kind of env config
    config.activation = params.get('activation', 'leaky_relu')
    config.min_std_dev = params.get('min_std_dev', 0.1)
    config.me_hidden_sizes = params.get('me_hidden_sizes', [1024, 1024])
    config.gc_hidden_sizes = params.get('gc_hidden_sizes', [128, 128])
    config.vec_hidden_sizes = params.get('vec_hidden_sizes', [32, 32])

    
def _set_models(config):
    # TODO this can be made nicer haveing e.g. get_encoder functions that you can just pass
    # in modality names and config and get the corresponding model
    
    config.models = {}
    multisensory_in_size = 0
    
    # Encoders
    if 'rgb' in config.obs_modalities:
        config.models['rgb_encoder'] = ImageEncoder(3, config.img_embedding_size, config.activation)
        multisensory_in_size += config.img_embedding_size
    if 'depth' in config.obs_modalities:
        config.models['depth_encoder'] = ImageEncoder(1, config.img_embedding_size, config.activation)
        multisensory_in_size += config.img_embedding_size
    if 'joint_positions' in config.obs_modalities:
        # TODO joint angle size is hard-coded for now
        config.models['joint_positions_encoder'] = VectorEncoder(7,
                                                                 config.vec_embedding_size,
                                                                 config.vec_hidden_sizes,
                                                                 activation=config.activation)
        multisensory_in_size += config.vec_embedding_size
    if 'gripper_joint_positions' in config.obs_modalities:
        # TODO size is hard-coded for now, also probably be overkill for length 2 vector
        config.models['gripper_joint_positions_encoder'] = VectorEncoder(2,
                                                                         config.vec_embedding_size,
                                                                         config.vec_hidden_sizes,
                                                                         activation=config.activation)
        multisensory_in_size += config.vec_embedding_size

    # Multisensory Encoder
    if len(config.obs_modalities) > 1:
        multisensory_encoder = MultisensoryEncoder(multisensory_in_size,
                                                   config.embedding_size,
                                                   hidden_sizes=config.me_hidden_sizes,
                                                   activation=config.activation)
        config.models['multisensory_encoder'] = multisensory_encoder

    # Decoders
    if 'rgb' in config.dec_modalities:
        config.models['rgb_decoder'] = ImageDecoder(3, config.state_size, config.activation,
                                                    out_activation='sigmoid')
    if 'depth' in config.dec_modalities:
        config.models['depth_decoder'] = ImageDecoder(1, config.state_size, config.activation,
                                                      out_activation='sigmoid')
    if 'joint_positions' in config.dec_modalities:
        config.models['joint_positions_decoder'] = VectorDecoder(config.state_size,
                                                                 7,
                                                                 config.vec_hidden_sizes,
                                                                 activation=config.activation)
    if 'gripper_joint_positions' in config.dec_modalities:
        config.models['gripper_joint_positions_decoder'] = VectorDecoder(config.state_size,
                                                                         2,
                                                                         config.vec_hidden_sizes,
                                                                         activation=config.activation)

    # Dynamics
    config.models['dynamics_model'] = RSSM(config.belief_size, config.state_size,
                                           config.action_size, config.hidden_size,
                                           config.embedding_size, config.activation,
                                           config.min_std_dev)
    
    # Goal classifier input is start (latent) state of demo and current state
    config.models['goal_classifier'] = MulticlassClassifier(2 * config.state_size,
                                                            config.n_goals,
                                                            config.gc_hidden_sizes,
                                                            activation=config.activation)

    
def _process_learning_params(config, params):
    config.seed = params.get('seed', 1)
    config.lr = params.get('lr', 1e-3)
    config.lr_step = params.get('lr_step', 25)
    config.adam_epsilon = params.get('adam_epsilon', 1e-3)
    config.free_nats = params.get('free_nats', 3)
    config.grad_clip_norm = params.get('grad_clip_norm', 10)
    config.batch_size = params.get('batch_size', 50)
    config.n_epochs_per_episode = params.get('n_epochs_per_episode', 100)
    config.checkpoint_interval = params.get('checkpoint_interval', 5)
    config.global_kl_beta = params.get('global_kl_beta', 0)
    

def _process_data_params(config, params):
    config.obs_modalities = params.get('obs_modalities', ['rgb'])
    config.act_modalities = params.get('act_modalities', ['delta_joint_positions'])
    config.dec_modalities = params.get('dec_modalities', ['rgb'])
    config.chunk_size = params.get('chunk_size', 25)
    config.time_subsample = params.get('time_subsample', 1)
    config.n_goals = params.get('n_goals', 4) # TODO infer from data?
