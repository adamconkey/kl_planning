import sys
import numpy as np
import torch
from torch import nn

from kl_planning.models import VectorEncoder, VectorDecoder, ImageEncoder, get_activation


def encode_obs(obs, obs_modalities, models):
    """
    Encodes observations by first applying modality-specific encoders and then passing
    those encoded values through multisensory encoder to final latent state.

    Some encoders have auxiliary outputs (e.g. skip-adds for image u-nets), those are
    returned as well.

    If only one modality is being encoded, then the output of its encoder is treated
    as the latent state and it does NOT go through a multisensory encoder.
    """
    # Apply modality-specific encoders
    encoded = []
    aux_outputs = {} # Some models have auxiliary outputs (e.g. skip-adds for image u-nets)
    for modality in obs_modalities:
        encoder_output = models[f"{modality}_obs_encoder"](obs[modality])
        if isinstance(encoder_output, tuple):
            if len(encoder_output) != 2:
                raise ValueError("Encoders should have at most 2 returns (out, aux)")
            output, aux_output = encoder_output
            encoded.append(output)
            aux_outputs[modality] = aux_output
        else:
            encoded.append(encoded_output)
    # Get latent state
    if len(obs_modalities) > 1:
        multisensory_inputs = torch.cat(encoded, dim=-1)
        latent = models['multisensory_encoder'](multisensory_inputs)
    else:
        latent = encoded[0]
                
    return latent, aux_outputs

def encode_act(act, act_modalities, models):
    encoded = [models[f"{m}_act_encoder"](act[m]) for m in act_modalities]    
    encoded = torch.cat(encoded, dim=-1) if len(act_modalities) > 1 else encoded[0]
    return encoded


def decode_latent(latent, decode_modalities, models, data_helper, aux_outputs={},
                  process_for_visualization=False):
    decoded = {}
    aux_decodes = {}
    for modality in decode_modalities:
        decoder = models[f'{modality}_decoder']
        if modality in aux_outputs:
            out, aux = decoder(latent, aux_outputs[modality])
        else:
            out, aux = decoder(latent)
        decoded[modality] = out
        aux_decodes[modality] = aux
        if process_for_visualization:
            if modality in data_helper.get_segmentation_modalities():
                softmax2d = torch.nn.Softmax2d()
                decoded[modality] = softmax2d(decoded[modality]).cpu().detach().numpy().squeeze()
                decoded[modality] = np.rint(decoded[modality])
    return decoded, aux_decodes


def predict_latent(latent, act, models, state=None):
    """
    act is potentially concatenated encoded actions
    """
    # Apply action with forward model to get predicted next latent state
    latent_next = models['forward_dynamics'](latent, act, state)
    return latent_next


def convert_to_float_tensors(tensor_dict, keys=[]):
    keys = keys if keys else tensor_dict.keys()
    for k in keys:
        if torch.is_tensor(tensor_dict[k]):
            tensor_dict[k] = tensor_dict[k].float()
        else:
            tensor_dict[k] = torch.FloatTensor(tensor_dict[k])


def convert_to_long_tensors(tensor_dict, keys=[]):
    keys = keys if keys else tensor_dict.keys()
    for k in keys:
        if torch.is_tensor(tensor_dict[k]):
            tensor_dict[k] = tensor_dict[k].long()
        else:
            tensor_dict[k] = torch.LongTensor(tensor_dict[k])

            
def move_batch_to_device(batch_dict, device):
    """
    Recursive function that moves a (nested) dictionary of tensors to the specified device.
    """
    for k, v in batch_dict.items():
        if isinstance(v, torch.Tensor):
            batch_dict[k] = v.to(device)
        elif isinstance(v, dict):
            move_batch_to_device(v, device)
        else:
            raise ValueError(f"Unknown type for moving to device: {type(v)}")

        
def move_models_to_device(models_dict, device):
    """
    Assuming flat dictionary where values are all type torch.nn.Module.
    """
    for k, v in models_dict.items():
        models_dict[k] = v.to(device)

        
def get_tensors_at_time(data_dict, idx):
    """
    Returns dictionary of Tensors selected at specified temporal index.

    Note: Assume first dim is batch and second dim is time.
    """
    return {k: v[:,idx] for k, v in data_dict.items()}


def get_decoder(modality, data_helper, latent_size, activation):
    # TODO I think it would be good to have a model_helper like the data_helper from
    # which you can retrieve the desired model configuration parameters to individually
    # configure each model. This could be used for all model creation.
    if modality in data_helper.get_vector_modalities():
        modality_size = data_helper.get_modality_shape(modality)[0]
        decoder = VectorDecoder(latent_size, modality_size)
    elif modality in data_helper.get_instance_segmentation_modalities():
        decoder = SegmentationDecoder(data_helper.get_num_instance_seg_masks(),
                                      activation=activation)
    elif modality in data_helper.get_semantic_segmentation_modalities():
        decoder = SegmentationDecoder(data_helper.get_num_semantic_seg_masks(),
                                      activation=activation)
    else:
        ui_util.print_error(f"\nUnknown modality for decoders: {modality}")
        sys.exit(1)
    return decoder


def get_combined_act(act, act_modalities, models, encode_act):
    if encode_act:
        combined = encode_act(act, act_modalities, models)
    else:
        combined = torch.cat([act[k] for k in act_modalities], dim=-1)
    return combined


def add_observation_encoders(obs_modalities, models, data_helper):
    for modality in obs_modalities:
        if modality in data_helper.get_image_modalities():
            if modality in data_helper.get_depth_modalities():
                encoder = ImageEncoder(1)
            elif modality in data_helper.get_rgb_modalities():
                encoder = ImageEncoder(3) 
            # Need to infer output size by passing mock input
            input_shape = [1] + data_helper.get_modality_shape(modality) # Add batch
            test_input = torch.empty(input_shape)
            test_input = np.transpose(test_input, (0, 3, 1, 2)) # (t, c, h, w)
            encoder_output, _ = encoder(test_input)
            encoder.output_size = encoder_output.shape[-1]
        elif modality in data_helper.get_vector_modalities():
            encoder = VectorEncoder(data_helper.get_modality_shape(modality)[0])
        else:
            ui_util.print_error(f"\nUnknown modality for observation encoders: {modality}")
            sys.exit(1)
        models[f"{modality}_obs_encoder"] = encoder
    multisensory_input_size = sum([e.output_size for k, e in models.items() if "_obs_encoder" in k])
    return multisensory_input_size


def add_action_encoders(act_modalities, models, data_helper, encode_act):
    action_size = 0
    for modality in act_modalities:
        # Assuming act modalities are always vectors
        modality_len = data_helper.get_modality_shape(modality)[0]
        if encode_act:
            if modality in data_helper.get_vector_modalities():
                encoder = VectorEncoder(modality_len)
            else:
                ui_util.print_error(f"\nUnknown modality for action encoders: {modality}")
                sys.exit(1)
            self.models[f"{modality}_act_encoder"] = encoder
            action_size += encoder.output_size
        else:
            action_size += modality_len
    return action_size


def add_state_encoders(state_modalities, models, data_helper):
    """
    TODO not yet doing encoding so models isn't used but need to add this option
    """
    state_size = 0
    for modality in state_modalities:
        # Assuming state modalities are always vectors, have to handle differently if not
        state_size += data_helper.get_modality_shape(modality)[0]
    return state_size


def add_decoders(decode_modalities, models, data_helper, latent_size, activation):
    for modality in decode_modalities:
        decoder = get_decoder(modality, data_helper, latent_size, activation)
        models[f'{modality}_decoder'] = decoder
