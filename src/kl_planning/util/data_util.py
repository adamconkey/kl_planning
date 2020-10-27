import os
import sys
import h5py
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from pyquaternion import Quaternion

from kl_planning.util import ui_util
from kl_planning.common.modalities import ModalityType, get_dataset_modalities

# Utility functions for working with data (scaling, data representations
# (e.g. rotations), error checking on dataset types, etc.


def scale_min_max(x, x_min, x_max, desired_min, desired_max):
    """
    Applies min/max scaling on one data instance.

    Args:
        x (ndarray): Data to be scaled.
        x_min (flt): Minimum value of data (over full dataset).
        x_max (flt): Maximum value of data (over full dataset).
        desired_min (flt): Desired minimum value.
        desired_max (flt): Desired maximum value.
    """
    return ((desired_max - desired_min) * (x - x_min) / (x_max - x_min)) + desired_min


def compute_h5_statistics(h5_root, data_helper, buffer_pct=0):
    task_names = os.listdir(h5_root)
    filenames = [glob(os.path.join(h5_root, task_name, '*')) for task_name in task_names]
    filenames = [f for l in filenames for f in l]
        
    print(f"\nComputing statistics for H5 files in directory: {h5_root}")

    stats = {}
    # TODO these are manually set because they are known, want a more general
    # way to handle these since this is specific to isaac data
    exclusions = ['joint_positions', 'rgb', 'instance_segmentation', 'semantic_segmentation']
    modalities = data_helper.get_modalities()
    if 'joint_positions' in modalities:
        stats['joint_positions'] = {'min': -np.pi, 'max': np.pi, 'model_min': -1., 'model_max': 1.}
    if 'rgb' in modalities:
        stats['rgb'] = {'min': 0., 'max': 255., 'model_min': 0., 'model_max': 1.}
    if 'depth' in modalities:
        # Still want to compute min/max for depth but want in [0,1] instead of [-1,1]
        stats['depth'] = {'min': sys.maxsize, 'max': -sys.maxsize, 'model_min': 0., 'model_max': 1.}

    for filename in tqdm(filenames, file=sys.stdout):
        with h5py.File(filename, 'r') as h5_file:
            for modality in [m for m in modalities if m not in exclusions]:
                if modality.startswith('delta_'):
                    nominal_modality = modality.replace('delta_', '')
                    nominal_data = np.array(h5_file[nominal_modality])
                    modality_data = nominal_data[1:] - nominal_data[:-1]
                else:
                    modality_data = h5_file[data_helper.get_h5_key(modality)]
                maybe_min = float(np.min(modality_data))
                maybe_max = float(np.max(modality_data))
                if modality in stats:
                    stats[modality]['min'] = min(stats[modality]['min'], maybe_min)
                    stats[modality]['max'] = max(stats[modality]['max'], maybe_max)
                else:
                    stats[modality] = {'min': maybe_min, 'max': maybe_max,
                                       'model_min': -1., 'model_max': 1.}
    # Can add a small buffer to the min/max scaling if you think your data distribution
    # is not fully captured by your samples
    if buffer_pct > 0:
        for modality, data in stats.items():
            if modality in exclusions:
                continue
            # Enforce pos/neg symmetry if min is negative and max is positive
            if data['min'] * data['max'] < 0:
                max_bound = max(abs(data['min']), abs(data['max']))
                data['min'] = -max_bound
                data['max'] = max_bound
            # Add a small buffer
            data['min'] = (buffer_pct * data['min']) + data['min']
            data['max'] = (buffer_pct * data['max']) + data['max']

    data_helper.set_data_statistics(stats)

    
def populate_data_helper_modalities(filename, data_helper, dataset_modalities, obs_modalities,
                                    act_modalities, state_modalities, dec_modalities):
    """
    The data helper manages all the information about modalities in use like shapes, type,
    h5 keys, etc. This function populates that information using one H5 file representative
    of all of them.
    """
    with h5py.File(filename, 'r') as h5_data:
        for modality in obs_modalities:
            if modality == 'object_positions':
                for obj_id in h5_data['objects'].keys():
                    h5_key = f'tf/{obj_id}/position'
                    nominal_data = h5_data[h5_key]
                    data_helper.add_observation_modality(f'{obj_id}_position',
                                                         ModalityType.VECTOR,
                                                         list(h5_data[h5_key].shape[1:]),
                                                         'objp',
                                                         h5_key)
            else:
                data_helper.add_observation_modality(modality,
                                                     dataset_modalities.get_type(modality),
                                                     list(h5_data[modality].shape[1:]),
                                                     dataset_modalities.get_abbreviation(modality))
        for modality in dec_modalities:
            modality_type = dataset_modalities.get_type(modality)
            if modality_type == ModalityType.INSTANCE_SEGMENTATION:
                n_masks = len(h5_data[modality].attrs['ids']) + 1 # Adding for background
                data_helper.set_num_instance_seg_masks(n_masks)
            elif modality_type == ModalityType.SEMANTIC_SEGMENTATION:
                n_masks = len(h5_data[modality].attrs['ids']) + 1 # Adding for background
                data_helper.set_num_semantic_seg_masks(n_masks)
            data_helper.add_decode_modality(modality,
                                            modality_type,
                                            list(h5_data[modality].shape[1:]),
                                            dataset_modalities.get_abbreviation(modality))

        for modality in act_modalities:            
            h5_key = modality.replace('delta_', '') if modality.startswith('delta_') else modality
            # Not the best way of doing this, but assuming delta always follows the actual
            # modality, so if the nominal modality is not yet there it still gets a key
            data_helper.set_h5_key(h5_key, h5_key)
            data_helper.add_action_modality(modality,
                                            dataset_modalities.get_type(modality),
                                            list(h5_data[h5_key].shape[1:]),
                                            dataset_modalities.get_abbreviation(modality))

        for modality in state_modalities:
            data_helper.add_state_modality(modality,
                                           dataset_modalities.get_type(modality),
                                           list(h5_data[modality].shape[1:]),
                                           dataset_modalities.get_abbreviation(modality))

            
def resize_images(imgs, shape):
    """
    Assuming imgs is over time (i.e. shape (t, h, w, c) or (t, h, w)), and shape
    is desired shape of either (h, w, c) or (h, w).
    """
    resized = np.zeros((len(imgs), *shape), dtype=imgs.dtype)
    for i in range(len(imgs)):
        resized[i] = cv2.resize(imgs[i], shape[:2], interpolation=cv2.INTER_NEAREST)
    return resized


def process_rgb_in(data, img_size):
    """
    Process numpy array data (e.g. from recorded H5 files) for input to
    network (e.g. on visualizing data after training)
    """
    if data.shape[1] != img_size:
        data = resize_images(data, (img_size, img_size, 3))
    data = np.transpose(data, (0, 3, 1, 2)) # Want (t, c, h, w)
    return data


def process_depth_in(data, img_size):
    data = np.expand_dims(data, -1) # Add channels dim
    if data.shape[1] != img_size:
        data = resize_images(data, (img_size, img_size, 1))
    data = np.transpose(data, (0, 3, 1, 2)) # Want (t, c, h, w)
    return data


def process_time_batch(f, *tensors):
    """
    Applies a function to time-distributed inputs in a batch by "superbatching"
    them, i.e. taking input tensors of shape (time, batch, *in_channels) and input
    to function as (time * batch, *in_channels), then puts the output back in the
    form (time, batch, *out_channels). Useful for models like encoder/decoder/reward
    because they are not dependent on other timesteps (unlike transition model).
    """
    reshaped_inputs = []
    for x in tensors:
        x_size = x.size()
        time = x_size[0]
        batch = x_size[1]
        superbatch_size = time * batch
        reshaped_inputs.append(x.reshape(superbatch_size, *x_size[2:]))
    y = f(*reshaped_inputs)
    y_size = y.size()
    y = y.view(time, batch, *y_size[1:])
    return y


def get_pose_from_h5(h5_file, base_link, end_link, start_idx=0, end_idx=None, subsample=1):
    """
    Computes pose of end link in base link frame. This is necessary since TF poses are all
    relative to parent, so if e.g. desire EE pose in world frame, need to apply all intermediate
    transforms to get that pose.
    """
    h5_tfs = h5_file['tf']
    if end_link not in h5_tfs:
        print(f"Link not found: {end_link}")
        return None, None

    end_idx = len(next(iter(h5_tfs.values()))['position']) if end_idx is None else end_idx
    idxs = range(start_idx, end_idx, subsample)
    
    poses = []
    for idx in idxs:
        # Work backwards through chain to get intermediate transforms
        current_link = end_link
        tf = np.eye(4)
        while current_link != base_link:
            pos = np.array(h5_tfs[current_link]['position'][idx])
            quat = np.array(h5_tfs[current_link]['orientation'][idx])
            tf = np.dot(pose_to_homogeneous(pos, quat), tf)
            current_link = h5_tfs[current_link].attrs['parent_frame']
        pos = pos_from_homogeneous(tf)
        quat = quat_from_homogeneous(tf)
        poses.append((pos, quat))

    pos = np.stack([p[0] for p in poses])
    quat = np.stack([p[1] for p in poses])
    return pos, quat
