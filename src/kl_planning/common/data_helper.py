import torch
from copy import deepcopy

from multisensory_learning.util import file_util, data_util
from multisensory_learning.common.modalities import ModalityType


class DataHelper:
    """
    Stores information about dataset modalities, data types, shapes, etc. Also
    offers some utility functions like performing data-specific input/output scaling.
    """

    def __init__(self, filename=""):
        if filename:
            self.load_from_yaml(filename)
        else:
            self._data = {}

    def add_observation_modality(self, modality, modality_type, shape=None,
                                 abbreviation=None, h5_key=None):
        if 'obs_modalities' not in self._data:
            self._data['obs_modalities'] = []
        if modality not in self._data['obs_modalities']:
            self._data['obs_modalities'].append(modality)
            self._add_modality(modality, modality_type, shape, abbreviation, h5_key)
            
    def add_action_modality(self, modality, modality_type, shape, abbreviation=None, h5_key=None):
        if 'act_modalities' not in self._data:
            self._data['act_modalities'] = []
        if modality not in self._data['act_modalities']:
            self._data['act_modalities'].append(modality)
            self._add_modality(modality, modality_type, shape, abbreviation, h5_key)

    def add_decode_modality(self, modality, modality_type, shape, abbreviation=None, h5_key=None):
        if 'dec_modalities' not in self._data:
            self._data['dec_modalities'] = []
        if modality not in self._data['dec_modalities']:
            self._data['dec_modalities'].append(modality)
            self._add_modality(modality, modality_type, shape, abbreviation, h5_key)

    def add_state_modality(self, modality, modality_type, shape, abbreviation=None, h5_key=None):
        if 'state_modalities' not in self._data:
            self._data['state_modalities'] = []
        if modality not in self._data['state_modalities']:
            self._data['state_modalities'].append(modality)
            self._add_modality(modality, modality_type, shape, abbreviation, h5_key)

    def _add_modality(self, modality, modality_type, shape=None, abbreviation=None, h5_key=None):
        if 'modality_types' not in self._data:
            self._data['modality_types'] = {}
        if 'modality_shapes' not in self._data:
            self._data['modality_shapes'] = {}
        if 'modality_abbreviations' not in self._data:
            self._data['modality_abbreviations'] = {}
        if 'h5_keys' not in self._data:
            self._data['h5_keys'] = {}
        self._data['modality_types'][modality] = modality_type
        self._data['modality_shapes'][modality] = shape
        self._data['modality_abbreviations'][modality] = abbreviation if abbreviation else modality
        self._data['h5_keys'][modality] = h5_key if h5_key else modality
        
    def set_dataset_type(self, dataset_type):
        self._data['dataset_type'] = dataset_type

    def set_h5_root(self, h5_root):
        self._data['h5_root'] = h5_root

    def set_lmdb_root(self, lmdb_root):
        self._data['lmdb_root'] = lmdb_root

    def set_time_subsample(self, time_subsample):
        self._data['time_subsample'] = time_subsample

    def set_tasks(self, tasks):
        self._data['tasks'] = tasks

    def set_train_filenames(self, train_filenames):
        self._data['train_filenames'] = train_filenames
    
    def set_validation_filenames(self, val_filenames):
        self._data['val_filenames'] = val_filenames

    def set_num_train_samples(self, num_train_samples):
        self._data['num_train_samples'] = num_train_samples

    def set_num_validation_samples(self, num_val_samples):
        self._data['num_val_samples'] = num_val_samples

    def set_num_instance_seg_masks(self, num_seg_masks):
        self._data['num_instance_seg_masks'] = num_seg_masks

    def set_num_semantic_seg_masks(self, num_seg_masks):
        self._data['num_semantic_seg_masks'] = num_seg_masks

    def set_data_statistics(self, stats):
        self._data['data_statistics'] = stats

    def set_forecast_steps(self, steps):
        self._data['forecast_steps'] = steps

    def set_image_size(self, size):
        self._data['image_size'] = size

    def set_h5_key(self, modality, key):
        self._data['h5_keys'][modality] = key

    def get_modalities(self):
        return (self.get_observation_modalities() +
                self.get_action_modalities() +
                self.get_decode_modalities() +
                self.get_state_modalities())
            
    def get_observation_modalities(self):
        return self._data['obs_modalities'] if 'obs_modalities' in self._data else []

    def get_action_modalities(self):
        return self._data['act_modalities'] if 'act_modalities' in self._data else []

    def get_decode_modalities(self):
        return self._data['dec_modalities'] if 'dec_modalities' in self._data else []

    def get_state_modalities(self):
        return self._data['state_modalities'] if 'state_modalities' in self._data else []

    def get_image_modalities(self):
        return self.get_rgb_modalities() + self.get_depth_modalities()

    def get_rgb_modalities(self):
        return self._get_modalities_of_type(ModalityType.RGB)

    def get_depth_modalities(self):
        return self._get_modalities_of_type(ModalityType.DEPTH)

    def get_instance_segmentation_modalities(self):
        return self._get_modalities_of_type(ModalityType.INSTANCE_SEGMENTATION)

    def get_semantic_segmentation_modalities(self):
        return self._get_modalities_of_type(ModalityType.SEMANTIC_SEGMENTATION)

    def get_segmentation_modalities(self):
        return (self.get_instance_segmentation_modalities() +
                self.get_semantic_segmentation_modalities())
    
    def get_vector_modalities(self):
        return self._get_modalities_of_type(ModalityType.VECTOR)

    def get_orientation_modalities(self):
        return self._get_modalities_of_type(ModalityType.ORIENTATION)

    def _get_modalities_of_type(self, modality_type):
        return [k for k, v in self._data['modality_types'].items() if v == modality_type]
    
    def get_tasks(self):
        return self._data['tasks']

    def get_modality_abbreviation(self, modality):
        return self._data['modality_abbreviations'][modality]

    def get_modality_shape(self, modality):
        return self._data['modality_shapes'][modality]

    def get_modality_type(self, modality):
        return self._data['modality_types'][modality]

    def get_modality_min(self, modality):
        return self._data['data_statistics'][modality]['min']

    def get_modality_max(self, modality):
        return self._data['data_statistics'][modality]['max']

    def get_modality_model_min(self, modality):
        return self._data['data_statistics'][modality]['model_min']

    def get_modality_model_max(self, modality):
        return self._data['data_statistics'][modality]['model_max']

    def get_modality_decoder_loss(self, modality):
        modality_type = self.get_modality_type(modality)
        if modality_type == ModalityType.VECTOR:
            return torch.nn.MSELoss()
        elif modality_type in [ModalityType.SEMANTIC_SEGMENTATION,
                               ModalityType.INSTANCE_SEGMENTATION]:
            return torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Decoder loss function not known for modality: {modality}")

    def get_modality_delta_sum(self, modality):
        modality_type = self.get_modality_type(modality)
        if modality_type in [ModalityType.SEMANTIC_SEGMENTATION,
                             ModalityType.INSTANCE_SEGMENTATION]:


            # TODO need to implement delta sum for k channel masks


            
            pass
        else:
            raise ValueError(f"Delta sum function not known for modality: {modality}")
          
    
    def get_dataset_type(self):
        return self._data['dataset_type']

    def get_h5_root(self):
        return self._data['h5_root']
    
    def get_h5_key(self, modality):
        return self._data['h5_keys'][modality]

    def get_lmdb_root(self):
        return self._data['lmdb_root']

    def get_time_subsample(self):
        return self._data['time_subsample']

    def get_forecast_steps(self):
        return self._data['forecast_steps']

    def get_image_size(self):
        return self._data['image_size']

    def get_num_semantic_seg_masks(self):
        return self._data['num_semantic_seg_masks']

    def get_num_instance_seg_masks(self):
        return self._data['num_instance_seg_masks']
    
    def scale_data_in(self, data, modality):
        if self.has_scaling(modality):
            return data_util.scale_min_max(data,
                                           self.get_modality_min(modality),
                                           self.get_modality_max(modality),
                                           self.get_modality_model_min(modality),
                                           self.get_modality_model_max(modality))
        else:
            # Pass-through if no scaling needs done (e.g. depth data already in [0,1])
            return data

    def scale_data_out(self, data, modality):
        if self.has_scaling(modality):
            return data_util.scale_min_max(data,
                                           self.get_modality_model_min(modality),
                                           self.get_modality_model_max(modality),
                                           self.get_modality_min(modality),
                                           self.get_modality_max(modality))
        else:
            # Pass-through if no scaling was done on input
            print("NO SCALE OUT", modality)
            return data

    def has_scaling(self, modality):
        return (modality in self._data['data_statistics'] and
                'min' in self._data['data_statistics'][modality] and
                'max' in self._data['data_statistics'][modality] and
                'model_min' in self._data['data_statistics'][modality] and
                'model_max' in self._data['data_statistics'][modality])
    
    def save_to_yaml(self, filename):
        save_data = deepcopy(self._data)
        for modality, modality_type in self._data['modality_types'].items():
            save_data['modality_types'][modality] = modality_type.name
        file_util.save_yaml(save_data, filename)
        
    def load_from_yaml(self, filename):
        self._data = file_util.load_yaml(filename)
        for modality, modality_type in self._data['modality_types'].items():
            self._data['modality_types'][modality] = ModalityType[modality_type]
