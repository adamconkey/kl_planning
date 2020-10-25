import sys
from enum import Enum


class ModalityType(Enum):
    RGB = 1
    DEPTH = 2
    VECTOR = 3
    ORIENTATION = 4
    INSTANCE_SEGMENTATION = 5
    SEMANTIC_SEGMENTATION = 6

    @staticmethod
    def get_types():
        return ModalityType._member_names_

    @staticmethod
    def is_valid_type(str_name):
        return str_name.upper() in ModalityType.get_types()


class DatasetModalities:

    def __init__(self):
        self._types = {}
        self._abbreviations = {}
        
    def get_type(self, modality):
        if modality not in self._types:
            raise ValueError(f"Unknown type for modality: {modality}")
        return self._types[modality]

    def get_abbreviation(self, modality):
        if modality not in self._abbreviations:
            raise ValueError(f"Unknown abbreviation for modality: {modality}")
        return self._abbreviations[modality]
    

class IsaacModalities(DatasetModalities):

    def __init__(self):
        super().__init__()

        self._types = {
            'joint_positions': ModalityType.VECTOR,
            'joint_velocities': ModalityType.VECTOR,
            'delta_joint_positions': ModalityType.VECTOR,
            'gripper_joint_positions': ModalityType.VECTOR,
            'gripper_joint_velocities': ModalityType.VECTOR,
            'delta_gripper_joint_positions': ModalityType.VECTOR,
            'rgb': ModalityType.RGB,
            'depth': ModalityType.DEPTH,
            'instance_segmentation': ModalityType.INSTANCE_SEGMENTATION,
            'semantic_segmentation': ModalityType.SEMANTIC_SEGMENTATION
        }
        
        self._abbreviations = {
            'joint_positions': 'jp',
            'joint_velocities': 'jv',
            'delta_joint_positions': 'djp',
            'gripper_joint_positions': 'gjp',
            'gripper_joint_velocities': 'gjv',
            'delta_gripper_joint_positions': 'dgjp',
            'rgb': 'rgb',
            'depth': 'dep',
            'instance_segmentation': 'iseg',
            'semantic_segmentation': 'sseg'
        }        
        

def get_dataset_modalities(dataset_name):
    if dataset_name == 'isaac':
        return IsaacModalities()
    else:
        ui_util.print_error(f"\nCould not resolve dataset type: {dataset_name}\n")
        sys.exit(1)
