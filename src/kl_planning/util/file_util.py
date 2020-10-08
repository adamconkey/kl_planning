import os
import sys
import yaml
import pickle
import shutil
from glob import glob

from kl_planning.util import ui_util


# Utility functions for working with files/directories and loading/saving
# data from various file formats (yaml, pickle).


def list_dir(directory):
    """
    Returns a list of files in directory as absolute paths.

    Args:
        directory (str): Absolute path to list files for.
    Returns:
        files (lst): List of absolute paths to files in directory.
    """
    files = glob(os.path.join(directory, '*')) if os.path.exists else []
    return files


def clear_dir(directory):
    """
    Removes all files and sub-directories in directory.

    Args:
        directory (str): Absolute path to directory to clear.
    """
    for path in list_dir(directory):
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def safe_create_dir(directory):
    """
    Creates new directory, checks with user if they want to overwrite data
    if the directory already exists.

    Args:
        directory (str): Absolute path to directory to be created.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)
    elif len(os.listdir(directory)) > 0:
        ui_util.print_warning(f"\nDirectory already exists: {directory}")
        overwrite = ui_util.query_yes_no("Overwrite data in this directory?")
        if overwrite:
            clear_dir(directory)
        else:
            ui_util.print_info(f"\nExiting. Existing data is intact in {directory}.\n\n")
            sys.exit(0)


def check_path_exists(path, msg_prefix="Path"):
    """
    Check if a path (file or directory) exists. If it does not, do a sys exit.

    This is a utility function that is useful for checking that files exist
    when parsing filenames from command line args.

    Args:
        path (str): Absolute path to check if it exists.
        msg_prefix (str): Descriptive prefix about that path to print a more
                          helpful error message if it exits (e.g. set 
                          msg_prefix='Config file' to better inform user which
                          file does not exist)
    """
    if not os.path.exists(path):
        ui_util.print_error(f"\n{msg_prefix} does not exist: {path}\n")
        sys.exit(1)


def copy_file(src, dest):
    """
    Copies file to a new location.

    Args:
        src (str): Absolute path of file to be copied.
        dest (str): Absolute path where file will be copied to.
    """
    shutil.copyfile(src, dest)


def save_yaml(data, filename):
    """
    Saves data stored in a dictionary to a yaml file.

    Args:
        data (dict): Dictionary of data to be saved to yaml file.
        filename (str): Absolute path to yaml file that data will be saved to.
    """
    with open(filename, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_yaml(filename):
    """
    Loads data form a yaml file to a dictionary.

    Args:
        filename (str): Absolute path to yaml file to load data from.
    Returns:
        data (dict): Dictionary of data loaded from the yaml file.
    """
    check_path_exists(filename, "YAML file")
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def save_pickle(data, filename):
    """
    Saves data stored in a dictionary to a pickle file.

    Args:
        data (dict): Dictionary of data to be saved to pickle file.
        filename (str): Absolute path to pickle file that data will be saved to.
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filename):
    """
    Loads data form a pickle file to a dictionary.

    Args:
        filename (str): Absolute path to pickle file to load data from.
    Returns:
        data (dict): Dictionary of data loaded from the pickle file.
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
