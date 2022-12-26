import os
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QDialog
from pathlib import Path

from glob import glob


def root_path():
    return os.path.abspath(os.sep)


# Retorna a home
def get_user_home():
    return os.path.expanduser('~')

# Valida se um caminho é valido


def exists(path_to_file):
    return os.path.exists(path_to_file)


def get_current_folder():
    return os.path.dirname(os.path.realpath(__file__))

# Retorna a pasta raiz do projeto


def get_src_projec():

    parent_dir = get_current_folder().split('DOF_Python')[0]
    parent_dir = os.path.join(parent_dir, 'DOF_Python')

    return parent_dir


def get_images_from_path(path):
    return [os.path.abspath(os.path.join(path, name)) for name in os.listdir(path) if os.path.isfile(os.path.join(path, name)) and name.endswith(".png")]

# Retorna todos os subfolder de uma dada path


def get_all_subfolder(path):

    folders = [name for name in os.listdir(
        path) if os.path.isdir(os.path.join(path, name))]
    return folders


def folders_in(path):
    folders = [f.path for f in os.scandir(path) if f.is_dir()]
    result = []
    for f in folders:
        result.append(f.split(os.sep)[-1])

    return sorted(result)


def folders_given_a_path(path):
    return [f.path for f in os.scandir(path) if f.is_dir()]


def files_in_path(path):
    l = None
    print(path)
    try:
        l = [file for file in os.listdir(
            path) if os.path.isfile(os.path.join(path, file))]
    except Exception:
        l = []

    return l


def exists(path_to_file):
    return os.path.exists(path_to_file)
