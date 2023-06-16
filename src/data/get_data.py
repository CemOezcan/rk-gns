import os
from typing import Tuple

from tfrecord.torch import TFRecordDataset

from torch_geometric.data import DataLoader

from src.data.graph_loader import GraphDataLoader
from src.data.preprocessing import Preprocessing
from src.data.trapez_preprocessing import TrapezPreprocessing
from src.util.util import device
from src.util.types import *

from src.util.types import ConfigDict
from src.util.util import get_from_nested_dict
from os.path import dirname as up

# TODO: Find a solution for this (incompatible with job_flag.sh scripts where CONFIG_NAME is supposed to be set)
ROOT_DIR = up(up(up(os.path.join(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

OUT_DIR = None


def get_directories(dataset_name):
    task_dir = os.path.join(DATA_DIR, dataset_name)
    out_dir = os.path.join(task_dir, 'output')
    in_dir = os.path.join(task_dir, 'input')

    return in_dir, out_dir


def get_data(config: ConfigDict, split='train', split_and_preprocess=True, add_targets=True, raw=False):
    dataset_name = get_from_nested_dict(config, list_of_keys=["task", "dataset"], raise_error=True)
    in_dir, _ = get_directories(dataset_name)

    if dataset_name == 'trapez' or dataset_name == 'deformable_plate':
        batch_size = get_from_nested_dict(config, list_of_keys=["task", "batch_size"], raise_error=True)
        pp = TrapezPreprocessing(split, ROOT_DIR, raw)
        train_data_list = pp.build_dataset_for_split()
        # TODO: shuffle
        trainloader = train_data_list if raw else DataLoader(train_data_list, shuffle=True, batch_size=1)
        return trainloader
    else:
        raise NotImplementedError("Implement your data loading here!")
