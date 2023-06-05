import os
from tfrecord.torch import TFRecordDataset

from src.data.graph_loader import GraphDataLoader
from src.data.preprocessing import Preprocessing
from src.util.util import read_yaml

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

def get_data(config: ConfigDict, split='train', split_and_preprocess=True, add_targets=True):
    dataset_name = get_from_nested_dict(config, list_of_keys=["task", "dataset"], raise_error=True)
    in_dir, _ = get_directories(dataset_name)

    if dataset_name == 'flag_minimal' or dataset_name == 'flag_minimal':
        pp = Preprocessing(config, split, split_and_preprocess, add_targets, in_dir=in_dir)
        tfrecord_path = os.path.join(in_dir, split + ".tfrecord")
        index_path = os.path.join(in_dir, split + ".idx")
        tf_dataset = TFRecordDataset(tfrecord_path, index_path, None, transform=pp.preprocess)
        return GraphDataLoader(tf_dataset)
    elif dataset_name == 'deforming_plate':
        pp = Preprocessing(config, split, split_and_preprocess, add_targets, in_dir=in_dir)
        tfrecord_path = os.path.join(in_dir, split + ".tfrecord")
        index_path = os.path.join(in_dir, split + ".idx")
        tf_dataset = TFRecordDataset(tfrecord_path, index_path, None, transform=pp.preprocess)
        return GraphDataLoader(tf_dataset)
    elif dataset_name == 'cylinder_flow':
        pp = Preprocessing(config, split, split_and_preprocess, add_targets, in_dir=in_dir)
        tfrecord_path = os.path.join(in_dir, split + ".tfrecord")
        index_path = os.path.join(in_dir, split + ".idx")
        tf_dataset = TFRecordDataset(tfrecord_path, index_path, None, transform=pp.preprocess)
        return GraphDataLoader(tf_dataset)
    else:
        raise NotImplementedError("Implement your data loading here!")