"""
Utility class to select a task based on a given config file
"""
from src.algorithms.mesh_task import MeshTask
from src.util.types import *
from src.util.util import get_from_nested_dict
from src.algorithms.abstract_task import AbstractTask


def get_task(config: ConfigDict) -> AbstractTask:
    return MeshTask(config=config)
