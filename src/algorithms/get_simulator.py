"""
Utility class to select an algorithm based on a given config file
"""
from src.algorithms.mesh_simulator import MeshSimulator
from src.util.types import *
from src.algorithms.abstract_simulator import AbstractSimulator
from src.util.util import get_from_nested_dict


def get_simulator(config: ConfigDict) -> AbstractSimulator:
    task = get_from_nested_dict(config, list_of_keys=["task", "task"], raise_error=True)

    if task == "mesh":
        return MeshSimulator(config=config)
    else:
        raise NotImplementedError("Implement your tasks here!")