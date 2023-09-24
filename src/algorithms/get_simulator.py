"""
Utility class to select an algorithm based on a given config file
"""
from src.algorithms.alternating_simulator import AlternatingSimulator
from src.algorithms.lstm_simulator import LSTMSimulator
from src.algorithms.mesh_simulator import MeshSimulator
from src.util.types import *
from src.algorithms.abstract_simulator import AbstractSimulator
from src.util.util import get_from_nested_dict


def get_simulator(config: ConfigDict) -> AbstractSimulator:
    model = get_from_nested_dict(config, list_of_keys=['task', 'model'], raise_error=True)

    # TODO: self-supervised
    if model == 'mgn':
        return MeshSimulator(config=config)
    elif model == 'lstm':
        return LSTMSimulator(config=config)
    elif model == 'supervised':
        return AlternatingSimulator(config=config)
    else:
        raise NotImplementedError("Implement your tasks here!")
