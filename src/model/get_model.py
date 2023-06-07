"""
Utility class to select a system model based on a given config file
"""
from src.model.abstract_system_model import AbstractSystemModel
from src.model.flag import FlagModel
from src.model.trapez import TrapezModel
from src.util.types import *
from src.algorithms.abstract_simulator import AbstractSimulator
from src.util.util import get_from_nested_dict


def get_model(config: ConfigDict) -> AbstractSystemModel:
    model_name = get_from_nested_dict(config, list_of_keys=["task", "dataset"], raise_error=True).lower()
    if 'flag' in model_name:
        return FlagModel(config.get('model'))
    elif 'trapez' in model_name:
        return TrapezModel(config.get('model'))
    else:
        raise NotImplementedError("Implement your algorithms here!")
