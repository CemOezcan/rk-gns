"""
Utility class to select a system model based on a given config file
"""
from src.model.abstract_system_model import AbstractSystemModel
from src.model.poisson import PoissonModel
from src.model import trapez_copy, trapez
from src.util.types import *
from src.util.util import get_from_nested_dict


def get_model(config: ConfigDict, poisson=False) -> AbstractSystemModel:
    task = config.get('task').get('task').lower()
    model = config.get('task').get('model').lower()
    # TODO: Alternating
    if poisson:
        return PoissonModel(config)
    elif model == 'supervised':
        return trapez_copy.TrapezModel(config)

    if 'trapez' == task:
        return trapez.TrapezModel(config)
    elif 'poisson' == task:
        return PoissonModel(config)
    else:
        raise NotImplementedError('Implement your algorithms here!')
