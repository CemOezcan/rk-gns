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
    model_name = config.get('task').get('model').lower()

    if poisson:
        return PoissonModel(config.get('model'), recurrence=task == 'lstm')
    elif task == 'alternating':
        return trapez_copy.TrapezModel(config.get('model'), recurrence=task == 'lstm')

    if 'trapez' == model_name:
        return trapez.TrapezModel(config.get('model'), recurrence=task == 'lstm')
    elif 'poisson' == model_name:
        return PoissonModel(config.get('model'), recurrence=task == 'lstm')
    else:
        raise NotImplementedError('Implement your algorithms here!')
