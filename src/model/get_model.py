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
    recurrence = config.get('task').get('recurrence')
    # TODO: Alternating
    if poisson:
        return PoissonModel(config, recurrence=recurrence)
    elif model == 'supervised':
        return trapez_copy.TrapezModel(config, recurrence=recurrence)

    if 'trapez' == task:
        return trapez.TrapezModel(config, recurrence=recurrence)
    elif 'poisson' == task:
        return PoissonModel(config, recurrence=recurrence)
    else:
        raise NotImplementedError('Implement your algorithms here!')
