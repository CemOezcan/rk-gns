"""
Utility class to select a system model based on a given config file
"""
from src.model.abstract_system_model import AbstractSystemModel
from src.model.poisson import PoissonModel
from src.model.trapez import TrapezModel
from src.util.types import *
from src.util.util import get_from_nested_dict


def get_model(config: ConfigDict, poison=False) -> AbstractSystemModel:
    model_name = get_from_nested_dict(config, list_of_keys=['task', 'dataset'], raise_error=True).lower()
    rec = config.get('task').get('task') == 'lstm'
    if poison:
        return PoissonModel(config.get('model'))
    if 'trapez' in model_name or 'plate' in model_name:
        return TrapezModel(config.get('model'), recurrence=rec)
    else:
        raise NotImplementedError('Implement your algorithms here!')
