import copy
import re

from src.algorithms.get_task import get_task
from src.util.util import device, read_yaml
import random
import sys
import warnings
from multiprocessing import set_start_method

import numpy as np
import torch

warnings.filterwarnings('ignore')


def main(config_name='trapez'):
    params = read_yaml(config_name)

    print(f'Device used for this run: {device}')
    random_seed = params.get('random_seed')
    random.seed(random_seed)
    np.random.seed(seed=random_seed)
    torch.manual_seed(seed=random_seed)
    torch.cuda.manual_seed_all(seed=random_seed)
    torch.cuda.manual_seed(seed=random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    global_model = None
    if params.get('task').get('model').lower() == 'supervised':
        params_copy = copy.deepcopy(params)
        params_copy['task']['task'] = 'poisson'
        params_copy['task']['model'] = 'mgn'
        params_copy['task']['learning_rate'] = params_copy['poisson']['learning_rate']
        params_copy['task']['weight_decay'] = params_copy['poisson']['weight_decay']
        params_copy['task']['epochs'] = params_copy['poisson']['epochs']
        # TODO get best.
        task = get_task(params_copy)
        task.run_iterations()
        #task.get_scalars()
        task.finish()
        global_model = task.get_model()

    task = get_task(params)
    task.set_model(global_model)
    task.run_iterations()

    random.seed(random_seed)
    np.random.seed(seed=random_seed)
    torch.manual_seed(seed=random_seed)
    torch.cuda.manual_seed_all(seed=random_seed)
    torch.cuda.manual_seed(seed=random_seed)
    task.get_scalars()


if __name__ == '__main__':
    try:
        args = sys.argv[1]
        main(args)
    except IndexError:
        main()
