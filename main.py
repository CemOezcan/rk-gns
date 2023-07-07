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
    config_file = read_yaml(config_name)

    params = config_file['params']
    print(f'Device used for this run: {device}')
    random_seed = params.get('random_seed')
    random.seed(random_seed)
    np.random.seed(seed=random_seed)
    torch.manual_seed(seed=random_seed)

    task = get_task(params)
    task.run_iterations()

    task.get_scalars()


if __name__ == '__main__':
    try:
        args = sys.argv[1]
        main(args)
    except IndexError:
        main()
