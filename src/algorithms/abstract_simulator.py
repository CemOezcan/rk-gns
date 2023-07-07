import os
import pickle
import torch
import wandb

import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

from typing import Optional, Dict, List, Tuple, Any, Union
from pandas import DataFrame
from torch import Tensor
from tqdm import tqdm
from abc import ABC, abstractmethod
from torch_geometric.loader import DataLoader

from src.data.get_data import get_directories
from src.model.get_model import get_model
from src.util.types import ConfigDict
from src.util.util import device


class AbstractSimulator(ABC):
    """
    Superclass for iterative algorithms
    """

    def __init__(self, config: ConfigDict) -> None:
        """
        Initializes the iterative algorithm.

        Parameters
        ----------
            config : ConfigDict
                A (potentially nested) dictionary containing the "params" section of the section in the .yaml file
                used by cw2 for the current run.

        """

        self._config = config
        self._network_config = config.get('model')
        self._dataset_name = config.get('task').get('dataset')
        _, self._out_dir = get_directories(self._dataset_name)
        self._wandb_mode = config.get('logging').get('wandb_mode')

        self._trajectories = config.get('task').get('trajectories')
        self._time_steps = config.get('task').get('n_timesteps')
        self._seq_len = config.get('task').get('sequence')
        self._prefetch_factor = config.get('task').get('prefetch_factor')

        self._batch_size = config.get('task').get('batch_size')
        self._network = None
        self._optimizer = None
        self._wandb_run = None
        self._wandb_url = None
        self._initialized = False

        self.loss_function = F.mse_loss
        self._learning_rate = self._network_config.get('learning_rate')
        self._gamma = self._network_config.get('gamma')

    def initialize(self, task_information: ConfigDict) -> None:
        """
        Due to the interplay between the algorithm and the task, it sometimes makes sense for the task to provide
        additional initial information to the algorithm. This information may e.g., be the dimensionality of the task,
        the kind of training regime to perform etc.

        Parameters
        ----------
            task_information : ConfigDict
                A dictionary containing information on how to execute the algorithm on the current task

        Returns
        -------

        """
        self._wandb_run = None
        self._wandb_mode = task_information.get('logging').get('wandb_mode')
        wandb.init(project='RK-GNS', config=task_information, mode=self._wandb_mode)
        wandb.define_metric('epoch')
        wandb.define_metric('validation_loss', step_metric='epoch')
        wandb.define_metric('position_loss', step_metric='epoch')
        wandb.define_metric('validation_mean', step_metric='epoch')
        wandb.define_metric('position_mean', step_metric='epoch')
        wandb.define_metric('rollout_loss', step_metric='epoch')
        wandb.define_metric('video', step_metric='epoch')

        if self._wandb_url is not None and self._wandb_mode == 'online':
            api = wandb.Api()
            run = api.run(self._wandb_url)
            this_run = api.run(wandb.run.path)
            curr_epoch = max([x['epoch'] for x in run.scan_history(keys=['epoch'])])
            for file in run.files():
                this_run.upload_file(file.download(replace=True).name)
            b = False
            for x in run.scan_history():
                if b:
                    break
                try:
                    b = x['epoch'] >= curr_epoch
                except (KeyError, TypeError):
                    b = False
                wandb.log(x)

        self._wandb_url = wandb.run.path

        if not self._initialized:
            self._batch_size = task_information.get('task').get('batch_size')
            self._network = get_model(task_information)
            self._optimizer = optim.Adam(self._network.parameters(), lr=self._learning_rate)
            self._initialized = True

    @abstractmethod
    def fetch_data(self, trajectory: List, is_training: bool) -> DataLoader:
        """
        Transform a collection of system states into batched graphs.

        Parameters
        ----------
            trajectory : DataLoader
                A collection of system states
            is_training : bool
                Whether this is a training or a test/validation run

        Returns
        -------
            DataLoader
                Collection of batched graphs.
        """
        raise NotImplementedError

    @abstractmethod
    def fit_iteration(self, train_dataloader: List) -> None:
        """
        Train your algorithm for a single iteration. This can e.g., be a single epoch of neural network training,
        a policy update step, or something more complex. Just see this as the outermost for-loop of your algorithm.

        Parameters
        ----------
            train_dataloader : List
                A List containing the training data

        Returns
        -------
            May return an optional dictionary of values produced during the fit. These may e.g., be statistics
            of the fit such as a training loss.

        """

        raise NotImplementedError

    @torch.no_grad()
    def one_step_evaluator(self, ds_loader: List, instances: int, task_name: str, logging: bool = True) -> Optional[Dict]:
        """
        Predict the system state for the next time step and evaluate the predictions over the test data.

        Parameters
        ----------
            ds_loader : List
                A list containing test/validation instances

            instances : int
                Number of trajectories used to estimate the one-step loss

            task_name : str
                Name of the task

            logging : bool
                Whether to log the results to wandb

        Returns
        -------
            Optional[Dict]
                A single result that scores the input, potentially per sample

        """
        trajectory_loss = list()
        test_loader = self.fetch_data(ds_loader, is_training=False)
        for i, batch in enumerate(tqdm(test_loader, desc='Validation', leave=True, position=0)):
            batch.to(device)
            instance_loss = self._network.validation_step(batch, i)

            trajectory_loss.append([instance_loss])

        mean = np.mean(trajectory_loss, axis=0)
        std = np.std(trajectory_loss, axis=0)

        path = os.path.join(self._out_dir, f'{task_name}_one_step.csv')
        data_frame = pd.DataFrame.from_dict(
            {'mean_loss': [x[0] for x in mean], 'std_loss': [x[0] for x in std],
             'mean_pos_error': [x[1] for x in mean], 'std_pos_error': [x[1] for x in std]
             }
        )
        data_frame.to_csv(path)

        if logging:
            table = wandb.Table(dataframe=data_frame)
            val_loss, pos_loss = zip(*mean)
            log_dict = {
                'validation_loss':
                    wandb.Histogram(
                        [x for x in val_loss if np.quantile(val_loss, 0.90) > x],
                        num_bins=256
                    ),
                'position_loss':
                    wandb.Histogram(
                        [x for x in pos_loss if np.quantile(pos_loss, 0.90) > x],
                        num_bins=256
                    ),
                'validation_mean': np.mean(val_loss),
                'position_mean': np.mean(pos_loss),
                f'{task_name}_one_step': table
            }
            return log_dict
        else:
            self._publish_csv(data_frame, f'one_step', path)

    @torch.no_grad()
    def n_step_evaluator(self, ds_loader: List, task_name: str, n_steps: int, n_traj: int, logging: bool = True) -> Optional[Dict]:
        """
        Predict the system state after n time steps. N step predictions are performed recursively within trajectories.
        Evaluate the predictions over the test data.

        Parameters
        ----------
            ds_loader : List
                A list containing test/validation instances

            task_name : str
                Name of the task

            n_steps : int
                Value of n, with which to estimate the n-step loss

            n_traj : int
                Number of trajectories used to estimate the n-step loss

        Returns
        -------

        """
        # Take n_traj trajectories from valid set for n_step loss calculation
        means = list()
        lasts = list()
        for i, trajectory in enumerate(tqdm(ds_loader, desc='N-Step', leave=True, position=0)):
            if i >= n_traj:
                break
            mean_loss, last_loss = self._network.n_step_computation(trajectory, n_steps, self._time_steps)
            means.append(mean_loss)
            lasts.append(last_loss)

        means = torch.mean(torch.stack(means))
        lasts = torch.mean(torch.stack(lasts))

        path = os.path.join(self._out_dir, f'{task_name}_n_step_losses.csv')
        n_step_stats = {'n_step': [n_steps] * n_steps, 'mean': means, 'lasts': lasts}
        data_frame = pd.DataFrame.from_dict(n_step_stats)
        data_frame.to_csv(path)

        if logging:
            table = wandb.Table(dataframe=data_frame)
            return {f'mean_{n_steps}_loss': torch.mean(torch.tensor(means), dim=0),
                    f'{n_steps}_loss': torch.mean(torch.tensor(lasts), dim=0),
                    f'{task_name}_n_step_losses': table}
        else:
            self._publish_csv(data_frame, f'n_step_losses', path)

    @torch.no_grad()
    def rollout_evaluator(self, ds_loader: List, rollouts: int, task_name: str, logging: bool = True) -> Optional[Dict]:
        """
        Recursive prediction of the system state at the end of trajectories.
        Evaluate the predictions over the test data.

        Parameters
        ----------
            ds_loader : List
                A list containing test/validation instances

            rollouts : int
                Number of trajectories used to estimate the rollout loss

            task_name : str
                Name of the task

            logging : bool
                Whether to log the results to wandb

        Returns
        -------
            Optional[Dict]
                A single result that scores the input, potentially per sample

        """
        trajectories = []
        mse_losses = []
        for i, trajectory in enumerate(tqdm(ds_loader, desc='Rollouts', leave=True, position=0)):
            if i >= rollouts:
                break
            prediction_trajectory, mse_loss = self._network.rollout(trajectory, num_steps=self._time_steps)
            trajectories.append(prediction_trajectory)
            mse_losses.append(mse_loss.cpu())

        rollout_hist = wandb.Histogram([x for x in torch.mean(torch.stack(mse_losses), dim=1)], num_bins=10)

        mse_means = torch.mean(torch.stack(mse_losses), dim=0)
        mse_stds = torch.std(torch.stack(mse_losses), dim=0)

        rollout_losses = {
            'mse_loss': [mse.item() for mse in mse_means],
            'mse_std': [mse.item() for mse in mse_stds]
        }

        self.save_rollouts(trajectories, task_name)

        path = os.path.join(self._out_dir, f'{task_name}_rollout_losses.csv')
        data_frame = pd.DataFrame.from_dict(rollout_losses)
        data_frame.to_csv(path)

        if logging:
            table = wandb.Table(dataframe=data_frame)
            return {'mean_rollout_loss': torch.mean(torch.tensor(rollout_losses['mse_loss']), dim=0),
                    'rollout_loss': rollout_losses['mse_loss'][-1],
                    f'{task_name}_rollout_losses': table, 'rollout_hist': rollout_hist}
        else:
            self._publish_csv(data_frame, f'rollout_losses', path)

    def save(self, name: str) -> None:
        """
        Save itself as a .pkl file.

        Parameters
        ----------
            name : str
                The name under which to store this mesh simulator
        """
        with open(os.path.join(self._out_dir, f'model_{name}.pkl'), 'wb') as file:
            pickle.dump(self, file)

    def save_rollouts(self, rollouts: List[Dict[str, Tensor]], task_name: str) -> None:
        """
        Save predicted and ground truth trajectories.

        Parameters
        ----------
            rollouts : Dict[str, Tensor]
                The rollout data

            task_name : str
                The task name
        """
        rollouts = [{key: value.to('cpu') for key, value in x.items()} for x in rollouts]
        with open(os.path.join(self._out_dir, f'{task_name}_rollouts.pkl'), 'wb') as file:
            pickle.dump(rollouts, file)

    @staticmethod
    def _publish_csv(data_frame: DataFrame, name: str, path: str) -> None:
        """
        Publish a table using wandb.

        Parameters
        ----------
            data_frame : DataFrame
                The table
            name : str
                The table name
            path : str
                The path of the table
        """
        table = wandb.Table(dataframe=data_frame)
        wandb.log({name: table})
        #artifact = wandb.Artifact(f'{name}_artifact', type='dataset')
        #artifact.add(table, f'{name}_table')
        #artifact.add_file(path)
        #wandb.log_artifact(artifact)

    @staticmethod
    def log_epoch(data: Dict[str, Any]) -> None:
        """
        Log the metrics of an epoch.

        Parameters
        ----------
            data : Dict[str, Any]
                The data to log

        """
        wandb.log(data)
