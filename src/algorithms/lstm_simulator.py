import os
import pickle
import time
from typing import Optional, Dict, List, Tuple, Any

import numpy as np

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
import wandb
from pandas import DataFrame
from torch import Tensor
from tqdm import tqdm

from src.data.get_data import get_directories
from src.algorithms.abstract_simulator import AbstractSimulator
from src.model.abstract_system_model import AbstractSystemModel
from src.model.get_model import get_model
from torch_geometric.data import DataLoader, Batch
from src.util.types import ConfigDict, NodeType


class LSTMSimulator(AbstractSimulator):
    """
    Class for training and evaluating a graph neural network for mesh based physics simulations.
    """

    def __init__(self, config: ConfigDict) -> None:
        """
        Initializes the mesh simulator.

        Parameters
        ----------
            config : ConfigDict
                A (potentially nested) dictionary containing the "params" section of the section in the .yaml file
                used by cw2 for the current run.

        """
        super().__init__(config=config)
        self._network_config = config.get('model')
        self._dataset_name = config.get('task').get('dataset')
        _, self._out_dir = get_directories(self._dataset_name)
        self._wandb_mode = config.get('logging').get('wandb_mode')

        self._trajectories = config.get('task').get('trajectories')
        self._time_steps = config.get('task').get('n_timesteps')
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
        Initialize wandb and attributes, that should not be reinitialized once the model has been trained
        to allow training to be continued.

        Parameters
        ----------
            task_information : ConfigDict
                A dictionary containing information on how to execute the algorithm on the current task

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

    def fit_iteration(self, train_dataloader: DataLoader) -> None:
        """
        Perform a training epoch, followed by a validation iteration to assess the model performance.
        Document relevant metrics with wandb.

        Parameters
        ----------
            train_dataloader : DataLoader
                A data loader containing the training data

        Returns
        -------
            May return an optional dictionary of values produced during the fit. These may e.g., be statistics
            of the fit such as a training loss.

        """
        self._network.train()
        data = self.fetch_data(train_dataloader, True)
        target_list = list()
        pred_list = list()
        for i, graph in enumerate(tqdm(data, desc='Batches', leave=True, position=0)):
            graph = Batch.from_data_list(graph)
            if i != 0 and i % self._time_steps == 0:
                target = torch.stack(target_list, dim=1)
                pred = torch.stack(pred_list, dim=1)
                loss = self._network.loss_fn(target, pred)
                loss.backward()

                end_instance = time.time()
                wandb.log({'loss': loss.detach(), 'training time per instance': end_instance - start_instance})

                self._optimizer.step()
                self._optimizer.zero_grad()

                target_list = list()
                pred_list = list()
            elif i != 0:
                graph.u = hidden
                graph.h = h
                graph.c = c

            start_instance = time.time()

            mask = torch.where(graph.node_type == NodeType.MESH)[0]

            output, (hidden, (h, c)) = self._network(graph)

            pred_velocity = output[mask]
            target_velocity = graph.y - graph.pos[mask]

            target_velocity = self._network._output_normalizer(target_velocity, True)
            target_list.append(target_velocity)
            pred_list.append(pred_velocity)

    def fetch_data(self, trajectory: DataLoader, is_training: bool) -> DataLoader:
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
        graphs = []
        # TODO: Parallelize
        for i, data_frame in enumerate(tqdm(trajectory, desc='Building Graphs', leave=False, position=0)):
            graph = self._network.build_graph(data_frame, is_training)
            graphs.append(graph)

        rest = self._trajectories % self._batch_size
        num_batched_trajectories = self._trajectories // self._batch_size
        if rest != 0:
            num_batched_trajectories += 1

        fst_batches = graphs[:(num_batched_trajectories - rest) * self._time_steps * self._batch_size]
        num_batches = len(fst_batches) // self._batch_size
        batches = [list() for _ in range(num_batches)]
        for i, graph in enumerate(fst_batches):
            trajectory = i // self._time_steps
            batch = (i - (self._time_steps * trajectory)) % num_batches
            batches[(batch + trajectory * self._time_steps) % len(batches)].append(graph)

        snd_batches = graphs[(num_batched_trajectories - rest) * self._time_steps * self._batch_size:]
        num_batches = rest * self._time_steps
        batches_2 = [list() for _ in range(num_batches)]
        for i, graph in enumerate(snd_batches):
            trajectory = i // self._time_steps
            batch = (i - (self._time_steps * trajectory)) % num_batches
            batches_2[(batch + trajectory * self._time_steps) % len(batches_2)].append(graph)

        batches = batches + batches_2



        return batches

    @torch.no_grad()
    def one_step_evaluator(self, ds_loader: DataLoader, instances: int, task_name: str, logging=True) -> Optional[Dict]:
        """
        Predict the system state for the next time step and evaluate the predictions over the test data.

        Parameters
        ----------
            ds_loader : DataLoader
                A data loader containing test/validation instances

            instances : int
                Number of trajectories used to estimate the one-step loss

            task_name : str
                Name of the task

            logging : bool
                Whether to log the results to wandb

        Returns
        -------
            Optional[Dict]
                Estimates of loss statistics over the input trajectories

        """
        trajectory_loss = list()
        test_loader = self.fetch_data(ds_loader, is_training=False)
        for i, batch in enumerate(tqdm(test_loader, desc='Validation', leave=True, position=0)):
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

    def rollout_evaluator(self, ds_loader: DataLoader, rollouts: int, task_name: str, logging=True) -> Optional[Dict]:
        """
        Recursive prediction of the system state at the end of trajectories.
        Evaluate the predictions over the test data.

        Parameters
        ----------
            ds_loader : DataLoader
                A data loader containing test/validation instances

            rollouts : int
                Number of trajectories used to estimate the rollout loss

            task_name : str
                Name of the task

            logging : bool
                Whether to log the results to wandb

        Returns
        -------
            Optional[Dict]
                Estimates of loss statistics over the input trajectories

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

    def n_step_evaluator(self, ds_loader: DataLoader, task_name: str, n_steps: int = 60, n_traj: int = 2,
                         logging: bool = True) -> Optional[Dict]:
        """
        Predict the system state after n time steps. N step predictions are performed recursively within trajectories.
        Evaluate the predictions over the test data.

        Parameters
        ----------
            ds_loader : DataLoader
                A data loader containing test/validation instances

            task_name : str
                Name of the task

            n_steps: int
                Evaluation interval.

            n_traj : int
                Number of trajectories used to estimate the n-step loss

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
        artifact = wandb.Artifact(f'{name}_artifact', type='dataset')
        artifact.add(table, f'{name}_table')
        artifact.add_file(path)
        wandb.log_artifact(artifact)

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

    @property
    def network(self) -> AbstractSystemModel:
        """

        Returns
        -------
            AbstractSystemModel
                The graph neural network
        """
        return self._network

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