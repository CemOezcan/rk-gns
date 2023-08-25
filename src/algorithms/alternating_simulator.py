import copy
import os
import pickle
import time
from typing import List, Union, Optional, Dict

import numpy as np
import pandas as pd
import torch
import wandb

from functools import partial

from torch import optim
from torch.nn import LSTM, GRU
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from src.data.datasets import SequenceNoReturnDataset, RegularDataset
from src.algorithms.abstract_simulator import AbstractSimulator

from src.model.get_model import get_model
from src.util import test
from src.util.types import ConfigDict, NodeType
from src.util.util import device


class AlternatingSimulator(AbstractSimulator):
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
        self.global_model = None
        self.global_optimizer = None

    def initialize(self, task_information: ConfigDict) -> None:
        if not self._initialized:
            self.global_model = get_model(task_information, poisson=True)
            self.global_optimizer = optim.Adam(self.global_model.parameters(), lr=self._learning_rate)

        super().initialize(task_information)

    def pretraining(self, train_dataloader: List) -> None:
        """
        Pretrain your algorithm if necessary. Default is no pretraining

        Parameters
        ----------
            train_dataloader: List
                A List containing the training data

        Returns
        -------

        """
        for _ in range(5):
            self.fit_poisson(train_dataloader)

    def fit_iteration(self, train_dataloader: List[Union[List[Data], Data]]) -> None:
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
        self.fit_poisson(train_dataloader)
        self.fit_gnn(train_dataloader)

    def fit_poisson(self, train_dataloader: List[Data]):
        self.global_model.train()
        data = self.fetch_data(train_dataloader, True, mgn=True)

        for i, batch in enumerate(tqdm(data, desc='Batches', leave=True, position=0)):
            start_instance = time.time()
            batch.to(device)
            loss = self.global_model.training_step(batch)
            loss.backward()

            self.global_optimizer.step()
            self.global_optimizer.zero_grad()

            end_instance = time.time()
            wandb.log({'poisson_loss': loss.detach(), 'training time per instance': end_instance - start_instance})

    def fit_lstm(self, train_dataloader: List[List[Data]]):
        self.global_model.train()
        data = self.fetch_data(train_dataloader, True)

        for i, sequence in enumerate(tqdm(data, desc='Batches', leave=True, position=0)):
            target_list = list()
            pred_list = list()

            for j, graph in enumerate(sequence):
                _, graph = self.split_graphs(graph)

                graph.to(device)
                if j != 0:
                    graph.h = h

                pred_velocity, h = self.global_model(graph, True)
                target_velocity = self.global_model.get_target(graph, True)

                target_list.append(target_velocity)
                pred_list.append(pred_velocity)

            target = torch.stack(target_list, dim=1)
            pred = torch.stack(pred_list, dim=1)

            loss = self.global_model.loss_fn(target, pred)
            loss.backward()

            self.global_optimizer.step()
            self.global_optimizer.zero_grad()
            wandb.log({'poisson_loss': loss.detach()})

    def fit_gnn(self, train_dataloader: List[Data]) -> None:
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
        self.global_model.eval()
        data = self.fetch_data(train_dataloader, True, mgn=True)

        for i, batch in enumerate(tqdm(data, desc='Batches', leave=True, position=0)):
            start_instance = time.time()
            batch.to(device)

            loss = self._network.training_step(batch, self.global_model)
            loss.backward()

            self._optimizer.step()
            self._optimizer.zero_grad()

            end_instance = time.time()
            wandb.log({'loss': loss.detach(), 'training time per instance': end_instance - start_instance})
        # self._network.train()
        # self.global_model.eval()
        # data = self.fetch_data(train_dataloader, True, mgn=True)
        #
        # for i, batch in enumerate(tqdm(data, desc='Batches', leave=True, position=0)):
        #     start_instance = time.time()
        #     batch, pc = self.split_graphs(batch)
        #     batch.to(device)
        #     pc.to(device)
        #
        #     output, _ = self.global_model(pc, False)
        #     poisson = self.global_model._output_normalizer.inverse(output)
        #     batch.u = poisson
        #
        #     loss = self._network.training_step(batch)
        #     loss.backward()
        #
        #     self._optimizer.step()
        #     self._optimizer.zero_grad()
        #
        #     end_instance = time.time()
        #     wandb.log({'loss': loss.detach(), 'training time per instance': end_instance - start_instance})

    @torch.no_grad()
    def one_step_evaluator(self, ds_loader: List, instances: int, task_name: str, logging: bool = True) -> Optional[
        Dict]:
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
            instance_loss = self._network.validation_step(batch, i, self.global_model)

            trajectory_loss.append([instance_loss])

        mean = np.mean(trajectory_loss, axis=0)
        std = np.std(trajectory_loss, axis=0)

        path = os.path.join(self._out_dir, f'{task_name}_one_step.csv')
        data_frame = pd.DataFrame.from_dict(
            {'mean_loss': [x[0] for x in mean], 'std_loss': [x[0] for x in std],
             'mean_pos_error': [x[1] for x in mean], 'std_pos_error': [x[1] for x in std],
             'mean_u_loss': [x[2] for x in mean], 'std_u_loss': [x[2] for x in std],
             'mean_poisson_error': [x[3] for x in mean], 'std_poisson_error': [x[3] for x in std]
             }
        )
        data_frame.to_csv(path)

        if logging:
            table = wandb.Table(dataframe=data_frame)
            val_loss, pos_loss, u_error, poisson_error = zip(*mean)
            log_dict = {
                'validation_loss':
                    wandb.Histogram(
                        [x for x in val_loss if np.quantile(val_loss, 0.90) > x],
                        num_bins=256
                    ),
                'hist_u_loss':
                    wandb.Histogram(
                        [x for x in val_loss if np.quantile(val_loss, 0.90) > x],
                        num_bins=256
                    ),
                'hist_poisson_loss':
                    wandb.Histogram(
                        [x for x in pos_loss if np.quantile(pos_loss, 0.90) > x],
                        num_bins=256
                    ),
                'position_loss':
                    wandb.Histogram(
                        [x for x in pos_loss if np.quantile(pos_loss, 0.90) > x],
                        num_bins=256
                    ),
                'validation_mean': np.mean(val_loss),
                'position_mean': np.mean(pos_loss),
                'u_mean': np.mean(val_loss),
                'poisson_mean': np.mean(pos_loss),
                f'{task_name}_one_step': table
            }
            return log_dict
        else:
            self._publish_csv(data_frame, f'one_step', path)

    @torch.no_grad()
    def rollout_evaluator(self, ds_loader: List, rollouts: int, task_name: str, logging: bool = True, freq=1) -> Optional[Dict]:
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
        self.global_model.eval()
        self._network.eval()
        trajectories = []
        mse_losses = []
        u_losses = list()
        for i, trajectory in enumerate(tqdm(ds_loader, desc='Rollouts', leave=True, position=0)):
            if i >= rollouts:
                break
            prediction_trajectory, mse_loss, u_loss = self._network.rollout(trajectory, self._time_steps, self.global_model, freq=freq)
            trajectories.append(prediction_trajectory)
            mse_losses.append(mse_loss.cpu())
            u_losses.append(u_loss.cpu())

        rollout_hist = wandb.Histogram([x for x in torch.mean(torch.stack(mse_losses), dim=1)], num_bins=10)

        mse_means = torch.mean(torch.stack(mse_losses), dim=0)
        u_means = [x.item() for x in torch.mean(torch.stack(u_losses), dim=0)]
        mse_stds = torch.std(torch.stack(mse_losses), dim=0)

        rollout_losses = {
            'mse_loss': [mse.item() for mse in mse_means],
            'mse_std': [mse.item() for mse in mse_stds]
        }

        self.save_rollouts(trajectories, task_name, freq)

        path = os.path.join(self._out_dir, f'{task_name}_rollout_losses_k={freq}.csv')
        data_frame = pd.DataFrame.from_dict(rollout_losses)
        data_frame.to_csv(path)

        if logging:
            table = wandb.Table(dataframe=data_frame)
            return {f'mean_rollout_loss_k={freq}': torch.mean(torch.tensor(rollout_losses['mse_loss']), dim=0),
                    f'rollout_loss_k={freq}': rollout_losses['mse_loss'][-1],
                    f'{task_name}_rollout_losses_k={freq}': table, 'rollout_hist': rollout_hist,
                    f'u_loss_mean_k={freq}': torch.mean(torch.tensor(u_means), dim=0),
                    f'u_loss_rollout_k={freq}': u_means[-1]}
        else:
            self._publish_csv(data_frame, f'rollout_losses_k={freq}', path)

    @torch.no_grad()
    def n_step_evaluator(self, ds_loader: List, task_name: str, n_steps: int, n_traj: int, logging: bool = True, freq=1) -> \
    Optional[Dict]:
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
        u_means = list()
        u_lasts = list()
        for i, trajectory in enumerate(tqdm(ds_loader, desc='N-Step', leave=True, position=0)):
            if i >= n_traj:
                break
            mean_loss, last_loss, u_loss, u_last_loss = self._network.n_step_computation(trajectory, n_steps, self._time_steps, self.global_model, freq)
            means.append(mean_loss)
            lasts.append(last_loss)
            u_means.append(u_loss)
            u_lasts.append(u_last_loss)

        means = torch.mean(torch.stack(means))
        lasts = torch.mean(torch.stack(lasts))
        u_means = torch.mean(torch.stack(u_means))
        u_lasts = torch.mean(torch.stack(u_lasts))

        path = os.path.join(self._out_dir, f'{task_name}_n_step_losses_k={freq}.csv')
        n_step_stats = {'n_step': [n_steps] * n_steps, 'mean': means, 'lasts': lasts}
        data_frame = pd.DataFrame.from_dict(n_step_stats)
        data_frame.to_csv(path)

        if logging:
            table = wandb.Table(dataframe=data_frame)
            return {
                f'mean_{n_steps}_loss_k={freq}': torch.mean(torch.tensor(means), dim=0),
                f'{n_steps}_loss_k={freq}': torch.mean(torch.tensor(lasts), dim=0),
                f'{task_name}_n_step_losses_k={freq}': table,
                f'mean_{n_steps}_u_loss_k={freq}': torch.mean(torch.tensor(means), dim=0),
                f'{n_steps}_u_loss_k={freq}': torch.mean(torch.tensor(u_lasts), dim=0)
            }
        else:
            self._publish_csv(data_frame, f'n_step_losses_k={freq}', path)

    def fetch_data(self, trajectory: List[Union[List[Data], Data]], is_training: bool, seq=None, mgn=False) -> DataLoader:
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
        if mgn:
            dataset = RegularDataset(trajectory, partial(self._network.build_graph, is_training=True), mgn)
            batches = DataLoader(dataset, batch_size=self._batch_size, shuffle=True, pin_memory=True, num_workers=8,
                                 prefetch_factor=2, worker_init_fn=self.seed_worker)

            return batches

        if seq is None:
            seq = self._seq_len

        if is_training:
            trajectories = [list() for _ in range(len(trajectory) // self._time_steps)]
            for i, graph in enumerate(trajectory):
                index = i // self._time_steps
                trajectories[index].append(graph)
            dataset = SequenceNoReturnDataset(trajectories, seq, partial(self._network.build_graph, is_training=True))
        else:
            dataset = RegularDataset(trajectory, partial(self._network.build_graph, is_training=False), mgn)

        batches = DataLoader(dataset, batch_size=self._batch_size, shuffle=True, pin_memory=True,
                             num_workers=8, prefetch_factor=2, worker_init_fn=self.seed_worker)

        return batches


    def split_graphs(self, graph):
        pc_mask = torch.where(graph['mesh'].node_type == NodeType.POINT)[0]
        obst_mask = torch.where(graph['mesh'].node_type == NodeType.COLLIDER)[0]
        mesh_mask = torch.where(graph['mesh'].node_type == NodeType.MESH)[0]

        poisson_mask = torch.cat([pc_mask, obst_mask], dim=0)
        mgn_mask = torch.cat([mesh_mask, obst_mask], dim=0)

        pc = copy.deepcopy(graph.subgraph({'mesh': poisson_mask}))
        pc['mesh'] = torch.cat([pc['mesh'].pos, pc['mesh'].x], dim=1)
        mesh = copy.deepcopy(graph.subgraph({'mesh': mgn_mask}))

        return mesh, pc
