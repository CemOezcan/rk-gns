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
        self.pretraining_epochs = config.get('task').get('pretraining')
        if self.mode == 'mgn':
            self.mode = 'poisson'

    def initialize(self, task_information: ConfigDict) -> None:
        if not self._initialized:
            self.global_model = get_model(task_information, poisson=True)
            self.global_optimizer = optim.Adam(self.global_model.parameters(), lr=1e-4)

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
        for _ in range(self.pretraining_epochs):
            if self.recurrence:
                self.fit_lstm_poisson(train_dataloader)
            else:
                self.fit_poisson(train_dataloader)

    def fit_iteration(self, train_dataloader: List[Union[List[Data], Data]]) -> float:
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
        #self.fit_poisson(train_dataloader)
        if self.recurrence:
            return self.fit_lstm(train_dataloader)
        else:
            return self.fit_gnn(train_dataloader)

    def fit_poisson(self, train_dataloader: List[Data]):
        self.global_model.train()
        data = self.fetch_data(train_dataloader, True, mode='poisson')

        for i, batch in enumerate(tqdm(data, desc='Batches', leave=True, position=0)):
            start_instance = time.time()
            batch.to(device)
            loss = self.global_model.training_step(batch)
            loss.backward()

            gradients = self.log_gradients(self.global_model)

            self.global_optimizer.step()
            self.global_optimizer.zero_grad()

            end_instance = time.time()
            wandb.log({**gradients, 'training/material_loss': loss.detach(), 'training/material_instance_time': end_instance - start_instance})

    def fit_lstm_poisson(self, train_dataloader: List[Data]):
        self.global_model.train()
        data = self.fetch_data(train_dataloader, True, mode='poisson', seq=True, seq_len=self._seq_len)

        start_instance = time.time()
        for i, sequence in enumerate(tqdm(data, desc='Batches', leave=True, position=0)):
            target_list = list()
            pred_list = list()

            for j, graph in enumerate(sequence):
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

            gradients = self.log_gradients(self.global_model)

            self.global_optimizer.step()
            self.global_optimizer.zero_grad()

            end_instance = time.time()
            wandb.log({**gradients, 'training/material_loss': loss.detach(), 'training/sequence_time': end_instance - start_instance})
            start_instance = time.time()

    def fit_lstm(self, train_dataloader: List[List[Data]]):
        self._network.train()
        self.global_model.eval()
        # TODO: check mode --> sequence dataset only differentiates between mode == mgn and mode != mgn
        data = self.fetch_data(train_dataloader, True, mode='poisson', seq=True, seq_len=49)
        total_loss = 0
        size = 0

        for i, sequence in enumerate(tqdm(data, desc='Batches', leave=True, position=0)):
            for j, graph in enumerate(sequence):
                start_instance = time.time()
                graph.to(device)
                if j != 0:
                    graph.h = h

                loss = self._network.training_step(graph, self.global_model)
                loss.backward()
                h = graph.h.detach()

                gradients = self.log_gradients(self._network)

                self._optimizer.step()
                self._optimizer.zero_grad()

                end_instance = time.time()
                wandb.log({**gradients, 'training/loss': loss.detach(), 'training/instance_time': end_instance - start_instance})

                total_loss += loss.detach()
                size += 1

        return total_loss / size

    def fit_gnn(self, train_dataloader: List[Data]) -> float:
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
        data = self.fetch_data(train_dataloader, True, mode=self.mode)
        total_loss = 0

        for i, batch in enumerate(tqdm(data, desc='Batches', leave=True, position=0)):
            start_instance = time.time()
            batch.to(device)

            loss = self._network.training_step(batch, self.global_model)
            loss.backward()

            gradients = self.log_gradients(self._network)

            self._optimizer.step()
            self._optimizer.zero_grad()

            end_instance = time.time()
            wandb.log({**gradients, 'training/loss': loss.detach(), 'training/instance_time': end_instance - start_instance})

            total_loss += loss.detach()
            size = i

        return total_loss / size

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
        self.global_model.eval()
        self._network.eval()
        trajectory_loss = list()
        if self.recurrence:
            trajectory_loss = list()
            test_loader = self.fetch_data(ds_loader, is_training=False, mode='supervised', seq=True, seq_len=50)
            for i, sequence in enumerate(tqdm(test_loader, desc='Batches', leave=True, position=0)):

                for j, graph in enumerate(sequence):
                    graph.to(device)
                    if j != 0:
                        graph.h = h

                    output, h = self.global_model(graph, False)
                    u_target = self.global_model.get_target(graph, False)
                    u_error = self._network.loss_fn(output, u_target).cpu()

                    poisson, _, _ = self.global_model.update(graph, output)
                    poisson_error = self._network.loss_fn(poisson, graph.u).cpu()
                    graph.u = poisson

                    prediction, _ = self._network(graph, False)
                    target = self._network.get_target(graph, False)
                    error = self._network.loss_fn(target, prediction).cpu()

                    pred_position, _, _ = self._network.update(graph, prediction)
                    pos_error = self._network.loss_fn(pred_position, graph.y).cpu()
                    trajectory_loss.append([(error, pos_error, u_error, poisson_error)])
        else:
            test_loader = self.fetch_data(ds_loader, is_training=False, mode='supervised')
            for i, batch in enumerate(tqdm(test_loader, desc='Validation', leave=True, position=0)):
                batch.to(device)
                instance_loss = self._network.validation_step(batch, i, self.global_model)

                trajectory_loss.append([instance_loss])

        mean = np.mean(trajectory_loss, axis=0)
        std = np.std(trajectory_loss, axis=0)

        val_loss, pos_loss, u_error, poisson_error = zip(*mean)
        val_std, pos_std, u_std, poisson_std = zip(*std)
        log_dict = {
            'single-step error/velocity_historgram':
                wandb.Histogram(
                    [x for x in val_loss],
                    num_bins=20
                ),
            'single-step error/material_historgram':
                wandb.Histogram(
                    [x for x in val_loss],
                    num_bins=20
                ),
            'single-step error/poisson_historgram':
                wandb.Histogram(
                    [x for x in pos_loss],
                    num_bins=20
                ),
            'single-step error/position_historgram':
                wandb.Histogram(
                    [x for x in pos_loss],
                    num_bins=20
                ),
            'single-step error/material_error': np.mean(u_error),
            'single-step error/poisson_error': np.mean(poisson_error),
            'single-step error/velocity_error': np.mean(val_loss),
            'single-step error/position_error': np.mean(pos_loss),
            'single-step error/velocity_std': np.mean(val_std),
            'single-step error/position_std': np.mean(pos_std),
            'single-step error/material_std': np.mean(u_std),
            'single-step error/poisson_std': np.mean(poisson_std)
        }
        return log_dict

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

        rollout_hist = wandb.Histogram([x for x in torch.mean(torch.stack(mse_losses), dim=1)], num_bins=20)

        mse_means = torch.mean(torch.stack(mse_losses), dim=0)
        u_means = [x.item() for x in torch.mean(torch.stack(u_losses), dim=0)]
        mse_stds = torch.std(torch.stack(mse_losses), dim=0)

        rollout_losses = {
            'mse_loss': [mse.item() for mse in mse_means],
            'mse_std': [mse.item() for mse in mse_stds]
        }

        self.save_rollouts(trajectories, task_name, freq)

        current_mean = torch.mean(torch.tensor(rollout_losses['mse_loss']), dim=0)
        prior_mean = self.best_models[freq][0]
        self.best_models[freq] = (current_mean, self._network) if current_mean < prior_mean else self.best_models[freq]

        return {f'rollout error/mean_k={freq}': torch.mean(torch.tensor(rollout_losses['mse_loss']), dim=0),
                f'rollout error/std_k={freq}': torch.mean(torch.tensor(rollout_losses['mse_std']), dim=0),
                f'rollout error/last_k={freq}': rollout_losses['mse_loss'][-1],
                f'rollout error/histogram_k={freq}': rollout_hist,

                f'rollout error/material_mean_k={freq}': torch.mean(torch.tensor(u_means), dim=0),
                f'rollout error/material_middle_k={freq}': u_means[int(len(u_means) / 2)],
                f'rollout error/material_last_k={freq}': u_means[-1]}

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
        self.global_model.eval()
        self._network.eval()
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

        return {
            f'{n_steps}-step error/mean_k={freq}': torch.mean(torch.tensor(means), dim=0),
            f'{n_steps}-step error/last_k={freq}': torch.mean(torch.tensor(lasts), dim=0),
            f'{n_steps}-step error/material_mean_k={freq}': torch.mean(torch.tensor(u_means), dim=0),
            f'{n_steps}-step error/material_last_k={freq}': torch.mean(torch.tensor(u_lasts), dim=0)
        }

    def fetch_data(self, trajectory: List[Union[List[Data], Data]], is_training: bool, mode=None, seq=False, seq_len=49) -> DataLoader:
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
        if seq:
            trajectories = [list() for _ in range(len(trajectory) // self._time_steps)]
            for i, graph in enumerate(trajectory):
                index = i // self._time_steps
                trajectories[index].append(graph)

            dataset = SequenceNoReturnDataset(trajectories, seq_len, partial(self._network.build_graph, is_training=is_training), mode)

            batches = DataLoader(dataset, batch_size=self._batch_size, shuffle=True, pin_memory=True, num_workers=8,
                                 prefetch_factor=2, worker_init_fn=self.seed_worker)

        else:

            dataset = RegularDataset(trajectory, partial(self._network.build_graph, is_training=is_training), mode)

            batches = DataLoader(dataset, batch_size=self._batch_size, shuffle=True, pin_memory=True, num_workers=8,
                                 prefetch_factor=2, worker_init_fn=self.seed_worker)

        return batches
