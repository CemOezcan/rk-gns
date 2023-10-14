import os
import pickle
import time
from typing import List, Union, Dict, Optional

import numpy as np
import torch
import wandb

from functools import partial

from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from src.data.datasets import SequenceNoReturnDataset, RegularDataset
from src.algorithms.abstract_simulator import AbstractSimulator
from src.util.types import ConfigDict
from src.util.util import device


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

    def fit_iteration(self, train_dataloader: List[List[Data]]) -> float:
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
        total_loss = 0

        start_instance = time.time()
        for i, sequence in enumerate(tqdm(data, desc='Batches', leave=True, position=0)):
            target_list = list()
            pred_list = list()
            pred_var_list = list()

            for j, graph in enumerate(sequence):
                graph.to(device)
                if j != 0:
                    graph.h = h
                    graph.c = c
                (pred_velocity, pred_var), (h, c) = self._network(graph, True)
                target_velocity = self._network.get_target(graph, True)

                target_list.append(target_velocity)
                pred_list.append(pred_velocity)
                pred_var_list.append(pred_var)

            target = torch.stack(target_list, dim=1)
            pred_mean = torch.stack(pred_list, dim=1)
            if pred_var_list[0] is not None:
                pred_var = torch.stack(pred_var_list, dim=1)
            else:
                pred_var = None

            loss = self._network.loss_fn(target, pred_mean, pred_var)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self._network.parameters(), 2)

            gradients = self.log_gradients(self._network)

            self._optimizer.step()
            self._optimizer.zero_grad()

            end_instance = time.time()
            wandb.log({**gradients, 'training/loss': loss.detach(), 'training/sequence_time': end_instance - start_instance})
            start_instance = time.time()

            total_loss += loss.detach()
            size = i

        return total_loss / size

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
        loss_fn = torch.nn.MSELoss()
        test_loader = self.fetch_data(ds_loader, is_training=False)
        for i, sequence in enumerate(tqdm(test_loader, desc='Batches', leave=True, position=0)):

            for j, graph in enumerate(sequence):
                graph.to(device)
                if j != 0:
                    graph.h = h
                    graph.c = c

                (pred_velocity, pred_var), (h, c) = self._network(graph, False)
                target_velocity = self._network.get_target(graph, False)
                error = self._network.loss_fn(target_velocity, pred_velocity, pred_var).cpu()

                pred_position, _, _ = self._network.update(graph, pred_velocity)
                if self.mode == 'poisson':
                    pos_error = loss_fn(pred_position, graph.poisson).cpu()
                else:
                    pos_error = loss_fn(pred_position, graph.y).cpu()
                trajectory_loss.append([(error, pos_error)])

        mean = np.mean(trajectory_loss, axis=0)
        std = np.std(trajectory_loss, axis=0)

        val_loss, pos_loss = zip(*mean)
        val_std, pos_std = zip(*std)

        log_dict = {
            'single-step error/velocity_historgram':
                wandb.Histogram(
                    [x for x in val_loss],
                    num_bins=20
                ),
            'single-step error/position_historgram':
                wandb.Histogram(
                    [x for x in pos_loss],
                    num_bins=20
                ),
            'single-step error/velocity_error': np.mean(val_loss),
            'single-step error/position_error': np.mean(pos_loss),
            'single-step error/velocity_std': np.mean(val_std),
            'single-step error/position_std': np.mean(pos_std)
        }
        return log_dict

    def fetch_data(self, trajectory: List[Union[List[Data], Data]], is_training: bool) -> DataLoader:
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
        seq = self._seq_len if is_training else 50
        trajectories = [list() for _ in range(len(trajectory) // self._time_steps)]

        for i, graph in enumerate(trajectory):
            index = i // self._time_steps
            trajectories[index].append(graph)

        dataset = SequenceNoReturnDataset(trajectories, seq, partial(self._network.build_graph, is_training=is_training), self.mode)
        batches = DataLoader(dataset, batch_size=self._batch_size, shuffle=True, pin_memory=True, num_workers=8,
                             prefetch_factor=2, worker_init_fn=self.seed_worker)

        return batches
