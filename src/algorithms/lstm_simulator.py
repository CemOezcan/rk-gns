import os
import pickle
import time
from typing import List, Union

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

    def fit_iteration(self, train_dataloader: List[List[Data]]) -> None:
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

            for j, graph in enumerate(sequence):
                graph.to(device)
                if j != 0:
                    graph.h = h
                pred_velocity, h = self._network(graph, True)
                target_velocity = self._network.get_target(graph, True)

                target_list.append(target_velocity)
                pred_list.append(pred_velocity)

            target = torch.stack(target_list, dim=1)
            pred = torch.stack(pred_list, dim=1)

            loss = self._network.loss_fn(target, pred)
            loss.backward()

            gradients = self.log_gradients(self._network)

            self._optimizer.step()
            self._optimizer.zero_grad()

            end_instance = time.time()
            wandb.log({**gradients, 'training/loss': loss.detach(), 'training/sequence_time': end_instance - start_instance})
            start_instance = time.time()

            total_loss += loss.detach()
            size = i

        return total_loss / size

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
        mgn = self._config.get('task').get('mgn')
        poisson = self._config.get('task').get('model').lower() == 'poisson'

        if mgn:
            mode = 'mgn'
        elif poisson:
            mode = 'poisson'
        else:
            mode = None

        if is_training:
            trajectories = [list() for _ in range(len(trajectory) // self._time_steps)]
            for i, graph in enumerate(trajectory):
                index = i // self._time_steps
                trajectories[index].append(graph)

            dataset = SequenceNoReturnDataset(trajectories, self._seq_len, partial(self._network.build_graph, is_training=True), mode)
        else:
            dataset = RegularDataset(trajectory, partial(self._network.build_graph, is_training=False), mode)

        batches = DataLoader(dataset, batch_size=self._batch_size, shuffle=True, pin_memory=True, num_workers=8,
                             prefetch_factor=2, worker_init_fn=self.seed_worker)

        return batches
