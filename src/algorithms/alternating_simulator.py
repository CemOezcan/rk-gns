import copy
import os
import pickle
import time
from typing import List, Union, Optional, Dict

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
            self.global_model = get_model(task_information, poison=True)
            self.global_optimizer = optim.Adam(self.global_model.parameters(), lr=self._learning_rate)

        super().initialize(task_information)

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
        self.fit_lstm(train_dataloader)
        self.fit_gnn(train_dataloader)

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
        data = self.fetch_data(train_dataloader, True, mgn=True)

        for i, batch in enumerate(tqdm(data, desc='Batches', leave=True, position=0)):
            start_instance = time.time()
            batch, _ = self.split_graphs(batch)
            batch.u = batch.poisson
            batch.to(device)

            loss = self._network.training_step(batch)
            loss.backward()

            self._optimizer.step()
            self._optimizer.zero_grad()

            end_instance = time.time()
            wandb.log({'loss': loss.detach(), 'training time per instance': end_instance - start_instance})

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
        self.global_model.eval()
        self._network.eval()
        trajectories = []
        mse_losses = []
        u_losses = list()
        for i, trajectory in enumerate(tqdm(ds_loader, desc='Rollouts', leave=True, position=0)):
            if i >= rollouts:
                break
            prediction_trajectory, mse_loss, u_loss = self._network.rollout(trajectory, self._time_steps, self.global_model)
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

        self.save_rollouts(trajectories, task_name)

        path = os.path.join(self._out_dir, f'{task_name}_rollout_losses.csv')
        data_frame = pd.DataFrame.from_dict(rollout_losses)
        data_frame.to_csv(path)

        if logging:
            table = wandb.Table(dataframe=data_frame)
            return {'mean_rollout_loss': torch.mean(torch.tensor(rollout_losses['mse_loss']), dim=0),
                    'rollout_loss': rollout_losses['mse_loss'][-1],
                    f'{task_name}_rollout_losses': table, 'rollout_hist': rollout_hist,
                    'u_loss_mean': torch.mean(torch.tensor(u_means), dim=0),
                    'u_loss_rollout': u_means[-1],
                    'u_loss_10': u_means[9]}
        else:
            self._publish_csv(data_frame, f'rollout_losses', path)

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
            dataset = RegularDataset(trajectory, partial(self._network.build_graph, is_training=True))
            batches = DataLoader(dataset, batch_size=self._batch_size, shuffle=True, pin_memory=True, num_workers=8,
                                 prefetch_factor=2)

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
            dataset = RegularDataset(trajectory, partial(self._network.build_graph, is_training=False))

        batches = DataLoader(dataset, batch_size=self._batch_size, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=2)

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
