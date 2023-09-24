import time
from typing import List

import wandb

from functools import partial
from tqdm import tqdm
from torch_geometric.data import DataLoader, Data

from src.data.datasets import RegularDataset
from src.algorithms.abstract_simulator import AbstractSimulator
from src.util.types import ConfigDict
from src.util.util import device


class MeshSimulator(AbstractSimulator):
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

    def fit_iteration(self, train_dataloader: List[Data]) -> None:
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
        size = 0

        for i, batch in enumerate(tqdm(data, desc='Batches', leave=True, position=0)):
            start_instance = time.time()
            batch.to(device)
            loss = self._network.training_step(batch)
            loss.backward()

            gradients = self.log_gradients(self._network)

            self._optimizer.step()
            self._optimizer.zero_grad()

            end_instance = time.time()
            wandb.log({**gradients, 'training/loss': loss.detach(), 'training/instance_time': end_instance - start_instance})

            total_loss += loss.detach()
            size = i

        return total_loss / size

    def fetch_data(self, trajectory: List[Data], is_training: bool) -> DataLoader:
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

        dataset = RegularDataset(trajectory, partial(self._network.build_graph, is_training=is_training), mode)

        batches = DataLoader(dataset, batch_size=self._batch_size, shuffle=True, pin_memory=True, num_workers=8,
                             prefetch_factor=2, worker_init_fn=self.seed_worker)

        return batches
