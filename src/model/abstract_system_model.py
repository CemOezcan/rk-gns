from typing import Tuple

import torch
from torch import nn, Tensor

from src.util.types import *
from abc import ABC, abstractmethod


class AbstractSystemModel(ABC, nn.Module):
    """
    Superclass for neural system model estimators
    """

    def __init__(self, params: ConfigDict) -> None:
        super(AbstractSystemModel, self).__init__()
        self._params = params

    @abstractmethod
    def training_step(self, graph: Batch) -> Tensor:
        """
        Perform a single training step.

        Parameters
        ----------
            graph : MultiGraph
                The current system state represented by a (heterogeneous hyper-) graph

            data_frame : Dict[str, Tensor]
                Additional information on the instance and the target system state

        Returns
        -------
            Tensor
                The training loss
        """

        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def validation_step(self, graph: Batch, data_frame: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Evaluate given input data and potentially auxiliary information to create a dictionary of resulting values.
        What kinds of things are scored/evaluated depends on the concrete algorithm.

        Parameters
        ----------
            graph : MultiGraph
                The current system state represented by a (heterogeneous hyper-) graph

            data_frame : Dict[str, Tensor]
                Additional information on the instance and the target system state

        Returns
        -------
            Tensor
                A dictionary with different values that are evaluated from the given input data. May e.g., the
                accuracy of the model.
        """

        raise NotImplementedError

    @abstractmethod
    def update(self, inputs: Batch, per_node_network_output: Tensor) -> Tensor:
        """
        Makes a prediction for the given input data and uses it to compute the predicted system state.

        Parameters
        ----------
            inputs : Dict[str, Tensor]
                Dictionary containing the current and the previous system states

            per_node_network_output : Tensor
                Predicted (normalized) system dynamics

        Returns
        -------
            Tensor
                Some representation of the predicted system state
        """

        raise NotImplementedError

    @abstractmethod
    def build_graph(self, inputs: Tuple[Data, Data], is_training: bool) -> HeteroData:
        """
        Constructs the input graph given a system state.

        Parameters
        ----------
            inputs : Dict[str, Tensor]
                System state information

            is_training : bool
                Whether the input is a training instance or not

        Returns
        -------
            The system state represented by a graph

        """

        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def rollout(self, trajectory: List[Dict[str, Tensor]], num_steps: int) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Predict a sub trajectory for n time steps by making n consecutive one-step predictions recursively.

        Parameters
        ----------
            trajectory : Dict[str, Tensor]
                A trajectory of subsequent system states

            num_steps : int
                Number of time steps

        Returns
        -------
            Tuple[Dict[str, Tensor], Tensor]
                The predicted and the ground truth trajectories as well as the corresponding losses for each time step

        """

        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def n_step_computation(self, trajectory: List[Dict[str, Tensor]], n_step: int) -> Tuple[Tensor, Tensor]:
        """
        Predict the system state after n time steps. N step predictions are performed recursively within trajectories.

        Parameters
        ----------
            trajectory : Dict[str, Tensor]
                A trajectory of subsequent system states

            n_step: int
                Number of time steps

        Returns
        -------
            Tensor
                The n-step loss

        """

        raise NotImplementedError

    def evaluate(self) -> None:
        """
        Activate evaluation mode.

        Returns
        -------

        """
        self.eval()
        self.learned_model.eval()
