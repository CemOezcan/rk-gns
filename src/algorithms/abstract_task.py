from abc import ABC, abstractmethod
from typing import Tuple
from matplotlib.animation import PillowWriter, FuncAnimation

from src.util.types import *


class AbstractTask(ABC):

    def __init__(self, config: ConfigDict):
        """
        Initializes the current task. Depending on the application, this can be something like a classification task
        for supervised learning (in which case this method would be used to load in the data and labels), or a gym
        environment for a Reinforcement Learning task.

        Parameters
        ----------
            config : ConfigDict
                A (potentially nested) dictionary containing the "params" section of the section in the .yaml file
                used by cw2 for the current run
        """
        self._config = config

    @abstractmethod
    def run_iterations(self) -> None:
        """
        Runs all iteration of its iterative algorithm.
        """
        raise NotImplementedError

    @abstractmethod
    def get_scalars(self) -> None:
        """
        Evaluates the final model after all training epochs.

        Returns
        -------

        """

        raise NotImplementedError

    @abstractmethod
    def plot(self, task_name: str) -> Tuple[FuncAnimation, PillowWriter]:
        """
        Create a plot for the current state of the task and its algorithm

        Parameters
        ----------
            task_name : str
                The name of the task

        Returns
        -------
            Tuple[FuncAnimation, PillowWriter]
                The simulations

        """

        raise NotImplementedError

