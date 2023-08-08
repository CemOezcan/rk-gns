import itertools

from copy import deepcopy
from typing import Callable

from torch.utils.data import Dataset


class SequenceNoReturnDataset(Dataset):
    """
        Dataset that draws sequences of specific length from a list of trajectories without replacement.
        In this case, we can still define a training epoch, if all samples are used once.
    """
    def __init__(self, trajectory_list: list, sequence_length: int, preprocessing: Callable):
        """
        Args:
            trajectory_list: List of lists of (PyG) data objects
            sequence_length: Length of the drawn sequence from a trajectory
        """
        self.trajectory_list = trajectory_list
        self.sequence_length = sequence_length
        self.preprocessing = preprocessing

        # create index list of tuples (i, t_i), where i indicates the index for the trajectory and t_i for the starting time step of the sequence
        self.indices = []
        for trajectory in range(len(trajectory_list)):
            difference = max(len(trajectory_list[trajectory]) - self.sequence_length, 1)
            self.indices.extend([(i, t_i) for i, t_i in itertools.product([trajectory], range(difference))])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        """
        Args:
            idx: Index from sampler or batch_sampler of the Dataloader
        Returns:
            data_list: List of Data elements containing a batch of graphs
        """
        self.index = self.indices[idx]
        self.trajectory_length = len(self.trajectory_list[self.index[0]])
        self.startpoint = self.index[1]
        data_list = self.trajectory_list[self.index[0]][self.startpoint: self.startpoint + self.sequence_length]
        copy = list()
        for x in data_list:
            data = deepcopy(x)
            copy.append(self.preprocessing(data))

        return copy


class RegularDataset(Dataset):
    """
        Implements a dataset containing Graphs
    """
    def __init__(self, trajectory_list: list, preprocessing: Callable):
        """
        Args:
            trajectory_list: List of lists of (PyG) data objects
            sequence_length: Length of the drawn sequence from a trajectory
        """
        self.trajectory_list = trajectory_list
        self.preprocessing = preprocessing
        self.indices = []
        self.indices.extend(range(len(self.trajectory_list)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        """
        Args:
            idx: Index from sampler or batch_sampler of the Dataloader
        Returns:
            data: Data element from the trajectory list at index 'idx'
        """
        self.index = self.indices[idx]
        data = self.trajectory_list[self.index]
        copy = deepcopy(data)

        return self.preprocessing(copy)
