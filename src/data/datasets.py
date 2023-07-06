import itertools

from torch.utils.data import Dataset


class SequenceNoReturnDataset(Dataset):
    """
        Dataset that draws sequences of specific length from a list of trajectories without replacement.
        In this case, we can still define a training epoch, if all samples are used once.
    """
    def __init__(self, trajectory_list: list, sequence_length: int):
        """
        Args:
            trajectory_list: List of lists of (PyG) data objects
            sequence_length: Length of the drawn sequence from a trajectory
        """
        self.trajectory_list = trajectory_list
        self.sequence_length = sequence_length

        # create index list of tuples (i, t_i), where i indicates the index for the trajectory and t_i for the starting time step of the sequence
        self.indices = []
        for trajectory in range(len(trajectory_list)):
            self.indices.extend([(i, t_i) for i, t_i in itertools.product([trajectory], range(len(trajectory_list[trajectory])-self.sequence_length))])

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
        data_list = self.trajectory_list[self.index[0]][self.startpoint:self.startpoint+self.sequence_length]
        return data_list

