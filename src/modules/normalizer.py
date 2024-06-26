import torch
import src.util.util

from torch import nn, Tensor

class Normalizer(nn.Module):
    """
    Feature normalizer that accumulates statistics online.
    """

    def __init__(self, name: str, device=None, max_accumulations=10 ** 6, std_epsilon=1e-8) -> None:
        """
        Initialize the normalizer.

        Parameters
        ----------
            size : int
                Dimensionality

            name : str
                Normalizer name

            max_accumulations : int
                Maximum number of accumulated values

            std_epsilon : float
                Epsilon for numerical stability
        """
        super(Normalizer, self).__init__()
        self._name = name
        self._max_accumulations = max_accumulations
        if device is None:
            self.device = src.util.util.device
        else:
            self.device = device
        self._std_epsilon = torch.tensor([std_epsilon], requires_grad=False).to(self.device)

        self._acc_count = torch.zeros(1, dtype=torch.float32, requires_grad=False).to(self.device)
        self._num_accumulations = torch.zeros(1, dtype=torch.float32, requires_grad=False).to(self.device)

        self.initialized = False
        self._acc_sum = None
        self._acc_sum_squared = None

    def initialize(self, size):
        self._acc_sum = torch.zeros(size, dtype=torch.float32, requires_grad=False).to(self.device)
        self._acc_sum_squared = torch.zeros(size, dtype=torch.float32, requires_grad=False).to(self.device)

        self.initialized = True

    def forward(self, batched_data: Tensor, accumulate=True) -> Tensor:
        """Normalizes input data and accumulates statistics."""
        if not self.initialized:
            self.initialize(batched_data.shape[1])

        if accumulate and self._num_accumulations < self._max_accumulations:
            # stop accumulating after a million updates, to prevent accuracy issues
            self._accumulate(batched_data)
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data: Tensor) -> Tensor:
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data: Tensor) -> None:
        """Function to perform the accumulation of the batch_data statistics."""
        count = torch.tensor(
            batched_data.shape[0], dtype=torch.float32, device=self.device)

        data_sum = torch.sum(batched_data, dim=0)
        squared_data_sum = torch.sum(batched_data ** 2, dim=0)
        self._acc_sum = self._acc_sum.add(data_sum)
        self._acc_sum_squared = self._acc_sum_squared.add(squared_data_sum)
        self._acc_count = self._acc_count.add(count)
        self._num_accumulations = self._num_accumulations.add(1.)

    def _mean(self) -> float:
        safe_count = torch.maximum(self._acc_count, torch.tensor([1.], device=self.device))
        return self._acc_sum / safe_count

    def _std_with_epsilon(self) -> float:
        safe_count = torch.maximum(self._acc_count, torch.tensor([1.], device=self.device))
        std = torch.sqrt(torch.abs(self._acc_sum_squared / safe_count - self._mean() ** 2))
        return torch.maximum(std, self._std_epsilon)

    def get_acc_sum(self) -> float:
        return self._acc_sum
