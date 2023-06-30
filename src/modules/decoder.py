from typing import Callable, Union, Tuple

from torch import nn, Tensor
from torch_geometric.data import Batch
import torch


class Decoder(nn.Module):
    """Decodes node features from graph."""

    def __init__(self, make_mlp: Callable, output_size: int, node_type: str, latent_size: int, recurrence: bool):
        super().__init__()
        self.model = make_mlp(output_size)
        self.node_type = node_type
        self.recurrence = recurrence
        self.latent_size = latent_size

        if self.recurrence:
            self.lstm = nn.LSTM(self.latent_size, self.latent_size, 1, batch_first=True)

    def forward(self, graph: Batch) -> Tuple[Tensor, Union[None, Tensor]]:
        if self.recurrence:
            if graph.h.shape[-1] == self.latent_size:
                hidden = self.lstm(graph.u.view(-1, 1, self.latent_size), (graph.h, graph.c))
            else:
                hidden = self.lstm(graph.u.view(-1, 1, self.latent_size))

            hidden = (torch.squeeze(hidden[0]), hidden[1])
        else:
            hidden = None

        return self.model(graph[self.node_type].x), hidden
