from typing import Callable, Union, Tuple

from torch import nn, Tensor
from torch_geometric.data import Batch
import torch

from src.util.types import NodeType


class Decoder(nn.Module):
    """Decodes node features from graph."""

    def __init__(self, make_mlp: Callable, output_size: int, node_type: str, latent_size: int, recurrence: bool):
        super().__init__()
        self.model = nn.Linear(self.latent_size, output_size) # make_mlp(output_size)
        self.node_type = node_type
        self.recurrence = recurrence
        self.latent_size = latent_size

        if self.recurrence:
            self.lstm = nn.GRUCell(self.latent_size, self.latent_size)

    def forward(self, graph: Batch) -> Tuple[Tensor, Union[None, Tensor]]:
        mask = torch.where(graph.node_type == NodeType.MESH)[0]
        node_features = graph[self.node_type].x[mask]

        if self.recurrence:
            if graph.h.shape[-1] == self.latent_size:
                hidden = self.lstm(node_features, graph.h)
            else:
                hidden = self.lstm(node_features)

            out = self.model(hidden)
        else:
            out, hidden = self.model(node_features), None

        return out, hidden
