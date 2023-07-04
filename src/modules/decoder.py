from typing import Callable, Union, Tuple

from torch import nn, Tensor
from torch_geometric.data import Batch
import torch

from src.util.types import NodeType


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
        mask = torch.where(graph.node_type == NodeType.MESH)[0]
        node_features = graph[self.node_type].x[mask]

        if self.recurrence:
            if graph.h.shape[-1] == self.latent_size:
                hidden = self.lstm(node_features.view(-1, 1, self.latent_size), (graph.h, graph.c))
            else:
                hidden = self.lstm(node_features.view(-1, 1, self.latent_size))

            out, hidden = (torch.squeeze(hidden[0], dim=1), hidden[1])
            out = self.model(out)
        else:
            out, hidden = self.model(node_features), None

        return out, hidden
