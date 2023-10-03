import torch

from torch_geometric.data import Batch
from typing import Callable, Union, Tuple
from torch import nn, Tensor

from src.util.types import NodeType


class Decoder(nn.Module):
    """Decodes node features from graph."""

    def __init__(self, make_mlp: Callable, output_size: int, node_type: str, latent_size: int, recurrence: bool, self_sup: bool = False):
        super().__init__()
        self.node_type = node_type
        self.recurrence = recurrence
        self.latent_size = latent_size
        self.self_sup = self_sup

        self.use_u = output_size == 1

        if self.recurrence:
            self.lstm = nn.GRUCell(self.latent_size, self.latent_size)

        self.model = nn.Sequential(nn.LazyLinear(latent_size), nn.LeakyReLU(), nn.Linear(latent_size, output_size))
        if self.self_sup:
            self.gl_model = nn.Sequential(nn.LazyLinear(latent_size), nn.LeakyReLU(), nn.Linear(latent_size, latent_size))

    def forward(self, graph: Batch) -> Tuple[Tensor, Union[None, Tensor]]:
        mask = torch.where(graph[self.node_type].node_type == NodeType.MESH)[0]
        if self.self_sup:
            graph.u = self.gl_model(graph.u)
        node_features = graph.u if self.use_u else graph[self.node_type].x[mask]

        if not self.use_u and self.self_sup:
            batch = graph[self.node_type].batch
            node_features = torch.cat([node_features, graph.u[batch][mask]], dim=-1)

        if self.recurrence:
            if graph.h.shape[-1] == self.latent_size:
                hidden = self.lstm(node_features, graph.h)
            else:
                hidden = self.lstm(node_features)

            out = self.model(hidden)
        else:
            out = self.model(node_features)
            hidden = out

        return out, hidden