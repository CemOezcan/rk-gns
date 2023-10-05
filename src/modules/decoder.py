import torch

from torch_geometric.data import Batch
from typing import Callable, Union, Tuple
from torch import nn, Tensor

from src.util.types import NodeType


class Decoder(nn.Module):
    """Decodes node features from graph."""

    def __init__(self, output_size: int, node_type: str, latent_size: int, recurrence: bool, self_sup: bool = False):
        super().__init__()
        self.node_type = node_type
        self.recurrence = recurrence
        self.latent_size = latent_size
        self.self_sup = self_sup
        self.use_u = output_size == 1

        self.model = nn.Sequential(nn.LazyLinear(latent_size), nn.LeakyReLU(), nn.Linear(latent_size, output_size))

        if self.recurrence:
            self.rnn = nn.GRUCell(self.latent_size, self.latent_size)

    def forward(self, graph: Batch) -> Tuple[Tensor, Union[None, Tensor]]:
        self.transform_global(graph)
        x_hat = self.transform_nodes(graph)
        y_hat = self.model(x_hat)

        return y_hat, graph.u

    def transform_global(self, graph):
        if self.recurrence:
            if graph.h.shape[-1] == self.latent_size:
                graph.u = self.rnn(graph.u, graph.h)
            else:
                graph.u = self.rnn(graph.u)

    def transform_nodes(self, graph):
        if self.use_u:
            return graph.u

        mask = torch.where(graph[self.node_type].node_type == NodeType.MESH)[0]
        node_features = graph[self.node_type].x[mask]

        if self.self_sup:
            batch = graph[self.node_type].batch
            return torch.cat([node_features, graph.u[batch][mask]], dim=-1)

        return node_features
