from typing import Callable

from torch import nn, Tensor
from torch_geometric.data import Batch


class Decoder(nn.Module):
    """Decodes node features from graph."""

    def __init__(self, make_mlp: Callable, output_size: int, node_type: str):
        super().__init__()
        self.model = make_mlp(output_size)
        self.node_type = node_type

    def forward(self, graph: Batch) -> Tensor:
        return self.model(graph[self.node_type].x)
