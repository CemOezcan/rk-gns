from typing import Callable, List, Tuple

import torch
from torch import nn
from torch_geometric.data import HeteroData

from src.util.types import MultiGraph


class Encoder(nn.Module):
    """Encodes node and edge features into latent features."""

    def __init__(self, make_mlp: Callable, latent_size: int, node_sets: List[str], edge_sets: List[str], use_global: bool = True):
        super().__init__()
        self._make_mlp = make_mlp
        self._latent_size = latent_size

        self.node_models = nn.ModuleDict({name: self._make_mlp(latent_size) for name in node_sets})
        self.edge_models = nn.ModuleDict({name: self._make_mlp(latent_size) for name in edge_sets})
        self.global_model = self._make_mlp(latent_size) if use_global else lambda x: x

    def forward(self, graph: HeteroData) -> HeteroData:
        for position, node_type in enumerate(graph.node_types):
            graph.node_stores[position]["x"] = self.node_models[node_type](
                graph.node_stores[position]["x"])

        for position, edge_type in enumerate(graph.edge_types):
            graph.edge_stores[position]["edge_attr"] = self.edge_models[''.join(edge_type)](
                graph.edge_stores[position]["edge_attr"])

        graph.u = self.global_model(graph.u)

        return graph
