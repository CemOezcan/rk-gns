from typing import Callable, List, Type

from torch import nn
from torch_geometric.data import HeteroData

from src.modules.graphnet import GraphNet
from src.util.types import MultiGraph


class Processor(nn.Module):
    """
    The Graph Neural Network that transforms the input graph.
    """

    def __init__(self, make_mlp: Callable, output_size: int, message_passing_steps: int,
                 message_passing_aggregator: str, node_sets: List[str], edge_sets: List[str], graphnet_block: Type[GraphNet]):
        super().__init__()
        blocks = []
        for _ in range(message_passing_steps):
            blocks.append(
                graphnet_block(model_fn=make_mlp, output_size=output_size,
                               message_passing_aggregator=message_passing_aggregator,
                               node_sets=node_sets,  edge_sets=edge_sets
                               )
            )
        self.graphnet_blocks = nn.Sequential(*blocks)

    def forward(self, latent_graph: HeteroData) -> HeteroData:
        return self.graphnet_blocks(latent_graph)