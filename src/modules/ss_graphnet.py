import torch

from typing import Callable, List
from torch import nn, Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from src.modules.graphnet import GraphNet
from src.util import test
from src.util.types import NodeType


class SSGraphNet(GraphNet):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, model_fn: Callable, output_size: int, message_passing_aggregator: str,
                 node_sets: List[str], edge_sets: List[str], use_global: bool = True, poisson: bool = False):
        super(SSGraphNet, self).__init__(model_fn, output_size, message_passing_aggregator, node_sets, edge_sets, use_global, poisson)
        self.global_model = model_fn(output_size)
        self._use_global = False

    def _update_global(self, graph: Batch):
        edge_feature_list = []
        node_feature_list = []
        pc = self.split_graphs(graph)
        for edge_type, edge_store in zip(pc.edge_types, pc.edge_stores):
            edge_attr = edge_store.get('edge_attr')
            source_indices, _ = edge_store.get('edge_index')
            source_node_type, _, _ = edge_type
            indices = pc[source_node_type].batch
            edge_feature_list.append(self.aggregation(edge_attr, indices[source_indices], pc.u.shape[0]))

        for node_type, node_store in zip(pc.node_types, pc.node_stores):
            node_attr = node_store.get('x')
            node_feature_list.append(self.aggregation(node_attr, pc[node_type].batch, pc.u.shape[0]))

        aggregated_edge_features = torch.cat(edge_feature_list, 1)
        aggregated_node_features = torch.cat(node_feature_list, 1)

        aggregated_features = torch.cat([aggregated_node_features, aggregated_edge_features, pc.u], 1)
        graph.u = self.global_model(aggregated_features)

    @staticmethod
    def split_graphs(graph):
        pc_mask = torch.where(graph['mesh'].node_type == NodeType.SHAPE)[0]
        obst_mask = torch.where(graph['mesh'].node_type == NodeType.COLLIDER)[0]

        poisson_mask = torch.cat([pc_mask, obst_mask], dim=0)
        pc = graph.subgraph({'mesh': poisson_mask})

        return pc

    def forward(self, graph: Batch) -> Batch:
        """Applies GraphNetBlock and returns updated MultiGraph."""

        self._update_edges(graph)
        self._update_nodes(graph)
        self._update_global(graph)

        return graph