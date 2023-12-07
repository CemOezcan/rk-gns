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
        self.pc_node_models = nn.ModuleDict({name: model_fn(output_size) for name in node_sets})
        self.pc_edge_models = nn.ModuleDict({name: model_fn(output_size) for name in edge_sets})
        self._use_global = False

    def _update_global(self, graph: Batch):
        edge_feature_list = []
        node_feature_list = []
        for edge_type, edge_store in zip(graph.edge_types, graph.edge_stores):
            edge_attr = edge_store.get('edge_attr')
            source_indices, _ = edge_store.get('edge_index')
            source_node_type, _, _ = edge_type
            indices = graph[source_node_type].batch
            edge_feature_list.append(self.aggregation(edge_attr, indices[source_indices], graph.u.shape[0]))

        for node_type, node_store in zip(graph.node_types, graph.node_stores):
            node_attr = node_store.get('x')
            node_feature_list.append(self.aggregation(node_attr, graph[node_type].batch, graph.u.shape[0]))

        aggregated_edge_features = torch.cat(edge_feature_list, 1)
        aggregated_node_features = torch.cat(node_feature_list, 1)

        aggregated_features = torch.cat([aggregated_node_features, aggregated_edge_features, graph.u], 1)
        graph.u = self.global_model(aggregated_features)

    def _update_pc_edges(self, graph: Batch):
        for position, (edge_type, edge_store) in enumerate(zip(graph.edge_types, graph.edge_stores)):
            edge_attr = edge_store.get('edge_attr')
            edge_indices = edge_store.get('edge_index')
            source_indices, dest_indices = edge_indices

            source_node_type, _, dest_node_type = edge_type
            source_node_index = graph.node_types.index(source_node_type)
            dest_node_index = graph.node_types.index(dest_node_type)

            edge_source_nodes = graph.node_stores[source_node_index]['x'][source_indices]
            edge_dest_nodes = graph.node_stores[dest_node_index]['x'][dest_indices]

            # concatenate everything
            aggregated_features = torch.cat([edge_source_nodes, edge_dest_nodes, edge_attr], 1)
            # global
            if self._use_global:
                indices = graph[source_node_type].batch
                global_features = graph.u[indices[source_indices]]
                aggregated_features = torch.cat([aggregated_features, global_features], 1)

            edge_store['edge_attr'] = torch.add(edge_attr, self.pc_edge_models["".join(edge_type)](aggregated_features))

    def _update_pc_nodes(self, graph: Batch):
        num_nodes = sum([node_store.x.shape[0] for node_store in graph.node_stores])

        edge_features = list()
        for edge_type, edge_store in zip(graph.edge_types, graph.edge_stores):
            edge_features.append(
                self.aggregation(edge_store.get('edge_attr'), edge_store.get('edge_index')[1], num_nodes))

        edge_features = torch.cat(edge_features, dim=-1)

        for position, (node_type, node_store) in enumerate(zip(graph.node_types, graph.node_stores)):
            node_features = node_store.get('x')
            aggregated_features = torch.cat([node_features, edge_features], 1)

            if self._use_global:
                batch = graph[node_type].batch
                aggregated_features = torch.cat([aggregated_features, graph.u[batch]], 1)

            node_store["x"] = torch.add(node_features, self.pc_node_models[node_type](aggregated_features))

    @staticmethod
    def split_graphs(graph):
        if isinstance(graph, tuple):
            return graph[0], graph[1]
        pc_mask = torch.where(graph['mesh'].node_type == NodeType.SHAPE)[0]
        obst_mask = torch.where(graph['mesh'].node_type == NodeType.COLLIDER)[0]
        mesh_mask = torch.where(graph['mesh'].node_type == NodeType.MESH)[0]

        poisson_mask = torch.cat([pc_mask, obst_mask], dim=0)
        mesh_mask = torch.cat([mesh_mask, obst_mask], dim=0)
        mesh = graph.subgraph({'mesh': mesh_mask})
        pc = graph.subgraph({'mesh': poisson_mask})

        return mesh, pc

    def forward(self, graph: Batch) -> Batch:
        """Applies GraphNetBlock and returns updated MultiGraph."""
        mesh, pc = self.split_graphs(graph)
        # TODO: This does only one mp step
        self._update_edges(mesh)
        self._update_nodes(mesh)
        self._update_pc_edges(pc)
        self._update_pc_nodes(pc)
        self._update_global(pc)
        return mesh, pc