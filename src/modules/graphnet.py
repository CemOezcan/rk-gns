from typing import Callable, List

import torch

from torch import nn, Tensor

from src.util import util
from src.util.util import device
from src.util.types import MultiGraph, EdgeSet


class GraphNet(nn.Module):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, model_fn: Callable, output_size: int, message_passing_aggregator: str, edge_sets: List[str]):
        super().__init__()

        self.node_model_cross = model_fn(output_size)
        self.edge_models = nn.ModuleDict({name: model_fn(output_size) for name in edge_sets})

        self.message_passing_aggregator = message_passing_aggregator

    def _update_edge_features(self, node_features: List[Tensor], edge_set: EdgeSet) -> Tensor:
        """Aggregrates node features, and applies edge function."""
        node_features = torch.cat(tuple(node_features), dim=0)
        senders = edge_set.senders.to(device)
        receivers = edge_set.receivers.to(device)

        sender_features = torch.index_select(input=node_features, dim=0, index=senders)
        receiver_features = torch.index_select(input=node_features, dim=0, index=receivers)
        features = torch.cat([sender_features, receiver_features, edge_set.features], dim=-1)

        return torch.add(edge_set.features, self.edge_models[edge_set.name](features))

    def _update_node_features(self, graph: MultiGraph, edge_sets: List[EdgeSet]):
        """Aggregrates edge features, and applies node function."""
        node_features = graph.node_features
        hyper_node_offset = len(node_features[0])
        node_features = torch.cat(tuple(node_features), dim=0)
        num_nodes = node_features.shape[0]
        features = [node_features]

        features = self.aggregation(
            list(filter(lambda x: x.name in self.edge_models.keys(), edge_sets)),
            features,
            num_nodes
        )
        updated_nodes_cross = self.node_model_cross(features[:hyper_node_offset])
        graph.node_features[0] = torch.add(updated_nodes_cross, graph.node_features[0])

    def aggregation(self, edge_sets: List[EdgeSet], features: List[Tensor], num_nodes: int) -> Tensor:
        for edge_set in edge_sets:
            if self.message_passing_aggregator == 'pna':
                features.append(
                    util.unsorted_segment_operation(edge_set.features, edge_set.receivers,
                                                    num_nodes, operation='sum'))
                features.append(
                    util.unsorted_segment_operation(edge_set.features, edge_set.receivers,
                                                    num_nodes, operation='mean'))
                features.append(
                    util.unsorted_segment_operation(edge_set.features, edge_set.receivers,
                                                    num_nodes, operation='max'))
                features.append(
                    util.unsorted_segment_operation(edge_set.features, edge_set.receivers,
                                                    num_nodes, operation='min'))
            else:
                features.append(
                    util.unsorted_segment_operation(edge_set.features, edge_set.receivers, num_nodes,
                                                    operation=self.message_passing_aggregator))

        return torch.cat(features, dim=-1)

    def forward(self, graph: MultiGraph, mask=None) -> MultiGraph:
        """Applies GraphNetBlock and returns updated MultiGraph."""
        # apply edge functions
        new_edge_sets = []
        for edge_set in graph.edge_sets:
            updated_features = self._update_edge_features(graph.node_features, edge_set)
            new_edge_sets.append(edge_set._replace(features=updated_features))

        # apply node function
        graph = graph._replace(edge_sets=new_edge_sets)
        self._update_node_features(graph, new_edge_sets)

        return graph

    def perform_edge_updates(self, graph, edge_set_name, new_edge_sets):
        if edge_set_name not in self.edge_models.keys():
            return

        edge_set = list(filter(lambda x: x.name == edge_set_name, graph.edge_sets))[0]
        updates_mesh_features = self._update_edge_features(graph.node_features, edge_set)
        new_edge_sets[edge_set_name] = edge_set._replace(features=updates_mesh_features)

    def _update_hyper_node_features(self, graph: MultiGraph, edge_sets: List[EdgeSet], model: nn.Module):
        """Aggregrates edge features, and applies node function."""
        node_features = graph.node_features
        hyper_node_offset = len(node_features[0])
        node_features = torch.cat(tuple(node_features), dim=0)
        num_nodes = node_features.shape[0]
        features = [node_features]

        features = self.aggregation(
            list(filter(lambda x: x.name in self.edge_models.keys(), edge_sets)),
            features,
            num_nodes
        )
        updated_nodes = model(features[hyper_node_offset:])
        graph.node_features[1] = torch.add(updated_nodes, graph.node_features[1])

    def _update_down(self, graph: MultiGraph, edge_sets: List[EdgeSet]):
        """Aggregrates edge features, and applies node function."""
        node_features = graph.node_features
        hyper_node_offset = len(node_features[0])
        node_features = torch.cat(tuple(node_features), dim=0)
        num_nodes = node_features.shape[0]
        features = [node_features]

        features = self.aggregation(
            list(filter(lambda x: x.name in self.edge_models.keys(), edge_sets)),
            features,
            num_nodes
        )
        updated_nodes_cross = self.node_model_down(features[:hyper_node_offset])
        graph.node_features[0] = torch.add(updated_nodes_cross, graph.node_features[0])
