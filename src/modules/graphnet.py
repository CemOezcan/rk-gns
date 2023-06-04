from typing import Callable, List

import torch

from torch import nn, Tensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter

from src.util.util import device
from src.util.types import MultiGraph, EdgeSet


class GraphNet(nn.Module):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, model_fn: Callable, output_size: int, message_passing_aggregator: str, node_sets: List[str], edge_sets: List[str]):
        super().__init__()

        self.node_models = nn.ModuleDict({name: model_fn(output_size) for name in node_sets})
        self.edge_models = nn.ModuleDict({name: model_fn(output_size) for name in edge_sets})
        self.global_model = model_fn(output_size)

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

    def _update_edges(self, graph: HeteroData):
        for position, (edge_type, edge_store) in enumerate(zip(graph.edge_types, graph.edge_stores)):
            edge_attr = edge_store.get("edge_attr")
            edge_indices = edge_store.get("edge_index")
            source_indices, dest_indices = edge_indices

            source_node_type, _, dest_node_type = edge_type
            source_node_index = graph.node_types.index(source_node_type)
            dest_node_index = graph.node_types.index(dest_node_type)

            edge_source_nodes = graph.node_stores[source_node_index]["x"][source_indices]
            edge_dest_nodes = graph.node_stores[dest_node_index]["x"][dest_indices]

            # concatenate everything
            aggregated_features = torch.cat([edge_source_nodes, edge_dest_nodes, edge_attr], 1)
            # global
            indices = graph[source_node_type].batch
            global_features = graph.u[indices[source_indices]]
            aggregated_features = torch.cat([aggregated_features, global_features], 1)

            edge_store["edge_attr"] = torch.add(edge_attr, self.edge_models["".join(edge_type)](aggregated_features))

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

    def _update_nodes(self, graph: HeteroData):
        node_features = graph.node_stores[0].get('x')
        num_nodes = node_features.shape[0]
        edge_features = self.aggregation_2(graph, num_nodes)

        for position, (node_type, node_store) in enumerate(zip(graph.node_types, graph.node_stores)):
            node_features = node_store.get('x')
            aggregated_features = torch.cat([node_features, edge_features], 1)
            # global
            batch = graph[node_type].batch
            aggregated_features = torch.cat([aggregated_features, graph.u[batch]], 1)
            # update
            node_store["x"] = torch.add(node_features, self.node_models[node_type](aggregated_features))

    def _update_global(self, graph: HeteroData):
        edge_feature_list = []
        node_feature_list = []
        for edge_type, edge_store in zip(graph.edge_types, graph.edge_stores):
            edge_attr = edge_store.get("edge_attr")
            edge_indices = edge_store.get("edge_index")
            source_indices, _ = edge_indices
            source_node_type, _, _ = edge_type
            indices = graph[source_node_type].batch
            edge_feature_list = self.one_step_aggregation(edge_feature_list, edge_attr, indices[source_indices], graph.u.shape[0])

        for node_type, node_store in zip(graph.node_types, graph.node_stores):
            node_attr = node_store.get("x")
            node_feature_list = self.one_step_aggregation(node_feature_list, node_attr, graph[node_type].batch, graph.u.shape[0])

        aggregated_edge_features = torch.cat(edge_feature_list, 1)
        aggregated_node_features = torch.cat(node_feature_list, 1)

        aggregated_features = torch.cat([aggregated_node_features, aggregated_edge_features, graph.u], 1)
        graph.u = torch.add(graph.u, self.global_model(aggregated_features))

    def aggregation(self, edge_sets: List[EdgeSet], features: List[Tensor], num_nodes: int) -> Tensor:
        for edge_set in edge_sets:
            if self.message_passing_aggregator == 'pna':
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers,
                                                    num_nodes, operation='sum'))
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers,
                                                    num_nodes, operation='mean'))
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers,
                                                    num_nodes, operation='max'))
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers,
                                                    num_nodes, operation='min'))
            else:
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers, num_nodes,
                                                    operation=self.message_passing_aggregator))

        return torch.cat(features, dim=-1)

    def one_step_aggregation(self, features, edge_features, indices, num_nodes: int) -> Tensor:
        if self.message_passing_aggregator == 'pna':
            features.append(
                self.unsorted_segment_operation(edge_features, indices,
                                                num_nodes, operation='sum'))
            features.append(
                self.unsorted_segment_operation(edge_features, indices,
                                                num_nodes, operation='mean'))
            features.append(
                self.unsorted_segment_operation(edge_features, indices,
                                                num_nodes, operation='max'))
            features.append(
                self.unsorted_segment_operation(edge_features, indices,
                                                num_nodes, operation='min'))
        else:
            features.append(
                self.unsorted_segment_operation(edge_features, indices,
                                                num_nodes, operation=self.message_passing_aggregator))

        return features

    def aggregation_2(self, graph, num_nodes: int) -> Tensor:
        features = list()
        for position, (edge_type, edge_store) in enumerate(zip(graph.edge_types, graph.edge_stores)):
            if self.message_passing_aggregator == 'pna':
                features.append(
                    self.unsorted_segment_operation(edge_store.get('edge_attr'), edge_store.get('edge_index')[1],
                                                    num_nodes, operation='sum'))
                features.append(
                    self.unsorted_segment_operation(edge_store.get('edge_attr'), edge_store.get('edge_index')[1],
                                                    num_nodes, operation='mean'))
                features.append(
                    self.unsorted_segment_operation(edge_store.get('edge_attr'), edge_store.get('edge_index')[1],
                                                    num_nodes, operation='max'))
                features.append(
                    self.unsorted_segment_operation(edge_store.get('edge_attr'), edge_store.get('edge_index')[1],
                                                    num_nodes, operation='min'))
            else:
                features.append(
                    self.unsorted_segment_operation(edge_store.get('edge_attr'), edge_store.get('edge_index')[1],
                                                    num_nodes, operation=self.message_passing_aggregator))

        return torch.cat(features, dim=-1)

    def forward(self, graph: HeteroData) -> HeteroData:
        """Applies GraphNetBlock and returns updated MultiGraph."""

        self._update_edges(graph)
        self._update_nodes(graph)
        self._update_global(graph)

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

    @staticmethod
    def unsorted_segment_operation(data, segment_ids, num_segments, operation):
        """
        Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

        :param data: A tensor whose segments are to be summed.
        :param segment_ids: The segment indices tensor.
        :param num_segments: The number of segments.
        :return: A tensor of same data type as the data argument.
        """
        assert all([i in data.shape for i in segment_ids.shape]
                   ), "segment_ids.shape should be a prefix of data.shape"

        shape = [num_segments] + list(data.shape[1:])
        result = torch.zeros(*shape).to(device)

        if operation == 'sum':
            result = scatter(data.float(), segment_ids, dim=0, dim_size=num_segments, reduce='add')
        elif operation == 'max':
            result = scatter(data.float(), segment_ids, dim=0, dim_size=num_segments, reduce='max')
        elif operation == 'mean':
            result = scatter(data.float(), segment_ids, dim=0, dim_size=num_segments, reduce='mean')
        elif operation == 'min':
            result = scatter(data.float(), segment_ids, dim=0, dim_size=num_segments, reduce='min')
        elif operation == 'std':
            result = scatter(data.float(), segment_ids, dim=0, dim_size=num_segments, reduce='mul')
        else:
            raise Exception('Invalid operation type!')

        result = result.type(data.dtype)

        return result

