from typing import Callable, List

import torch

from torch import nn, Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import scatter


class GraphNet(nn.Module):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, model_fn: Callable, output_size: int, message_passing_aggregator: str,
                 node_sets: List[str], edge_sets: List[str], use_global: bool = True):
        super().__init__()

        self.node_models = nn.ModuleDict({name: model_fn(output_size) for name in node_sets})
        self.edge_models = nn.ModuleDict({name: model_fn(output_size) for name in edge_sets})
        self.global_model = model_fn(output_size) if use_global else None

        self._use_global = use_global
        self.message_passing_aggregator = message_passing_aggregator

    def _update_edges(self, graph: Batch):
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

            edge_store['edge_attr'] = torch.add(edge_attr, self.edge_models["".join(edge_type)](aggregated_features))

    def _update_nodes(self, graph: Batch):
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

            node_store["x"] = torch.add(node_features, self.node_models[node_type](aggregated_features))

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
        graph.u = torch.add(graph.u, self.global_model(aggregated_features))

    def aggregation(self, edge_features, indices, num_nodes: int) -> Tensor:
        if self.message_passing_aggregator == 'pna':
            latent_dimension = edge_features.shape[1]
            reduced = torch.zeros((num_nodes, latent_dimension * 4), device=edge_features.device)

            for position, reducer in enumerate(['sum', 'mean', 'max', 'min']):
                reduced[:, edge_features.shape[1] * position:edge_features.shape[1] * (position + 1)] = \
                    self.unsorted_segment_operation(edge_features, indices, num_nodes, operation=reducer)
        else:
            reduced = self.unsorted_segment_operation(edge_features, indices, num_nodes, operation=self.message_passing_aggregator)

        return reduced

    def forward(self, graph: Batch) -> Batch:
        """Applies GraphNetBlock and returns updated MultiGraph."""

        self._update_edges(graph)
        self._update_nodes(graph)
        if self._use_global:
            self._update_global(graph)

        return graph

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

        if operation == 'sum':
            result = scatter(data.float(), segment_ids, dim=0, dim_size=num_segments, reduce='add')
        elif operation == 'max':
            result = scatter(data.float(), segment_ids, dim=0, dim_size=num_segments, reduce='max')
        elif operation == 'mean':
            result = scatter(data.float(), segment_ids, dim=0, dim_size=num_segments, reduce='mean')
        elif operation == 'min':
            result = scatter(data.float(), segment_ids, dim=0, dim_size=num_segments, reduce='min')
        elif operation == 'mul':
            result = scatter(data.float(), segment_ids, dim=0, dim_size=num_segments, reduce='mul')
        else:
            raise Exception('Invalid operation type!')

        result = result.type(data.dtype)

        return result

