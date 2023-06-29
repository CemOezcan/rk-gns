
import functools

from collections import OrderedDict
from typing import List, Type, Tuple

from torch import nn, Tensor
from torch_geometric.data import Batch

from src.modules.encoder import Encoder
from src.modules.graphnet import GraphNet
from src.modules.processor import Processor
from src.modules.decoder import Decoder
from src.util.util import device


class MeshGraphNets(nn.Module):
    """Encode-Process-Decode GraphNet model."""

    def __init__(self, output_size: int, latent_size: int, num_layers: int, message_passing_aggregator: str,
                 message_passing_steps: int, node_sets: List[str], edge_sets: List[str], dec: str,
                 use_global: bool, recurrence: bool):
        super().__init__()
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._message_passing_steps = message_passing_steps
        self._message_passing_aggregator = message_passing_aggregator
        graphnet_block = GraphNet

        self.encoder = Encoder(make_mlp=self._make_mlp,
                               latent_size=self._latent_size,
                               node_sets=node_sets,
                               edge_sets=edge_sets,
                               use_global=use_global)
        self.processor = Processor(make_mlp=self._make_mlp, output_size=self._latent_size,
                                   message_passing_steps=self._message_passing_steps,
                                   message_passing_aggregator=self._message_passing_aggregator,
                                   node_sets=node_sets,
                                   edge_sets=edge_sets,
                                   graphnet_block=graphnet_block,
                                   use_global=use_global)
        self.decoder = Decoder(make_mlp=functools.partial(self._make_mlp, layer_norm=False),
                               output_size=self._output_size, node_type=dec, latent_size=latent_size, recurrence=recurrence)

    def forward(self, graph: Batch) -> Tensor:
        """Encodes and processes a multigraph, and returns node features."""
        latent_graph = self.encoder(graph)
        latent_graph = self.processor(latent_graph)
        return self.decoder(latent_graph)

    def _make_mlp(self, output_size: int, layer_norm=True) -> nn.Module:
        """Builds an MLP."""
        widths = [self._latent_size] * self._num_layers + [output_size]
        network = LazyMLP(widths)
        if layer_norm:
            network = nn.Sequential(
                network, nn.LayerNorm(normalized_shape=widths[-1]))
        return network


# TODO refactor into new file
class LazyMLP(nn.Module):
    def __init__(self, output_sizes: List[int]):
        super().__init__()
        num_layers = len(output_sizes)
        self._layers_ordered_dict = OrderedDict()
        for index, output_size in enumerate(output_sizes):
            self._layers_ordered_dict["linear_" +
                                      str(index)] = nn.LazyLinear(output_size)
            if index < (num_layers - 1):
                self._layers_ordered_dict["relu_" + str(index)] = nn.ReLU()
        self.layers = nn.Sequential(self._layers_ordered_dict)

    def forward(self, input: Tensor) -> Tensor:
        input = input.to(device)
        y = self.layers(input)
        return y
