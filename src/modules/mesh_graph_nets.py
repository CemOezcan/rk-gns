import functools

from collections import OrderedDict
from typing import List, Type, Tuple, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch_geometric.data import Batch

from src.modules.encoder import Encoder
from src.modules.graphnet import GraphNet
from src.modules.processor import Processor
from src.modules.decoder import Decoder
from src.modules.ss_graphnet import SSGraphNet
from src.util.util import device


class MeshGraphNets(nn.Module):
    """Encode-Process-Decode GraphNet model."""

    def __init__(self, output_size: int, latent_size: int, num_layers: int, message_passing_aggregator: str,
                 message_passing_steps: int, node_sets: List[str], edge_sets: List[str], dec: str,
                 use_global: bool, recurrence: Union[bool, str], layer_norm: bool = False, self_sup: bool = False):
        super().__init__()
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._message_passing_steps = message_passing_steps
        self._message_passing_aggregator = message_passing_aggregator
        self.self_sup = self_sup
        graphnet_block = SSGraphNet if self_sup else GraphNet

        self.encoder = Encoder(make_mlp=nn.LazyLinear,
                               latent_size=self._latent_size,
                               node_sets=node_sets,
                               edge_sets=edge_sets,
                               use_global=use_global)
        self.processor = Processor(make_mlp=functools.partial(self._make_mlp, layer_norm=layer_norm), output_size=self._latent_size,
                                   message_passing_steps=self._message_passing_steps,
                                   message_passing_aggregator=self._message_passing_aggregator,
                                   node_sets=node_sets,
                                   edge_sets=edge_sets,
                                   graphnet_block=graphnet_block,
                                   use_global=use_global,
                                   poisson=self._output_size == 1)
        self.decoder = Decoder(output_size=self._output_size, node_type=dec, latent_size=latent_size, rnn_type=recurrence, self_sup=self_sup)

    def forward(self, graph: Batch) -> Tensor:
        """Encodes and processes a multigraph, and returns node features."""
        latent_graph = self.encoder(graph)
        latent_graph = self.processor(latent_graph)
        if self.self_sup:
            mesh, pc = latent_graph
            mesh.u = pc.u
            latent_graph = mesh
        return self.decoder(latent_graph)

    def _make_mlp(self, output_size: int, layer_norm=False) -> nn.Module:
        """Builds an MLP."""
        widths = [self._latent_size] * self._num_layers #+ [output_size]
        network = LazyMLP(widths, layer_norm)
        return network


# TODO refactor into new file
class LazyMLP(nn.Module):
    def __init__(self, output_sizes: List[int], layer_norm):
        super().__init__()
        num_layers = len(output_sizes)
        self._layers_ordered_dict = OrderedDict()
        for index, output_size in enumerate(output_sizes):
            self._layers_ordered_dict["linear_" +
                                      str(index)] = nn.LazyLinear(output_size)
            #if index < (num_layers - 1):
            #    self._layers_ordered_dict["leakyrelu_" + str(index)] = nn.LeakyReLU()
            self._layers_ordered_dict["leakyrelu_" + str(index)] = nn.LeakyReLU()

        if layer_norm:
            self._layers_ordered_dict['layernorm_0'] = nn.LayerNorm(normalized_shape=output_sizes[-1])

        self.layers = nn.Sequential(self._layers_ordered_dict)

    def forward(self, input: Tensor) -> Tensor:
        input = input.to(device)
        y = self.layers(input)
        return y
