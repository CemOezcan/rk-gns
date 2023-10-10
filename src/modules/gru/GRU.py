import numpy as np
import torch
from torch import nn

from src.modules.rkn.RKNCell import RKNCell
from src.modules.rkn.RKNLayer import RKNLayer
from src.util.types import ConfigDict
from src.util.util import device


class GRU(nn.Module):
    def __init__(self, latent_obs_dim: int):
        """
        RKN Cell (mostly) as described in the original RKN paper
        :param latent_obs_dim: latent observation dimension
        :param config: config dict object, for configuring the cell
        :param dtype: datatype
        """
        super(GRU, self).__init__()
        #TODO: dtype
        self._lod = latent_obs_dim
        self._cell = nn.GRUCell(latent_obs_dim, latent_obs_dim)
        #self.reduce = nn.LazyLinear(latent_obs_dim)

    def forward(self, graph):
        #graph.u = self.reduce(graph.u)
        if graph.h.shape[-1] == self._lod:
            return self._cell(graph.u, graph.h), None
        else:
            return self._cell(graph.u), None