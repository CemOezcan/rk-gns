import numpy as np
import torch
from torch import nn

from src.modules.rkn.RKNCell import RKNCell
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

    def forward(self, graph):
        if graph.h.shape[-1] == self._lod:
            u = self._cell(graph.u, graph.h)
            return u, None, u, None
        else:
            u = self._cell(graph.u)
            return u, None, u, None