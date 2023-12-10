import math

import numpy as np
import torch
from torch import nn

from src.modules.rkn.RKNCell import RKNCell
from src.util.types import ConfigDict
from src.util.util import device


class RKN(nn.Module):
    def __init__(self, latent_obs_dim: int):
        """
        RKN Cell (mostly) as described in the original RKN paper
        :param latent_obs_dim: latent observation dimension
        :param config: config dict object, for configuring the cell
        :param dtype: datatype
        """
        super(RKN, self).__init__()
        self._initial_state_variance = 10.0
        self._lod = latent_obs_dim
        self._lsd = 2 * self._lod
        self._initial_mean = torch.zeros(1, self._lsd).to(device)

        log_ic_init = np.log(np.exp(self._initial_state_variance) - 1.0)
        self._log_icu = torch.nn.Parameter(log_ic_init * torch.ones(1, self._lod).to(device))
        self._log_icl = torch.nn.Parameter(log_ic_init * torch.ones(1, self._lod).to(device))
        self._ics = torch.zeros(1, self._lod).to(device)

        self.mean_encoder = nn.LazyLinear(latent_obs_dim) # nn.Sequential(nn.LazyLinear(latent_obs_dim), nn.LayerNorm(normalized_shape=latent_obs_dim))
        self.var_encoder = nn.Sequential(nn.LazyLinear(latent_obs_dim), ScaledShiftedSigmoidActivation())

        #TODO: dtype?
        self._cell = RKNCell(latent_obs_dim, RKNCell.get_default_config(), dtype=torch.float32)

    def forward(self, graph):
        batch, hidden_mean, hidden_var = graph.u, graph.h, graph.c
        if hidden_mean.shape[-1] == self._lsd:
            prior_mean, prior_cov = graph.h, [graph.c[i] for i in range(3)]
        else:
            prior_mean, prior_cov = self._initial_mean, [var_activation(self._log_icu), var_activation(self._log_icl),
                                                         self._ics]

        w, w_var = self.mean_encoder(batch), self.var_encoder(batch)

        # w = nn.functional.normalize(w, p=2, dim=-1, eps=1e-8)
        # TODO: Validity indices
        obs_valid = graph.valid.bool()# None #if obs_valid is not None else None

        # TODO: Use priors for recurrence, but posteriors for predictions
        post_means, post_covs, prior_means, prior_cov = self._cell(prior_mean, prior_cov, w, w_var, obs_valid)

        return post_means, post_covs, prior_means, prior_cov


def var_activation(x: torch.Tensor) -> torch.Tensor:
    """
    elu + 1 activation faction to ensure positive covariances
    :param x: input
    :return: exp(x) if x < 0 else x + 1
    """
    return torch.log(torch.exp(x) + 1.0)


class SoftPlus(nn.Module):

    def __init__(self) -> None:
        super(SoftPlus, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x).where(x < 0.0, x + 1.0)


class ScaledShiftedSigmoidActivation(nn.Module):

    def __init__(self,
                 init_val: float = 1.0,
                 min_val: float = 0.01,
                 max_val: float = 5.0,
                 steepness: float = 1.0) -> None:
        super(ScaledShiftedSigmoidActivation, self).__init__()
        shift_init = init_val - min_val
        self._scale = max_val - min_val
        self._shift = math.log(shift_init / (self._scale - shift_init))
        self._min_val = min_val
        self._steepness = steepness

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._scale * torch.sigmoid(x * self._steepness + self._shift) + self._min_val

