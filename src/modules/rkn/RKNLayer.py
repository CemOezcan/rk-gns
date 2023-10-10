import torch

from src.modules.rkn.RKNCell import RKNCell

nn = torch.nn


class RKNLayer(nn.Module):

    def __init__(self, latent_obs_dim, cell_config, dtype=torch.float32):
        super().__init__()
        self._cell = RKNCell(latent_obs_dim, cell_config, dtype)

    def forward(self, latent_obs, obs_vars, initial_mean, initial_cov, obs_valid=None):
        """
        This currently only returns the posteriors. If you also need the priors uncomment the corresponding parts

        :param latent_obs: latent observations
        :param obs_vars: uncertainty estimate in latent observations
        :param initial_mean: mean of initial belief
        :param initial_cov: covariance of initial belief (as 3 vectors)
        :param obs_valid: flags indicating which observations are valid, which are not
        """

        # tif you need a version that also returns the prior uncomment the respective parts below
        # prepare list for return


        # initialize prior
        prior_mean, prior_cov = initial_mean, initial_cov

        # actual computation
        cur_obs_valid = obs_valid if obs_valid is not None else None
        post_means, post_covs, next_prior_means, next_prior_covs = \
            self._cell(prior_mean, prior_cov, latent_obs, obs_vars, cur_obs_valid)


        return post_means, post_covs #, prior_means, prior_covs
