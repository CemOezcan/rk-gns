import copy
import math
import os
import pickle
import random

import torch
import wandb

import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

from typing import Optional, Dict, List, Tuple, Any, Union
from pandas import DataFrame
from torch import Tensor
from tqdm import tqdm
from abc import ABC, abstractmethod
from torch_geometric.loader import DataLoader

from src.data.get_data import get_directories
from src.model.get_model import get_model
from src.util.types import ConfigDict
from src.util.util import device


class AbstractSimulator(ABC):
    """
    Superclass for iterative algorithms
    """

    def __init__(self, config: ConfigDict) -> None:
        """
        Initializes the iterative algorithm.

        Parameters
        ----------
            config : ConfigDict
                A (potentially nested) dictionary containing the "params" section of the section in the .yaml file
                used by cw2 for the current run.

        """

        self._config = config
        self._network_config = config.get('model')
        self._dataset_name = config.get('directory')
        _, self._out_dir = get_directories(self._dataset_name)
        self._wandb_mode = config.get('wandb_mode')

        self._trajectories = config.get('task').get('trajectories')
        self._time_steps = config.get('task').get('n_timesteps')
        self._seq_len = config.get('task').get('sequence')
        self.recurrence = config.get('task').get('recurrence') is not False
        ggns = self._config.get('task').get('ggns')
        poisson = self._config.get('task').get('task').lower() == 'poisson'
        ss = self._config.get('task').get('model').lower() == 'self-supervised'

        if poisson:
            self.mode = 'poisson'
        elif ss:
            self.mode = 'self-supervised'
        elif not ggns:
            self.mode = 'mgn'
        else:
            self.mode = None

        self._batch_size = config.get('task').get('batch_size')
        self._network = None
        self._optimizer = None
        self._wandb_run = None
        self._wandb_url = None
        self._initialized = False
        self.random_seed, self.np_seed, self.torch_seed = None, None, None

        self.loss_function = F.mse_loss
        self._learning_rate = config.get('task').get('learning_rate')
        self._weight_decay = config.get('task').get('weight_decay')

    def initialize(self, task_information: ConfigDict) -> None:
        """
        Due to the interplay between the algorithm and the task, it sometimes makes sense for the task to provide
        additional initial information to the algorithm. This information may e.g., be the dimensionality of the task,
        the kind of training regime to perform etc.

        Parameters
        ----------
            task_information : ConfigDict
                A dictionary containing information on how to execute the algorithm on the current task

        Returns
        -------

        """
        self._wandb_run = None
        self._wandb_mode = task_information.get('wandb_mode')
        wandb.init(project='RK-GNS', config=task_information, mode=self._wandb_mode)
        wandb.define_metric('epoch')
        wandb.define_metric('validation_loss', step_metric='epoch')
        wandb.define_metric('position_loss', step_metric='epoch')
        wandb.define_metric('validation_mean', step_metric='epoch')
        wandb.define_metric('position_mean', step_metric='epoch')
        wandb.define_metric('rollout_loss', step_metric='epoch')
        wandb.define_metric('video', step_metric='epoch')

        if self._wandb_url is not None and self._wandb_mode == 'online':
            api = wandb.Api()
            run = api.run(self._wandb_url)
            this_run = api.run(wandb.run.path)
            curr_epoch = max([x['epoch'] for x in run.scan_history(keys=['epoch'])])
            for file in run.files():
                this_run.upload_file(file.download(replace=True).name)
            b = False
            for x in run.scan_history():
                if b:
                    break
                try:
                    b = x['epoch'] >= curr_epoch
                except (KeyError, TypeError):
                    b = False
                wandb.log(x)

        self._wandb_url = wandb.run.path

        if not self._initialized:
            self._batch_size = task_information.get('task').get('batch_size')
            self._network = get_model(task_information)
            self._optimizer = optim.Adam(self._network.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay)
            self.best_models = {x: (math.inf, None) for x in [0, 1, 2, 5]}
            self._initialized = True
        else:
            self.set_seed()

    def set_seed(self):
        random.setstate(self.random_seed)
        np.random.set_state(self.np_seed)
        torch.set_rng_state(self.torch_seed)

    @abstractmethod
    def fetch_data(self, trajectory: List, is_training: bool) -> DataLoader:
        """
        Transform a collection of system states into batched graphs.

        Parameters
        ----------
            trajectory : DataLoader
                A collection of system states
            is_training : bool
                Whether this is a training or a test/validation run

        Returns
        -------
            DataLoader
                Collection of batched graphs.
        """
        raise NotImplementedError

    def pretraining(self, train_dataloader: List) -> None:
        """
        Pretrain your algorithm if necessary. Default is no pretraining

        Parameters
        ----------
            train_dataloader: List
                A List containing the training data

        Returns
        -------

        """
        return

    @abstractmethod
    def fit_iteration(self, train_dataloader: List) -> float:
        """
        Train your algorithm for a single iteration. This can e.g., be a single epoch of neural network training,
        a policy update step, or something more complex. Just see this as the outermost for-loop of your algorithm.

        Parameters
        ----------
            train_dataloader : List
                A List containing the training data

        Returns
        -------
            May return an optional dictionary of values produced during the fit. These may e.g., be statistics
            of the fit such as a training loss.

        """

        raise NotImplementedError

    @torch.no_grad()
    def one_step_evaluator(self, ds_loader: List, instances: int, task_name: str, logging: bool = True) -> Optional[Dict]:
        """
        Predict the system state for the next time step and evaluate the predictions over the test data.

        Parameters
        ----------
            ds_loader : List
                A list containing test/validation instances

            instances : int
                Number of trajectories used to estimate the one-step loss

            task_name : str
                Name of the task

            logging : bool
                Whether to log the results to wandb

        Returns
        -------
            Optional[Dict]
                A single result that scores the input, potentially per sample

        """
        self._network.eval()
        trajectory_loss = list()
        test_loader = self.fetch_data(ds_loader, is_training=False)
        for i, batch in enumerate(tqdm(test_loader, desc='Validation', leave=True, position=0)):
            batch.to(device)
            instance_loss = self._network.validation_step(batch, i)

            trajectory_loss.append([instance_loss])

        mean = np.mean(trajectory_loss, axis=0)
        std = np.std(trajectory_loss, axis=0)

        val_loss, pos_loss = zip(*mean)
        val_std, pos_std = zip(*std)

        log_dict = {
            'single-step error/velocity_historgram':
                wandb.Histogram(
                    [x for x in val_loss],
                    num_bins=20
                ),
            'single-step error/position_historgram':
                wandb.Histogram(
                    [x for x in pos_loss],
                    num_bins=20
                ),
            'single-step error/velocity_error': np.mean(val_loss),
            'single-step error/position_error': np.mean(pos_loss),
            'single-step error/velocity_std': np.mean(val_std),
            'single-step error/position_std': np.mean(pos_std)
        }
        return log_dict

    @torch.no_grad()
    def n_step_evaluator(self, ds_loader: List, task_name: str, n_steps: int, n_traj: int, logging: bool = True, freq: int = 1) -> Optional[Dict]:
        """
        Predict the system state after n time steps. N step predictions are performed recursively within trajectories.
        Evaluate the predictions over the test data.

        Parameters
        ----------
            ds_loader : List
                A list containing test/validation instances

            task_name : str
                Name of the task

            n_steps : int
                Value of n, with which to estimate the n-step loss

            n_traj : int
                Number of trajectories used to estimate the n-step loss

        Returns
        -------

        """
        self._network.eval()
        track_var = True
        # Take n_traj trajectories from valid set for n_step loss calculation
        means = list()
        lasts = list()
        vars = list()
        lst_vars = list()
        for i, trajectory in enumerate(tqdm(ds_loader, desc='N-Step', leave=True, position=0)):
            if i >= n_traj:
                break
            mean_loss, last_loss, mean_var, last_var = self._network.n_step_computation(trajectory, n_steps, self._time_steps, freq=freq)
            if mean_var is None or last_var is None:
                track_var = False
            vars.append(mean_var)
            lst_vars.append(last_var)
            means.append(mean_loss)
            lasts.append(last_loss)

        means = torch.mean(torch.stack(means))
        lasts = torch.mean(torch.stack(lasts))

        scalars = {
            f'{n_steps}-step error/mean_k={freq}': torch.mean(torch.tensor(means), dim=0),
            f'{n_steps}-step error/last_k={freq}': torch.mean(torch.tensor(lasts), dim=0)
        }

        if track_var:
            vars = torch.mean(torch.stack(vars))
            lst_vars = torch.mean(torch.stack(lst_vars))

            scalars[f'{n_steps}-step error/mean_var_k={freq}'] = torch.mean(torch.tensor(vars), dim=0)
            scalars[f'{n_steps}-step error/last_var_k={freq}'] = torch.mean(torch.tensor(lst_vars), dim=0)

        return scalars

    @torch.no_grad()
    def rollout_evaluator(self, ds_loader: List, rollouts: int, task_name: str, logging: bool = True, freq: int = 1) -> Optional[Dict]:
        """
        Recursive prediction of the system state at the end of trajectories.
        Evaluate the predictions over the test data.

        Parameters
        ----------
            ds_loader : List
                A list containing test/validation instances

            rollouts : int
                Number of trajectories used to estimate the rollout loss

            task_name : str
                Name of the task

            logging : bool
                Whether to log the results to wandb

        Returns
        -------
            Optional[Dict]
                A single result that scores the input, potentially per sample

        """
        track_var = True
        self._network.eval()
        trajectories = []
        mse_losses = []
        vars = []
        pred = list()
        for i, trajectory in enumerate(tqdm(ds_loader, desc='Rollouts', leave=True, position=0)):
            if i >= rollouts:
                break
            prediction_trajectory, mse_loss, var, p = self._network.rollout(trajectory, num_steps=self._time_steps, freq=freq)
            if var is None or p is None:
                track_var = False
            trajectories.append(prediction_trajectory)
            pred.append(p)
            mse_losses.append(mse_loss.cpu())
            vars.append(var)

        rollout_hist = wandb.Histogram([x for x in torch.mean(torch.stack(mse_losses), dim=1)], num_bins=20)
        mse_means = torch.mean(torch.stack(mse_losses), dim=0)
        mse_stds = torch.std(torch.stack(mse_losses), dim=0)

        rollout_losses = {
            'mse_loss': [mse.item() for mse in mse_means],
            'mse_std': [mse.item() for mse in mse_stds]
        }

        self.save_rollouts(trajectories, task_name, freq)
        current_mean = torch.mean(torch.tensor(rollout_losses['mse_loss']), dim=0)
        prior_mean = self.best_models[freq][0]
        if current_mean < prior_mean:
            self.best_models[freq] = (current_mean, copy.deepcopy(self._network))

        scalars = {
            f'rollout error/mean_k={freq}': torch.mean(torch.tensor(rollout_losses['mse_loss']), dim=0),
            f'rollout error/std_k={freq}': torch.mean(torch.tensor(rollout_losses['mse_std']), dim=0),
            f'rollout error/last_k={freq}': rollout_losses['mse_loss'][-1],
            f'rollout error/fst_k={freq}': torch.mean(torch.tensor(rollout_losses['mse_loss'][:10]), dim=0),
            f'rollout error/lst_k={freq}': torch.mean(torch.tensor(rollout_losses['mse_loss'][-10:]), dim=0),
            f'rollout error/histogram_k={freq}': rollout_hist
        }

        if track_var:
            var_means = torch.mean(torch.stack(vars), dim=0)
            pred_max = torch.max(torch.stack(pred), dim=0).values
            pred_min = torch.min(torch.stack(pred), dim=0).values

            scalars = {**scalars, f'rollout_error/var_fst_k={freq}': torch.mean(var_means[:10], dim=0),
                       f'rollout_error/var_lst_k={freq}': torch.mean(var_means[-10:], dim=0),
                       f'rollout_error/var_k={freq}': torch.mean(var_means, dim=0),
                       f'rollout_error/pred_max_fst_k={freq}': torch.mean(pred_max[:10], dim=0),
                       f'rollout_error/pred_max_lst_k={freq}': torch.mean(pred_max[-10:], dim=0),
                       f'rollout_error/pred_min_fst_k={freq}': torch.mean(pred_min[:10], dim=0),
                       f'rollout_error/pred_min_lst_k={freq}': torch.mean(pred_min[-10:], dim=0)}

        return scalars

    def save(self, name: str) -> None:
        """
        Save itself as a .pkl file.

        Parameters
        ----------
            name : str
                The name under which to store this mesh simulator
        """
        self.random_seed = random.getstate()
        self.np_seed = np.random.get_state()
        self.torch_seed = torch.get_rng_state()
        with open(os.path.join(self._out_dir, f'model_{name}.pkl'), 'wb') as file:
            pickle.dump(self, file)

    def save_rollouts(self, rollouts: List[Dict[str, Tensor]], task_name: str, freq: int) -> None:
        """
        Save predicted and ground truth trajectories.

        Parameters
        ----------
            rollouts : Dict[str, Tensor]
                The rollout data

            task_name : str
                The task name
        """
        rollouts = [{key: value.to('cpu') for key, value in x.items()} for x in rollouts]
        with open(os.path.join(self._out_dir, f'{task_name}_rollouts_k={freq}.pkl'), 'wb') as file:
            pickle.dump(rollouts, file)

    @staticmethod
    def _publish_csv(data_frame: DataFrame, name: str, path: str) -> None:
        """
        Publish a table using wandb.

        Parameters
        ----------
            data_frame : DataFrame
                The table
            name : str
                The table name
            path : str
                The path of the table
        """
        table = wandb.Table(dataframe=data_frame)
        wandb.log({name: table})
        #artifact = wandb.Artifact(f'{name}_artifact', type='dataset')
        #artifact.add(table, f'{name}_table')
        #artifact.add_file(path)
        #wandb.log_artifact(artifact)

    @staticmethod
    def log_epoch(data: Dict[str, Any]) -> None:
        """
        Log the metrics of an epoch.

        Parameters
        ----------
            data : Dict[str, Any]
                The data to log

        """
        wandb.log(data)

    def log_gradients(self, model):
        grad_first_layer = self.calculate_gradients(model, 0)
        grad_last_layer = self.calculate_gradients(model, -1)
        grad_enc = torch.cat([param.grad.view(-1) for param in model.learned_model.encoder.parameters()]).abs().mean()
        grad_proc = torch.cat([param.grad.view(-1) for param in model.learned_model.processor.parameters()]).abs().mean()
        #print([(name, param) for name, param in model.learned_model.decoder.named_parameters()])
        grad_dec = torch.cat([param.grad.view(-1) for param in model.learned_model.decoder.parameters() if param.grad is not None]).abs().mean()

        #param_rkn = torch.cat([param.view(-1) for param in model.learned_model.decoder.rnn.parameters()]).abs().mean()
        #param_rkn_enc = torch.cat([param.view(-1) for param in model.learned_model.decoder.rnn.mean_encoder.parameters()]).abs().mean()
        #param_rkn_enc_var = torch.cat([param.view(-1) for param in model.learned_model.decoder.rnn.mean_encoder.parameters()]).abs().mean()

        param_enc = torch.cat([param.view(-1) for param in model.learned_model.encoder.parameters()]).abs().mean()
        param_proc = torch.cat(
            [param.view(-1) for param in model.learned_model.processor.parameters()]).abs().mean()
        param_dec = torch.cat(
            [param.view(-1) for param in model.learned_model.decoder.parameters()]).abs().mean()

        grad = []
        for param in model.parameters():
            if param.grad is not None:
                grad.append(param.grad.view(-1))
        grad = torch.cat(grad).abs().mean()
        return {"gradients/first_layer": grad_first_layer,
                "gradients/last_layer": grad_last_layer,
                "gradients/encoder": grad_enc,
                "gradients/decoder": grad_dec,
                "gradients/processor": grad_proc,
                "gradients/all_layers": grad, #'gradients/param_rkn': param_rkn,
                #'gradients/param_rkn_enc': param_rkn_enc, 'gradients/param_rkn_enc_var': param_rkn_enc_var,
                'gradients/param_enc': param_enc, 'gradients/param_proc': param_proc, 'gradients/param_dec': param_dec}

    def calculate_gradients(self, model, layer):
        """
        Calculates the mean gradient of our GNN for a given layer
        Args:
            GNN: gnn_base object
            layer: layer number

        Returns:
            Mean gradient for the layer
        """
        grad = []
        for name, edge_parameter in model.learned_model.processor.graphnet_blocks[
            layer].edge_models['mesh0mesh'].layers.named_parameters():
            grad.append(edge_parameter.grad.view(-1))
        for name, node_parameter in model.learned_model.processor.graphnet_blocks[
            layer].node_models['mesh'].layers.named_parameters():
            grad.append(node_parameter.grad.view(-1))
        grad = torch.cat(grad).abs().mean()

        return grad

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)