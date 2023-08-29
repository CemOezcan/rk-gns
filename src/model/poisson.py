import copy
import torch
import torch_geometric.transforms as T
import numpy as np

from typing import Dict, Tuple, Union, List
from torch_geometric.data import Data, Batch, HeteroData
from torch import Tensor

from src.data.preprocessing import Preprocessing
from src.modules.mesh_graph_nets import MeshGraphNets
from src.modules.normalizer import Normalizer
from src.model.abstract_system_model import AbstractSystemModel
from src.util import test
from src.util.util import device
from src.util.types import NodeType
from src.util.types import ConfigDict

# TODO: integrate
class PoissonModel(AbstractSystemModel):
    """
    Model for static flag simulation.
    """

    def __init__(self, params: ConfigDict, recurrence: bool = False):
        super(PoissonModel, self).__init__(params)
        self.loss_fn = torch.nn.MSELoss()
        self.recurrence = recurrence
        self.learned_model = MeshGraphNets(
            output_size=1,
            latent_size=128,
            num_layers=self.num_layers,
            message_passing_steps=self.message_passing_steps,
            message_passing_aggregator=self.message_passing_aggregator,
            edge_sets=self._edge_sets,
            node_sets=self._node_sets,
            dec=self._node_sets[0],
            use_global=True, recurrence=self.recurrence, layer_norm=self.layer_norm
        ).to(device)

    def forward(self, graph: Batch, is_training: bool) -> Tuple[Tensor, Tensor]:
        _, graph = self.split_graphs(graph)
        if self.feature_norm:
            graph[('mesh', '0', 'mesh')].edge_attr = self._mesh_edge_normalizer(graph[('mesh', '0', 'mesh')].edge_attr, is_training)
            graph['mesh'].x = self._feature_normalizer(graph['mesh'].x, is_training)

        return self.learned_model(graph)

    def training_step(self, graph: Batch):
        graph.to(device)
        pred_u, _ = self(graph, True)
        target_u = self.get_target(graph, True)

        loss = self.loss_fn(target_u, pred_u)

        return loss

    def get_target(self, graph: Batch, is_training: bool) -> Tensor:
        return self._output_normalizer(graph.poisson, is_training)

    @torch.no_grad()
    def validation_step(self, graph: Batch, data_frame: Dict) -> Tuple[Tensor, Tensor]:
        graph.to(device)
        pred_u, _ = self(graph, False)
        target_u = self.get_target(graph, False)
        error = self.loss_fn(target_u, pred_u).cpu()

        pred_poisson, _, _ = self.update(graph, pred_u)
        true_error = self.loss_fn(pred_poisson, graph.poisson).cpu()

        return error, true_error

    def update(self, inputs: Batch, per_node_network_output: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Integrate model outputs."""
        poisson = self._output_normalizer.inverse(per_node_network_output)

        return poisson, poisson, poisson

    @torch.no_grad()
    def rollout(self, trajectory: List[Dict[str, Tensor]], num_steps: int, freq: int) -> Tuple[Dict[str, Tensor], Tensor]:
        """Rolls out a model trajectory."""
        num_steps = len(trajectory) if num_steps is None else num_steps
        hidden = trajectory[0]['h']
        point_index = trajectory[0]['point_index']

        pred_trajectory = []
        for step in range(num_steps):
            cur_pos, hidden = self._step_fn(hidden, None, trajectory[step], step, freq)
            pred_trajectory.append(cur_pos)

        prediction = torch.stack([t['pos'][:point_index] for t in trajectory][:num_steps]).cpu()
        gt_pos = torch.stack([t['pos'][:point_index] for t in trajectory][:num_steps]).cpu()

        u_pred = torch.stack([t for t in pred_trajectory][:num_steps]).cpu()
        u_gt = torch.stack([t['poisson'] for t in trajectory][:num_steps]).cpu()

        traj_ops = {
            'cells': trajectory[0]['cells'],
            'cell_type': trajectory[0]['cell_type'],
            'node_type': trajectory[0]['node_type'],
            'gt_pos': gt_pos,
            'pred_pos': prediction
        }

        mse_loss_fn = torch.nn.MSELoss(reduction='none')

        mse_loss = mse_loss_fn(u_pred, u_gt).cpu()
        mse_loss = torch.mean(torch.mean(mse_loss, dim=-1), dim=-1).detach()

        return traj_ops, mse_loss

    @torch.no_grad()
    def _step_fn(self, hidden, cur_pos, ground_truth, step, freq):
        input = {**ground_truth, 'h': hidden}

        keep_pc = False if self.mgn else step % freq == 0
        index = 0 if keep_pc else 1

        data = Preprocessing.postprocessing(Data.from_dict(input).cpu(), True)[index]
        graph = Batch.from_data_list([self.build_graph(data, is_training=False)]).to(device)

        output, hidden = self(graph, False)
        prediction, _, _ = self.update(graph.to(device), output)

        return prediction, hidden

    @torch.no_grad()
    def n_step_computation(self, trajectory: List[Dict[str, Tensor]], n_step: int, num_timesteps=None, freq: int = 1) -> Tuple[Tensor, Tensor]:
        mse_losses = list()
        last_losses = list()
        num_timesteps = len(trajectory) if num_timesteps is None else num_timesteps
        for step in range(num_timesteps - n_step):
            eval_traj = trajectory[step: step + n_step + 1]
            prediction_trajectory, mse_loss = self.rollout(eval_traj, n_step + 1, freq)
            mse_losses.append(torch.mean(mse_loss).cpu())
            last_losses.append(mse_loss.cpu()[-1])

        return torch.mean(torch.stack(mse_losses)), torch.mean(torch.stack(last_losses))
