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


class TrapezModel(AbstractSystemModel):
    """
    Model for static flag simulation.
    """

    def __init__(self, params: ConfigDict):
        super(TrapezModel, self).__init__(params)
        self.learned_model = MeshGraphNets(
            output_size=2,
            latent_size=128,
            num_layers=self.num_layers,
            message_passing_steps=self.message_passing_steps,
            message_passing_aggregator=self.message_passing_aggregator,
            edge_sets=self._edge_sets,
            node_sets=self._node_sets,
            dec=self._node_sets[0],
            use_global=self.use_global, recurrence=self.rnn_type, self_sup=self.self_sup
        ).to(device)

    def forward(self, graph: Batch, is_training: bool) -> Tuple[Tensor, Tensor]:
        if self.self_sup:
            graph = self.split(graph, self.ggns)
        if self.feature_norm:
            graph[('mesh', '0', 'mesh')].edge_attr = self._mesh_edge_normalizer(graph[('mesh', '0', 'mesh')].edge_attr, is_training)
            graph['mesh'].x = self._feature_normalizer(graph['mesh'].x, is_training)
        return self.learned_model(graph)

    def training_step(self, graph: Batch):
        graph.to(device)
        pred_velocity, _ = self(graph, True)
        target_velocity = self.get_target(graph, True)

        loss = self.loss_fn(target_velocity, pred_velocity)

        return loss

    @torch.no_grad()
    def validation_step(self, graph: Batch, data_frame: Dict) -> Tuple[Tensor, Tensor]:
        graph.to(device)
        pred_velocity = self(graph, False)[0]
        target_velocity = self.get_target(graph, False)
        error = self.loss_fn(target_velocity, pred_velocity).cpu()

        pred_position, _, _ = self.update(graph, pred_velocity)
        pos_error = self.loss_fn(pred_position, graph.y).cpu()

        return error, pos_error

    @torch.no_grad()
    def rollout(self, trajectory: List[Dict[str, Tensor]], num_steps: int, freq: int) -> Tuple[Dict[str, Tensor], Tensor]:
        """Rolls out a model trajectory."""
        num_steps = len(trajectory) if num_steps is None else num_steps

        pred_trajectory = []
        cur_pos = trajectory[0]['pos']
        hidden = trajectory[0]['h']
        for step in range(num_steps):
            cur_pos, hidden = self._step_fn(hidden, cur_pos, trajectory[step], step, freq)
            pred_trajectory.append(cur_pos)

        point_index = trajectory[0]['point_index']
        prediction = torch.stack([x[:point_index] for x in pred_trajectory][:num_steps]).cpu()
        gt_pos = torch.stack([t['next_pos'][:point_index] for t in trajectory][:num_steps]).cpu()

        traj_ops = {
            'cells': trajectory[0]['cells'],
            'cell_type': trajectory[0]['cell_type'],
            'node_type': trajectory[0]['node_type'],
            'gt_pos': gt_pos,
            'pred_pos': prediction
        }

        mask = torch.where(trajectory[0]['node_type'] == NodeType.MESH)[0].cpu()
        mse_loss_fn = torch.nn.MSELoss(reduction='none')
        mse_loss = mse_loss_fn(gt_pos[:, mask], prediction[:, mask]).cpu()
        mse_loss = torch.mean(torch.mean(mse_loss, dim=-1), dim=-1).detach()

        return traj_ops, mse_loss

    @torch.no_grad()
    def _step_fn(self, hidden, cur_pos, ground_truth, step, freq):
        mask = torch.where(ground_truth['node_type'] == NodeType.MESH)[0].cpu()
        next_pos = copy.deepcopy(ground_truth['next_pos']).to(device)

        input = {**ground_truth, 'pos': cur_pos, 'h': hidden}

        keep_pc = False if freq == 0 else step % freq == 0
        index = 0 if keep_pc else 1

        data = Preprocessing.postprocessing(Data.from_dict(input).cpu(), self.self_sup, self.reduced)[index]
        graph = Batch.from_data_list([self.build_graph(data, is_training=False)]).to(device)

        output, hidden = self(graph, False)

        prediction, cur_position, cur_velocity = self.update(graph.to(device), output[mask])
        next_pos[mask] = prediction

        return next_pos, hidden

    @torch.no_grad()
    def n_step_computation(self, trajectory: List[Dict[str, Tensor]], n_step: int, num_timesteps=None, freq=1) -> Tuple[Tensor, Tensor]:
        mse_losses = list()
        last_losses = list()
        num_timesteps = len(trajectory) if num_timesteps is None else num_timesteps
        for step in range(num_timesteps // n_step):
            start = step * n_step
            if start < num_timesteps:
                eval_traj = trajectory[start: start + n_step]
                prediction_trajectory, mse_loss = self.rollout(eval_traj, n_step, freq)
                mse_losses.append(torch.mean(mse_loss).cpu())
                last_losses.append(mse_loss.cpu()[-1])

        return torch.mean(torch.stack(mse_losses)), torch.mean(torch.stack(last_losses))

    @staticmethod
    def split(graph, ggns=False):
        pc_mask = torch.where(graph['mesh'].node_type == NodeType.POINT)[0]
        shape_mask = torch.where(graph['mesh'].node_type == NodeType.SHAPE)[0]
        obst_mask = torch.where(graph['mesh'].node_type == NodeType.COLLIDER)[0]
        mesh_mask = torch.where(graph['mesh'].node_type == NodeType.MESH)[0]

        mgn_list = [mesh_mask, pc_mask, obst_mask, shape_mask] if ggns else [mesh_mask, obst_mask, shape_mask]
        mgn_mask = torch.cat(mgn_list, dim=0)

        # pc['mesh'].x = torch.cat([pc['mesh'].pos, pc['mesh'].x], dim=1)
        mesh = graph.subgraph({'mesh': mgn_mask}).clone()

        return mesh