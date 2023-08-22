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

    def __init__(self, params: ConfigDict, recurrence: bool = False):
        super(TrapezModel, self).__init__(params)
        self.recurrence = recurrence
        self.learned_model = MeshGraphNets(
            output_size=params.get('size'),
            latent_size=128,
            num_layers=1,
            message_passing_steps=self.message_passing_steps,
            message_passing_aggregator=self.message_passing_aggregator,
            edge_sets=self._edge_sets,
            node_sets=self._node_sets,
            dec=self._node_sets[0],
            use_global=True, recurrence=self.recurrence
        ).to(device)

    def forward(self, graph: Batch, is_training: bool) -> Tuple[Tensor, Tensor]:
        graph, _ = self.split_graphs(graph)
        graph[('mesh', '0', 'mesh')].edge_attr = self._mesh_edge_normalizer(graph[('mesh', '0', 'mesh')].edge_attr, is_training)
        graph['mesh'].x = self._feature_normalizer(graph['mesh'].x, is_training)

        return self.learned_model(graph)

    def training_step(self, graph: Batch, poisson_model):
        graph.to(device)

        output, _ = poisson_model(graph, False)
        poisson, _, _ = poisson_model.update(graph, output)
        graph.u = poisson

        prediction, _ = self(graph, True)
        target = self.get_target(graph, True)

        loss = self.loss_fn(target, prediction)

        return loss

    def get_target(self, graph: Batch, is_training: bool) -> Tensor:
        mask = torch.where(graph['mesh'].node_type == NodeType.MESH)[0]
        target_velocity = graph.y - graph['mesh'].pos[mask]

        return self._output_normalizer(target_velocity, is_training)

    @torch.no_grad()
    def validation_step(self, graph: Batch, data_frame: Dict, poisson_model) -> Tuple[Tensor, Tensor]:
        graph.to(device)

        output, _ = poisson_model(graph, False)
        u_target = poisson_model.get_target(graph, False)
        u_error = self.loss_fn(output, u_target).cpu()

        poisson, _, _ = poisson_model.update(graph, output)
        poisson_error = self.loss_fn(poisson, graph.u).cpu()
        graph.u = poisson

        prediction, _ = self(graph, False)
        target = self.get_target(graph, False)
        error = self.loss_fn(target, prediction).cpu()

        pred_position, _, _ = self.update(graph, prediction)
        pos_error = self.loss_fn(pred_position, graph.y).cpu()

        return error, pos_error, u_error, poisson_error

    def update(self, inputs: Batch, per_node_network_output: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Integrate model outputs."""
        mask = torch.where(inputs['mesh'].node_type == NodeType.MESH)[0]
        velocity = self._output_normalizer.inverse(per_node_network_output)

        # integrate forward
        cur_position = inputs['mesh'].pos[mask]

        # vel. = next_pos - cur_pos
        position = cur_position + velocity

        return (position, cur_position, velocity)

    @torch.no_grad()
    def rollout(self, trajectory: List[Dict[str, Tensor]], num_steps: int, poisson_model) -> Tuple[Dict[str, Tensor], Tensor]:
        """Rolls out a model trajectory."""
        num_steps = len(trajectory) if num_steps is None else num_steps
        initial_state = trajectory[0]
        point_index = initial_state['point_index']

        pred_trajectory = []
        pred_u = list()
        cur_pos = torch.squeeze(initial_state['pos'], 0)
        for step in range(num_steps):
            cur_pos, hidden, u = self._step_fn(initial_state, cur_pos, trajectory[step], step, poisson_model)
            initial_state['h'] = hidden
            pred_u.append(u)
            pred_trajectory.append(cur_pos)

        prediction = torch.stack([x[:point_index] for x in pred_trajectory][:num_steps]).cpu()
        gt_pos = torch.stack([t['pos'][:point_index] for t in trajectory][:num_steps]).cpu()

        u_pred = torch.stack([t for t in pred_u][:num_steps]).cpu()
        u_gt = torch.stack([t['poisson'] for t in trajectory][:num_steps]).cpu()

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

        u_loss = mse_loss_fn(u_pred, u_gt).cpu()
        u_loss = torch.mean(torch.mean(u_loss, dim=-1), dim=-1).detach()

        return traj_ops, mse_loss, u_loss

    @torch.no_grad()
    def _step_fn(self, initial_state, cur_pos, ground_truth, step, poisson_model=None):
        mask = torch.where(ground_truth['node_type'] == NodeType.MESH)[0].cpu()
        next_pos = copy.deepcopy(ground_truth['next_pos']).to(device)

        input = {**initial_state, 'x': ground_truth['x'], 'pos': cur_pos, 'next_pos': ground_truth['next_pos'],
                 'y': ground_truth['next_pos'][mask], 'node_type': ground_truth['node_type']}

        data = Preprocessing.postprocessing(Data.from_dict(input).cpu(), True)
        keep_pc = False if self.mgn else step % self.pc_frequency == 0
        graph = Batch.from_data_list([self.build_graph(data, is_training=False, keep_point_cloud=keep_pc)]).to(device)

        output, hidden = poisson_model(graph, False)
        poisson, _, _ = poisson_model.update(graph, output)
        graph.u = poisson

        output, hidden = self(graph, False)

        prediction, cur_position, cur_velocity = self.update(graph.to(device), output[mask])
        next_pos[mask] = prediction

        return next_pos, hidden, poisson

    @torch.no_grad()
    def n_step_computation(self, trajectory: List[Dict[str, Tensor]], n_step: int, num_timesteps=None, poisson_model=None) -> Tuple[Tensor, Tensor]:
        mse_losses = list()
        last_losses = list()
        u_losses = list()
        u_last_losses = list()
        num_timesteps = len(trajectory) if num_timesteps is None else num_timesteps
        for step in range(num_timesteps - n_step):
            # TODO: clusters/balancers are reset when computing n_step loss
            eval_traj = trajectory[step: step + n_step + 1]
            prediction_trajectory, mse_loss, u_loss = self.rollout(eval_traj, n_step + 1, poisson_model)

            mse_losses.append(torch.mean(mse_loss).cpu())
            last_losses.append(mse_loss.cpu()[-1])

            u_losses.append(torch.mean(u_loss).cpu())
            u_last_losses.append(u_loss.cpu()[-1])

        return torch.mean(torch.stack(mse_losses)), torch.mean(torch.stack(last_losses)), \
            torch.mean(torch.stack(u_losses)), torch.mean(torch.stack(u_last_losses))

    @staticmethod
    def add_noise(data: Data, sigma: float, node_type: int):
        """
        Adds training noise to the mesh node positions with standard deviation sigma
        Args:
            data: PyG data element containing (a batch of) graph(s)
            sigma: standard deviation of used noise
            node_type: The type of node to add noise to

        Returns:
            data: updated graph with noise

        """
        if sigma > 0.0:
            indices = torch.where(data.node_type == node_type)[0]
            num_node_features = data.pos.shape[1]
            noise = (torch.randn(indices.shape[0], num_node_features) * sigma).cpu()
            data.pos[indices, :num_node_features] = data.pos[indices, :num_node_features] + noise

        return data

    @staticmethod
    def transform_position_to_edges(data: Data, euclidian_distance: bool) -> Data:
        """
        Transform the node positions in a homogeneous data element to the edges as relative distance together with (if needed) Euclidean norm
        Args:
            data: Data element
            euclidian_distance: True if Euclidean norm included as feature

        Returns:
            out_data: Transformed data object
        """
        if euclidian_distance:
            data_transform = T.Compose([T.Cartesian(norm=False, cat=True), T.Distance(norm=False, cat=True)])
        else:
            data_transform = T.Compose([T.Cartesian(norm=False, cat=True)])
        out_data = data_transform(data)
        return out_data


