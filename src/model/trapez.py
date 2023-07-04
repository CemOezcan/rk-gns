import copy
import math
from typing import Dict, Tuple, Union, List

import torch_cluster
import torch_geometric.transforms as T
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch, HeteroData

from src.data.preprocessing import Preprocessing
from src.util import util, test
from src.modules.mesh_graph_nets import MeshGraphNets
from src.modules.normalizer import Normalizer
from src.model.abstract_system_model import AbstractSystemModel
from src.util.util import device
from src.util.types import NodeType
from torch import nn, Tensor

from src.util.types import ConfigDict


class TrapezModel(AbstractSystemModel):
    """
    Model for static flag simulation.
    """

    def __init__(self, params: ConfigDict):
        super(TrapezModel, self).__init__(params)
        self.loss_fn = torch.nn.MSELoss()

        self._output_normalizer = Normalizer(size=2, name='output_normalizer')
        self._mesh_edge_normalizer = Normalizer(size=14, name='mesh_edge_normalizer')

        self.message_passing_steps = params.get('message_passing_steps')
        self.message_passing_aggregator = params.get('aggregation')
        self.recurrence = True

        self._edge_sets = [''.join(('mesh', '0', 'mesh'))]
        self._node_sets = ['mesh']

        self.learned_model = MeshGraphNets(
            output_size=params.get('size'),
            latent_size=128,
            num_layers=1,
            message_passing_steps=self.message_passing_steps,
            message_passing_aggregator=self.message_passing_aggregator,
            edge_sets=self._edge_sets,
            node_sets=self._node_sets,
            dec=self._node_sets[0],
            use_global=params.get('use_global'), recurrence=self.recurrence
        ).to(device)

        self.euclidian_distance = True
        self.pc_frequency = params.get('pc_frequency')
        self.mgn = params.get('mgn')
        self.hetero = params.get('heterogeneous')
        self.input_mesh_noise = params.get('noise')
        self.input_pcd_noise = params.get('pc_noise')

    def build_graph(self, data: Tuple[Data, Data], is_training: bool, keep_point_cloud: Union[bool, None] = None) -> HeteroData:
        """Builds input graph."""
        if self.mgn:
            data = data[1]
        elif keep_point_cloud is None:
            x = np.random.rand(1)
            data = data[0] if x < (1 / self.pc_frequency) else data[1]
        elif keep_point_cloud:
            data = data[0]
        else:
            data = data[1]

        data.to(device)
        if is_training:
            data = self.add_noise_to_mesh_nodes(data, self.input_mesh_noise, device)
        data = self.add_noise_to_pcd_points(data, self.input_pcd_noise, device)
        data = self.transform_position_to_edges(data, self.euclidian_distance)
        data.edge_attr = self._mesh_edge_normalizer(data.edge_attr, is_training)

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        node_attr = data.x
        node_type = data.node_type
        edge_type = data.edge_type

        # Create a HeteroData object
        hetero_data = HeteroData()

        # Add node data to the HeteroData object
        hetero_data[self._node_sets[0]].x = node_attr
        hetero_data.node_type = node_type

        # Add edge data to the HeteroData object
        hetero_data[('mesh', '0', 'mesh')].edge_index = edge_index
        hetero_data[('mesh', '0', 'mesh')].edge_attr = edge_attr
        hetero_data.edge_type = edge_type

        hetero_data.u = data.u
        hetero_data.h = data.h
        hetero_data.c = data.c
        hetero_data.pos = data.pos
        hetero_data.y = data.y
        hetero_data.next_pos = data.next_pos
        hetero_data.cpu()

        return hetero_data

    def forward(self, graph):
        return self.learned_model(graph)

    def training_step(self, graph: Batch):
        pred_velocity, _ = self(graph)
        target_velocity = self.get_target(graph, True)

        loss = self.loss_fn(target_velocity, pred_velocity)

        return loss

    def get_target(self, graph, is_training):
        mask = torch.where(graph.node_type == NodeType.MESH)[0]
        target_velocity = graph.y - graph.pos[mask]

        return self._output_normalizer(target_velocity, is_training)

    @torch.no_grad()
    def validation_step(self, graph: Batch, data_frame: Dict) -> Tuple[Tensor, Tensor]:
        pred_velocity = self(graph)[0]
        target_velocity = self.get_target(graph, False)
        error = self.loss_fn(target_velocity, pred_velocity).cpu()

        pred_position, _, _ = self.update(graph, pred_velocity)
        pos_error = self.loss_fn(pred_position, graph.y).cpu()

        return error, pos_error

    def update(self, inputs: Batch, per_node_network_output: Tensor) -> Tensor:
        """Integrate model outputs."""
        mask = torch.where(inputs.node_type == NodeType.MESH)[0]
        velocity = self._output_normalizer.inverse(per_node_network_output)

        # integrate forward
        cur_position = inputs.pos[mask]

        # vel. = next_pos - cur_pos
        position = cur_position + velocity

        return (position, cur_position, velocity)

    @torch.no_grad()
    def rollout(self, trajectory: List[Dict[str, Tensor]], num_steps: int) -> Tuple[Dict[str, Tensor], Tensor]:
        """Rolls out a model trajectory."""
        num_steps = len(trajectory) if num_steps is None else num_steps
        initial_state = trajectory[0] # {k: torch.squeeze(v, 0)[0] for k, v in trajectory.items()}

        point_index = initial_state['point_index']

        cur_pos = torch.squeeze(initial_state['pos'], 0).to(device)
        target_pos = [t['next_pos'].to(device) for t in trajectory]
        features = [t['x'].to(device) for t in trajectory]
        pred_trajectory = []
        hidden = (initial_state['u'], (initial_state['h'], initial_state['c']))
        for step in range(num_steps):
            node_type = trajectory[step]['node_type']
            mask = torch.where(node_type == NodeType.MESH)[0].to(device)
            cur_pos, pred_trajectory, hidden = self._step_fn(initial_state, cur_pos, pred_trajectory,
                                                             target_pos[step], features[step], mask, step, hidden)

        prediction = torch.stack([x[:point_index] for x in pred_trajectory][:num_steps]).cpu()
        gt_pos = torch.stack([t['pos'][:point_index] for t in trajectory][:num_steps]).cpu()

        traj_ops = {
            'cells': trajectory[0]['cells'],
            'cell_type': trajectory[0]['cell_type'],
            'node_type': node_type,
            'gt_pos': gt_pos,
            'pred_pos': prediction
        }
        mask = mask.cpu()

        mse_loss_fn = torch.nn.MSELoss(reduction='none')
        mse_loss = mse_loss_fn(gt_pos[:, mask], prediction[:, mask]).cpu()
        mse_loss = torch.mean(torch.mean(mse_loss, dim=-1), dim=-1).detach()

        return traj_ops, mse_loss

    @torch.no_grad()
    def _step_fn(self, initial_state, cur_pos, trajectory, target_world_pos, x, mask, step, hidden):
        next_pos = copy.deepcopy(target_world_pos)
        h, c = hidden
        input = {**initial_state, 'x': x, 'pos': cur_pos, 'next_pos': target_world_pos, 'y': target_world_pos[mask],
                 'h': h, 'c': c}

        data = Preprocessing.postprocessing(Data.from_dict(input).cpu())
        keep_pc = False if self.mgn else step % self.pc_frequency == 0
        graph = Batch.from_data_list([self.build_graph(data, is_training=False, keep_point_cloud=keep_pc)]).to(device)
        data = data[0] if keep_pc else data[1]

        output, hidden = self(graph)

        prediction, cur_position, cur_velocity = self.update(data.to(device), output[mask])
        next_pos[mask] = prediction

        trajectory.append(next_pos)
        return next_pos, trajectory, hidden

    @torch.no_grad()
    def n_step_computation(self, trajectory: List[Dict[str, Tensor]], n_step: int, num_timesteps=None) -> Tuple[Tensor, Tensor]:
        mse_losses = list()
        last_losses = list()
        num_timesteps = len(trajectory) if num_timesteps is None else num_timesteps
        for step in range(num_timesteps - n_step):
            # TODO: clusters/balancers are reset when computing n_step loss
            eval_traj = trajectory[step: step + n_step + 1]
            prediction_trajectory, mse_loss = self.rollout(eval_traj, n_step + 1)
            mse_losses.append(torch.mean(mse_loss).cpu())
            last_losses.append(mse_loss.cpu()[-1])

        return torch.mean(torch.stack(mse_losses)), torch.mean(torch.stack(last_losses))

    @staticmethod
    def add_noise_to_mesh_nodes(data: Data, sigma: float, device):
        """
        Adds training noise to the mesh node positions with standard deviation sigma
        Args:
            data: PyG data element containing (a batch of) graph(s)
            sigma: standard deviation of used noise
            device: working device (cuda or cpu)

        Returns:
            data: updated graph with noise

        """
        if sigma > 0.0:
            indices = torch.where(data.node_type == NodeType.MESH)[0]
            num_noise_features = data.pos.shape[1]
            num_node_features = data.pos.shape[1]
            noise = (torch.randn(indices.shape[0], num_noise_features) * sigma).to(device)
            data.pos[indices, num_node_features - num_noise_features:num_node_features] = \
                data.pos[indices, num_node_features - num_noise_features:num_node_features] + noise

        return data

    @staticmethod
    def add_noise_to_pcd_points(data: Data, sigma: float, device):
        """
        Adds training noise to the point cloud positions with standard deviation sigma
        Args:
            data: PyG data element containing (a batch of) graph(s)
            sigma: standard deviation of used noise
            device: working device (cuda or cpu)

        Returns:
            data: updated graph with noise

        """
        if sigma > 0.0:
            indices = torch.where(data.node_type == NodeType.MESH)[0]
            num_noise_features = data.pos.shape[1]
            num_node_features = data.pos.shape[1]
            noise = (torch.randn(indices.shape[0], num_noise_features) * sigma).to(device)
            data.pos[indices, num_node_features - num_noise_features:num_node_features] = data.pos[indices,
                                                                                          num_node_features - num_noise_features:num_node_features] + noise
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


