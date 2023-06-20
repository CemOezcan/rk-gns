import math
from typing import Dict, Tuple

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
from src.util.types import NodeType, MultiGraph
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
        self._mesh_edge_normalizer = Normalizer(size=12, name='mesh_edge_normalizer')

        self.message_passing_steps = params.get('message_passing_steps')
        self.message_passing_aggregator = params.get('aggregation')

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
            use_global=False
        ).to(device)

        self.pointcloud_dropout = 1
        self.hetero = False
        self.use_world_edges = False
        self.input_mesh_noise = 0.03
        self.input_pcd_noise = 0.01
        self.euclidian_distance = True

    def build_graph(self, data: Data, is_training: bool) -> Data:
        """Builds input graph."""
        data.to(device)
        if is_training:
            #data = self.add_pointcloud_dropout(data, self.pointcloud_dropout, self.hetero, self.use_world_edges)
            #data.to(device)
            data = self.add_noise_to_mesh_nodes(data, self.input_mesh_noise, device)
        #data = self.add_noise_to_pcd_points(data, self.input_pcd_noise, device)
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
        hetero_data.pos = data.pos
        hetero_data.y = data.y
        hetero_data.to(device)

        return hetero_data

    def forward(self, graph):
        return self.learned_model(graph)

    def training_step(self, graph, data_frame):
        mask = torch.where(graph.node_type == NodeType.MESH)[0]

        pred_velocity = self(graph)[mask]
        target_velocity = graph.y[mask] - graph.pos[mask]

        target_velocity = self._output_normalizer(target_velocity, True)
        loss = self.loss_fn(target_velocity, pred_velocity)

        return loss

    @torch.no_grad()
    def validation_step(self, graph: MultiGraph, data_frame: Dict) -> Tuple[Tensor, Tensor]:
        mask = torch.where(graph.node_type == NodeType.MESH)[0]

        pred_velocity = self(graph)[mask]
        target_velocity = graph.y[mask] - graph.pos[mask]
        # TODO: compute target with or without noise?

        target_velocity = self._output_normalizer(target_velocity, False)
        error = self.loss_fn(target_velocity, pred_velocity).cpu()

        pred_position, _, _ = self.update(graph, pred_velocity)
        pos_error = self.loss_fn(pred_position, graph.y[mask]).cpu()

        return error, pos_error

    def update(self, inputs, per_node_network_output: Tensor) -> Tensor:
        """Integrate model outputs."""
        mask = torch.where(inputs.node_type == NodeType.MESH)[0]
        velocity = self._output_normalizer.inverse(per_node_network_output)

        # integrate forward
        cur_position = inputs.pos[mask]

        # vel. = next_pos - cur_pos
        position = cur_position + velocity

        return (position, cur_position, velocity)

    @torch.no_grad()
    def rollout(self, trajectory: Dict[str, Tensor], num_steps: int) -> Tuple[Dict[str, Tensor], Tensor]:
        """Rolls out a model trajectory."""
        num_steps = len(trajectory) if num_steps is None else num_steps
        initial_state = trajectory[0] # {k: torch.squeeze(v, 0)[0] for k, v in trajectory.items()}

        node_type = initial_state['node_type']
        mask = torch.where(node_type == NodeType.MESH)[0].to(device)
        point_index = initial_state['point_index']

        cur_pos = torch.squeeze(initial_state['pos'], 0).to(device)
        target_pos = [t['y'].to(device) for t in trajectory]
        pred_trajectory = []
        cur_positions = []
        cur_velocities = []
        for step in range(num_steps):
            cur_pos,  pred_trajectory, cur_positions, cur_velocities = \
                self._step_fn(initial_state, cur_pos, pred_trajectory, cur_positions,
                              cur_velocities, target_pos[step], step, mask, num_steps)

        prediction = torch.stack([x[:point_index].cpu() for x in pred_trajectory][:num_steps]).cpu()
        gt_pos = torch.stack([t['pos'][:point_index].cpu() for t in trajectory][:num_steps]).cpu()

        traj_ops = {
            'cells': trajectory[0]['cells'],
            'cell_type': trajectory[0]['cell_type'],
            'node_type': node_type,
            'gt_pos': gt_pos,
            'pred_pos': prediction
        }

        mse_loss_fn = torch.nn.MSELoss(reduction='none')
        mse_loss = mse_loss_fn(gt_pos[:, mask].cpu(), prediction[:, mask].cpu())
        mse_loss = torch.mean(torch.mean(mse_loss, dim=-1), dim=-1).detach()

        return traj_ops, mse_loss

    @torch.no_grad()
    def _step_fn(self, initial_state, cur_pos, trajectory, cur_positions, cur_velocities, target_world_pos, step, mask, num_steps):
        input = {**initial_state, 'pos': cur_pos, 'y': target_world_pos}
        data = Preprocessing.postprocessing(Data.from_dict(input).cpu())
        graph = self.build_graph(data, is_training=False)

        prediction, cur_position, cur_velocity = self.update(data.to(device), self(graph)[mask])
        next_pos = target_world_pos
        next_pos[mask] = prediction

        trajectory.append(next_pos)
        cur_positions.append(cur_position)
        cur_velocities.append(cur_velocity)
        return next_pos, trajectory, cur_positions, cur_velocities

    @torch.no_grad()
    def n_step_computation(self, trajectory: Dict[str, Tensor], n_step: int, num_timesteps=None) -> Tuple[Tensor, Tensor]:
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
    def add_pointcloud_dropout(data: Data, pointcloud_dropout: float, hetero: bool, use_world_edges=False) -> Data:
        """
        Randomly drops the pointcloud (with nodes and edges) for the input batch. A bit hacky
        data.batch and data.ptr are used
        Args:
            data: PyG data element containing (a batch of) heterogeneous or homogeneous graph(s)
            pointcloud_dropout: Probability of dropping the point cloud for a batch
            hetero: Use hetero data
            use_world_edges: Use world edges

        Returns:
            data: updated data element
        """
        x = np.random.rand(1)
        if x < pointcloud_dropout:
            # node and edge types to keep
            node_types = [1, 2]
            if use_world_edges:
                edge_types = [1, 2, 5, 8, 9]
            else:
                edge_types = [1, 2, 5, 8]

            # extract correct edge indices
            edge_indices = []
            for edge_type in edge_types:
                edge_indices.append(torch.where(data.edge_type == edge_type)[0])
            edge_indices = torch.cat(edge_indices, dim=0)

            # create index shift lists for edge index
            num_node_type = []
            num_node_type_0 = []
            graph_pointer = []
            for batch in range(int(torch.max(data.batch) + 1)):
                batch_data = data.node_type[data.batch == batch]
                num_node_type_0.append(len(batch_data[batch_data == 0]))
                graph_pointer.append(len(batch_data[batch_data == 1]) + len(batch_data[batch_data == 2]))
                num_node_type.append(len(batch_data))

            num_node_type_0 = list(np.cumsum(num_node_type_0))
            num_node_type = list(np.cumsum(num_node_type))
            num_node_type = [0] + num_node_type
            graph_pointer = [0] + list(np.cumsum(graph_pointer))

            # extract correct node indices (in batch order)
            # therefore the index shift list num_node_type is needed
            # to_heterogeneous does not care about batch indices, so to make this work, we need to keep the order of the batch when extracting the mesh only data
            node_indices = []
            for batch in range(int(torch.max(data.batch) + 1)):
                batch_data = data.node_type[data.batch == batch]
                for node_type in node_types:
                    node_indices.append(torch.where(batch_data == node_type)[0] + num_node_type[batch])
            node_indices = torch.cat(node_indices, dim=0)

            # create updated tensors
            new_pos = data.pos[node_indices]
            new_x = data.x[node_indices]
            new_batch = data.batch[node_indices]
            new_node_type = data.node_type[node_indices]
            new_edge_index = data.edge_index[:, edge_indices]
            new_edge_type = data.edge_type[edge_indices]

            # shift indices for updated edge_index tensor:
            for index in range(len(num_node_type_0)):
                new_edge_index = torch.where(
                    torch.logical_and(new_edge_index > num_node_type[index], new_edge_index < num_node_type[index + 1]),
                    new_edge_index - num_node_type_0[index], new_edge_index)

            # update data object
            data.pos = new_pos
            data.x = new_x
            data.batch = new_batch
            data.node_type = new_node_type
            data.edge_index = new_edge_index
            data.edge_type = new_edge_type
            data.ptr = torch.tensor(graph_pointer)

            # edge_attr are only used for homogeneous graphs at this stage
            if not hetero:
                new_edge_attr = data.edge_attr[edge_indices]
                data.edge_attr = new_edge_attr

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


