import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from src.util import util
from src.modules.mesh_graph_nets import MeshGraphNets
from src.modules.normalizer import Normalizer
from src.model.abstract_system_model import AbstractSystemModel
from src.util.util import NodeType, device
from src.util.types import EdgeSet, MultiGraph
from torch import nn, Tensor

from src.util.types import ConfigDict


class FlagModel(AbstractSystemModel):
    """
    Model for static flag simulation.
    """

    def __init__(self, params: ConfigDict):
        super(FlagModel, self).__init__(params)
        self.loss_fn = torch.nn.MSELoss()

        self._output_normalizer = Normalizer(size=3, name='output_normalizer')
        self._node_normalizer = Normalizer(size=5, name='node_normalizer')
        self._node_dynamic_normalizer = Normalizer(size=1, name='node_dynamic_normalizer')
        self._mesh_edge_normalizer = Normalizer(size=7, name='mesh_edge_normalizer')
        self._intra_edge_normalizer = Normalizer(size=7, name='intra_edge_normalizer')
        self._inter_edge_normalizer = Normalizer(size=7, name='inter_edge_normalizer')
        self._hyper_node_normalizer = Normalizer(size=3, name='hyper_node_normalizer')

        self._model_type = 'flag'
        self.message_passing_steps = params.get('message_passing_steps')
        self.message_passing_aggregator = params.get('aggregation')

        self._edge_sets = [''.join(('mesh_nodes', 'mesh_edges', 'mesh_nodes'))]
        self._node_sets = ['mesh_nodes']

        self.learned_model = MeshGraphNets(
            output_size=params.get('size'),
            latent_size=128,
            num_layers=2,
            message_passing_steps=self.message_passing_steps,
            message_passing_aggregator=self.message_passing_aggregator,
            edge_sets=self._edge_sets,
            node_sets=self._node_sets,
            dec=self._node_sets[0],
            use_global=True
        ).to(device)

    def build_graph(self, inputs: Dict, is_training: bool) -> Data:
        """Builds input graph."""
        world_pos = inputs['world_pos']
        prev_world_pos = inputs['prev|world_pos']
        node_type = inputs['node_type']
        velocity = world_pos - prev_world_pos

        node_types = torch.flatten(torch.ne(node_type[:, 0], 0)).long()
        one_hot_node_type = F.one_hot(node_types)
        node_features = torch.cat((velocity, one_hot_node_type), dim=-1)

        cells = inputs['cells']
        decomposed_cells = util.triangles_to_edges(cells)
        senders, receivers = decomposed_cells['two_way_connectivity']

        mesh_pos = inputs['mesh_pos']
        relative_world_pos = (torch.index_select(input=world_pos, dim=0, index=senders) -
                              torch.index_select(input=world_pos, dim=0, index=receivers))
        relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                             torch.index_select(mesh_pos, 0, receivers))
        edge_features = torch.cat(
            (
                relative_world_pos,
                torch.sqrt(relative_world_pos.pow(2).sum(-1, keepdim=True)),
                relative_mesh_pos,
                torch.sqrt(relative_mesh_pos.pow(2).sum(-1, keepdim=True))
            ), dim=-1
        )

        mesh_edges = EdgeSet(
            name='mesh_edges',
            features=self._mesh_edge_normalizer(edge_features, is_training),
            receivers=receivers,
            senders=senders
        )
        graph = MultiGraph(node_features=[self._node_normalizer(node_features, is_training)], edge_sets=[mesh_edges])
        edge_index = torch.stack((graph.edge_sets[0].senders, graph.edge_sets[0].receivers), dim=0)
        node_features = graph.node_features[0]
        loss_mask = torch.eq(node_type[:, 0], torch.tensor([NodeType.NORMAL.value], device=device).int())
        graph = Data(x=node_features, edge_index=edge_index, edge_attr=graph.edge_sets[0].features,
                     y=self.get_target(inputs, is_training), u=torch.Tensor([[0, 0, 0]]), mask=loss_mask,
                     cur_pos=inputs['world_pos'], prev_pos=inputs['prev|world_pos'], next_pos=inputs['target|world_pos'])
        graph = graph.to_heterogeneous(node_type_names=['mesh_nodes'], edge_type_names=[('mesh_nodes', 'mesh_edges', 'mesh_nodes')])

        return graph

    def forward(self, graph):
        return self.learned_model(graph)

    def training_step(self, graph, data_frame):
        network_output = self(graph)
        target_normalized = graph['mesh_nodes'].y

        loss_mask = graph['mesh_nodes'].mask
        loss = self.loss_fn(target_normalized[loss_mask], network_output[loss_mask])

        return loss

    @torch.no_grad()
    def validation_step(self, graph: MultiGraph, data_frame: Dict) -> Tuple[Tensor, Tensor]:
        prediction = self(graph)
        target_normalized = graph['mesh_nodes'].y

        loss_mask = graph['mesh_nodes'].mask
        acc_loss = self.loss_fn(target_normalized[loss_mask], prediction[loss_mask]).item()

        predicted_position = self.update(graph, prediction)
        pos_error = self.loss_fn(graph['mesh_nodes'].next_pos[loss_mask], predicted_position[loss_mask]).item()

        return acc_loss, pos_error

    def update(self, inputs, per_node_network_output: Tensor) -> Tensor:
        """Integrate model outputs."""
        acceleration = self._output_normalizer.inverse(per_node_network_output)

        # integrate forward
        cur_position = inputs['mesh_nodes'].cur_pos
        prev_position = inputs['mesh_nodes'].prev_pos

        # vel. = cur_pos - prev_pos
        position = 2 * cur_position + acceleration - prev_position

        return position

    def get_target(self, data_frame, is_training=True):
        cur_position = data_frame['world_pos']
        prev_position = data_frame['prev|world_pos']
        target_position = data_frame['target|world_pos']

        # next_pos = cur_pos + acc + vel <=> acc = next_pos - cur_pos - vel | vel = cur_pos - prev_pos
        target_acceleration = target_position - 2 * cur_position + prev_position

        return self._output_normalizer(target_acceleration, is_training).to(device)

    @torch.no_grad()
    def rollout(self, trajectory: Dict[str, Tensor], num_steps: int) -> Tuple[Dict[str, Tensor], Tensor]:
        """Rolls out a model trajectory."""
        num_steps = trajectory['cells'].shape[0] if num_steps is None else num_steps
        initial_state = {k: torch.squeeze(v, 0)[0] for k, v in trajectory.items()}

        node_type = initial_state['node_type']
        mask = torch.eq(node_type[:, 0], torch.tensor([NodeType.NORMAL.value], device=device))
        mask = torch.stack((mask, mask, mask), dim=1)

        prev_pos = torch.squeeze(initial_state['prev|world_pos'], 0)
        cur_pos = torch.squeeze(initial_state['world_pos'], 0)

        pred_trajectory = list()
        for i in range(num_steps):
            prev_pos, cur_pos, pred_trajectory = \
                self._step_fn(initial_state, prev_pos, cur_pos, pred_trajectory, mask, i)

        predictions = torch.stack(pred_trajectory)

        traj_ops = {
            'faces': trajectory['cells'],
            'mesh_pos': trajectory['mesh_pos'],
            'gt_pos': trajectory['world_pos'],
            'pred_pos': predictions
        }

        mse_loss_fn = torch.nn.MSELoss(reduction='none')
        mse_loss = mse_loss_fn(trajectory['world_pos'][:num_steps], predictions)
        mse_loss = torch.mean(torch.mean(mse_loss, dim=-1), dim=-1).detach()

        return traj_ops, mse_loss

    @torch.no_grad()
    def _step_fn(self, initial_state, prev_pos, cur_pos, trajectory, mask, step):
        input = {**initial_state, 'prev|world_pos': prev_pos, 'world_pos': cur_pos}

        graph = self.build_graph(input, is_training=False)
        graph = Batch.from_data_list([graph])

        prediction = self.update(graph, self(graph))
        next_pos = torch.where(mask, torch.squeeze(prediction), torch.squeeze(cur_pos))
        trajectory.append(cur_pos)

        return cur_pos, next_pos, trajectory

    @torch.no_grad()
    def n_step_computation(self, trajectory: Dict[str, Tensor], n_step: int, num_timesteps=None) -> Tuple[Tensor, Tensor]:
        mse_losses = list()
        last_losses = list()
        num_timesteps = trajectory['cells'].shape[0] if num_timesteps is None else num_timesteps
        for step in range(num_timesteps - n_step):
            # TODO: clusters/balancers are reset when computing n_step loss
            eval_traj = {k: v[step: step + n_step + 1] for k, v in trajectory.items()}
            prediction_trajectory, mse_loss = self.rollout(eval_traj, n_step + 1)
            mse_losses.append(torch.mean(mse_loss).cpu())
            last_losses.append(mse_loss.cpu()[-1])

        return torch.mean(torch.stack(mse_losses)), torch.mean(torch.stack(last_losses))
