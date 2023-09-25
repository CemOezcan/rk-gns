import copy

import torch

from typing import Tuple
from abc import ABC, abstractmethod
from torch import nn, Tensor

from src.modules.mesh_graph_nets import MeshGraphNets
from src.modules.normalizer import Normalizer
from src.util.types import *
from src.util.util import device
import torch_geometric.transforms as T


class AbstractSystemModel(ABC, nn.Module):
    """
    Superclass for neural system model estimators
    """

    def __init__(self, params: ConfigDict) -> None:
        super(AbstractSystemModel, self).__init__()
        self._params = params
        self.loss_fn = torch.nn.MSELoss()

        self._output_normalizer = Normalizer(name='output_normalizer')
        self._mesh_edge_normalizer = Normalizer(name='mesh_edge_normalizer')
        self._feature_normalizer = Normalizer(name='node_normalizer')

        self.message_passing_steps = params.get('message_passing_steps')
        self.message_passing_aggregator = params.get('aggregation')

        self._edge_sets = [''.join(('mesh', '0', 'mesh'))]
        self._node_sets = ['mesh']

        self.euclidian_distance = True
        self.hetero = params.get('heterogeneous')
        self.input_mesh_noise = params.get('noise')
        self.input_pcd_noise = params.get('pc_noise')
        self.feature_norm = params.get('feature_norm')
        self.layer_norm = params.get('layer_norm')
        self.num_layers = params.get('layers')



    def get_target(self, graph: Batch, is_training: bool) -> Tensor:
        """
        Computes and normalizes the target variable.

        Parameters
        ----------
            graph : Batch
                A data instance

            is_training : bool
                Whether the input is a training instance or not

        Returns
        -------
        Normalized velocity

        """
        mask = torch.where(graph['mesh'].node_type == NodeType.MESH)[0]
        target_velocity = graph.y - graph['mesh'].pos[mask]

        return self._output_normalizer(target_velocity, is_training)

    def update(self, inputs: Batch, per_node_network_output: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Makes a prediction for the given input data and uses it to compute the predicted system state.

        Parameters
        ----------
            inputs : Dict[str, Tensor]
                Dictionary containing the current and the previous system states

            per_node_network_output : Tensor
                Predicted (normalized) system dynamics

        Returns
        -------
            Tensor
                Some representation of the predicted system state
        """
        mask = torch.where(inputs['mesh'].node_type == NodeType.MESH)[0]
        velocity = self._output_normalizer.inverse(per_node_network_output)

        # integrate forward
        cur_position = inputs['mesh'].pos[mask]

        # vel. = next_pos - cur_pos
        position = cur_position + velocity

        return (position, cur_position, velocity)

    def build_graph(self, data: Data, is_training: bool) -> HeteroData:
        """
        Constructs the input graph given a system state.

        Parameters
        ----------
            data : Tuple[Data, Data]
                System state information

            is_training : bool
                Whether the input is a training instance or not

            keep_point_cloud: Union[bool, None]
                Whether to keep point cloud nodes or not

        Returns
        -------
            The system state represented by a graph

        """
        if is_training:
            data = self.add_noise(data, self.input_mesh_noise, NodeType.MESH)
        data = self.add_noise(data, self.input_pcd_noise, NodeType.POINT)
        data = self.transform_position_to_edges(data, self.euclidian_distance)

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        node_attr = data.x
        node_type = data.node_type
        edge_type = data.edge_type

        # Create a HeteroData object
        hetero_data = HeteroData().cpu()

        # Add node data to the HeteroData object
        hetero_data[self._node_sets[0]].x = node_attr
        hetero_data[self._node_sets[0]].node_type = node_type
        hetero_data[self._node_sets[0]].pos = data.pos
        hetero_data[self._node_sets[0]].next_pos = data.next_pos

        # Add edge data to the HeteroData object
        hetero_data[('mesh', '0', 'mesh')].edge_index = edge_index
        hetero_data[('mesh', '0', 'mesh')].edge_attr = edge_attr
        hetero_data[('mesh', '0', 'mesh')].edge_type = edge_type

        hetero_data.u = data.u
        hetero_data.poisson = data.poisson
        hetero_data.h = data.h
        hetero_data.y = data.y
        hetero_data.cpu()

        return hetero_data

    @abstractmethod
    def training_step(self, graph: Batch) -> Tensor:
        """
        Perform a single training step.

        Parameters
        ----------
            graph : MultiGraph
                The current system state represented by a (heterogeneous hyper-) graph

            data_frame : Dict[str, Tensor]
                Additional information on the instance and the target system state

        Returns
        -------
            Tensor
                The training loss
        """

        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def validation_step(self, graph: Batch, data_frame: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Evaluate given input data and potentially auxiliary information to create a dictionary of resulting values.
        What kinds of things are scored/evaluated depends on the concrete algorithm.

        Parameters
        ----------
            graph : MultiGraph
                The current system state represented by a (heterogeneous hyper-) graph

            data_frame : Dict[str, Tensor]
                Additional information on the instance and the target system state

        Returns
        -------
            Tensor
                A dictionary with different values that are evaluated from the given input data. May e.g., the
                accuracy of the model.
        """

        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def rollout(self, trajectory: List[Dict[str, Tensor]], num_steps: int) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Predict a sub trajectory for n time steps by making n consecutive one-step predictions recursively.

        Parameters
        ----------
            trajectory : Dict[str, Tensor]
                A trajectory of subsequent system states

            num_steps : int
                Number of time steps

        Returns
        -------
            Tuple[Dict[str, Tensor], Tensor]
                The predicted and the ground truth trajectories as well as the corresponding losses for each time step

        """

        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def n_step_computation(self, trajectory: List[Dict[str, Tensor]], n_step: int) -> Tuple[Tensor, Tensor]:
        """
        Predict the system state after n time steps. N step predictions are performed recursively within trajectories.

        Parameters
        ----------
            trajectory : Dict[str, Tensor]
                A trajectory of subsequent system states

            n_step: int
                Number of time steps

        Returns
        -------
            Tensor
                The n-step loss

        """

        raise NotImplementedError

    def evaluate(self) -> None:
        """
        Activate evaluation mode.

        Returns
        -------

        """
        self.eval()
        self.learned_model.eval()

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

    @staticmethod
    def split_graphs(graph):
        pc_mask = torch.where(graph['mesh'].node_type == NodeType.POINT)[0]
        shape_mask = torch.where(graph['mesh'].node_type == NodeType.SHAPE)[0]
        obst_mask = torch.where(graph['mesh'].node_type == NodeType.COLLIDER)[0]
        mesh_mask = torch.where(graph['mesh'].node_type == NodeType.MESH)[0]

        poisson_mask = torch.cat([shape_mask, obst_mask], dim=0)
        mgn_mask = torch.cat([mesh_mask, pc_mask, obst_mask], dim=0)

        pc = graph.subgraph({'mesh': poisson_mask}).clone()
        #pc['mesh'].x = torch.cat([pc['mesh'].pos, pc['mesh'].x], dim=1)
        mesh = graph.subgraph({'mesh': mgn_mask}).clone()

        return mesh, pc



