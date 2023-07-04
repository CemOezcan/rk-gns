import copy
import math
import os
import pickle
import random
from typing import Dict, Tuple
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch
import torch_cluster
from torch import Tensor
from torch_geometric.data import Data
from tqdm import tqdm

from src.util.types import NodeType
from src.util.util import get_from_nested_dict


class Preprocessing:

    def __init__(self, split, path, raw, config):

        self.hetero = config.get('model').get('heterogeneous')
        self.use_poisson = config.get('model').get('poisson_ratio')
        self.split = split
        self.path = path
        self.raw = raw
        self.trajectories = config.get('task').get('trajectories')
        self.trajectories = math.inf if isinstance(self.trajectories, str) else self.trajectories

        # dataset parameters
        self.input_dataset = 'deformable_plate'
        self.directory = os.path.join(self.path, self.input_dataset + '_' + self.split + '.pkl')


    def build_dataset_for_split(self):
        print(f'Generating {self.split} data')
        with open(self.directory, 'rb') as file:
            rollout_data = pickle.load(file)

        trajectory_list = []
        for index, trajectory in enumerate(tqdm(rollout_data)):
            rollout_length = len(trajectory['nodes_grid'])
            data_list = []

            if index >= self.trajectories:
                break

            for timestep in range(rollout_length - 2):
                data_timestep = self.extract_system_parameters(trajectory, timestep)
                data = self.create_graph(data_timestep)

                if not self.raw:
                    data = Preprocessing.postprocessing(Data.from_dict(data))

                data_list.append(data)

            trajectory_list.append(data_list)

        trajectory_list = trajectory_list if self.raw else self.split_data(trajectory_list)

        return trajectory_list

    @staticmethod
    def extract_system_parameters(data: Dict, timestep: int) -> Dict:
        """
        Extract relevant parameters regarding the system State for a given instance.

        Parameters
        ----------
            data : Dict
                An entire trajectory of system states

            timestep :
                The desired timestep within the given trajectory

        Returns
        -------
            Dict
                The system parameters of the desired time step.

        """
        # Transpose: edge list to sender, receiver list
        instance = dict()
        instance['poisson_ratio'] = torch.tensor(data['poisson_ratio']).reshape(-1, 1)

        instance['pcd_pos'] = torch.tensor(data['pcd_points'][timestep])
        instance['target_pcd_pos'] = torch.tensor(data['pcd_points'][timestep + 1])

        instance['mesh_pos'] = torch.tensor(data['nodes_grid'][timestep])
        instance['target_mesh_pos'] = torch.tensor(data['nodes_grid'][timestep + 1])
        instance['init_mesh_pos'] = torch.tensor(data['nodes_grid'][0])
        instance['mesh_edge_index'] = torch.tensor(data['edge_index_grid'].T).long()
        instance['mesh_cells'] = torch.tensor(data['triangles_grid']).long()

        instance['target_collider_pos'] = torch.tensor(data['nodes_collider'][timestep + 1])
        instance['collider_pos'] = torch.tensor(data['nodes_collider'][timestep])
        instance['init_collider_pos'] = torch.tensor(data['nodes_collider'][0])
        instance['collider_edge_index'] = torch.tensor(data['edge_index_collider'].T).long()
        instance['collider_cells'] = torch.tensor(data['triangles_collider']).long()

        return instance

    def create_graph(self, input_data: Dict) -> Dict:
        """
        Convert system state parameters into a graph.

        Parameters
        ----------
            input_data: Dict
                Defines the state of a system at a particular time step.

        Returns
        -------
            Dict
                A graph with mesh edges.

        """

        # dictionary for positions
        pos_dict = {'mesh': input_data['mesh_pos'], 'collider': input_data['collider_pos'], 'point': input_data['pcd_pos']}
        init_pos_dict = {'mesh': input_data['init_mesh_pos'], 'collider': input_data['init_collider_pos']}
        target_dict = {'mesh': input_data['target_mesh_pos'], 'collider': input_data['target_collider_pos'], 'point': input_data['target_pcd_pos']}

        # build nodes features (one hot)
        num_nodes = [values.shape[0] for values in pos_dict.values()]
        x = self.build_one_hot_features(num_nodes)
        node_type = self.build_type(num_nodes)

        # # used if poisson ratio needed as input feature, but atm incompatible with Imputation training
        poisson_ratio = input_data['poisson_ratio'] if self.use_poisson else torch.tensor([0.0]).reshape(-1, 1)

        # index shift dict for edge index matrix
        index_shift_dict = {'mesh': 0, 'collider': num_nodes[0], 'point': num_nodes[0] + num_nodes[1]}
        mesh_edges = torch.cat((input_data['mesh_edge_index'], input_data['mesh_edge_index'][[1, 0]]), dim=1)
        mesh_edges[:, :] += index_shift_dict['mesh']

        collider_edges = torch.cat((input_data['collider_edge_index'], input_data['collider_edge_index'][[1, 0]]), dim=1)
        collider_edges[:, :] += index_shift_dict['collider']

        edge_index_dict = {('mesh', 0, 'mesh'): mesh_edges, ('collider', 1, 'collider'): collider_edges}

        num_edges = [value.shape[1] for value in edge_index_dict.values()]
        edge_attr = self.build_one_hot_features(num_edges)
        edge_type = self.build_type(num_edges)

        mesh_cells = input_data['mesh_cells'][:, :] + index_shift_dict['mesh']
        collider_cells = input_data['collider_cells'][:, :] + index_shift_dict['collider']
        cells_dict = {'mesh': mesh_cells, 'collider': collider_cells}

        num_cells = [value.shape[0] for value in cells_dict.values()]
        cell_type = self.build_type(num_cells)

        # create node positions tensor and edge_index from dicts
        pos = torch.cat(tuple(pos_dict.values()), dim=0)
        init_pos = torch.cat(tuple(init_pos_dict.values()), dim=0)
        target = torch.cat(tuple(target_dict.values()), dim=0)
        edge_index = torch.cat(tuple(edge_index_dict.values()), dim=1)
        cells = torch.cat(tuple(cells_dict.values()), dim=0)

        # create data object for torch
        data = {'x': x.float(),
                'u': poisson_ratio.float(),
                'h': poisson_ratio.float(),
                'pos': pos.float(),
                'next_pos': target.float(),
                'point_index': num_nodes[0] + num_nodes[1],
                'init_pos': init_pos,
                'edge_index': edge_index.long(),
                'edge_attr': edge_attr.float(),
                'cells': cells.long(),
                'y': target[:index_shift_dict['collider']].float(),
                'node_type': node_type,
                'edge_type': edge_type,
                'cell_type': cell_type}

        return data

    @staticmethod
    def build_one_hot_features(num_per_type: list) -> Tensor:
        """
        Builds one-hot feature tensor indicating the edge/node type from numbers per type
        Args:
            num_per_type: List of numbers of nodes per type

        Returns:
            features: One-hot features Tensor
        """
        total_num = sum(num_per_type)
        features = torch.zeros(total_num, len(num_per_type))
        for typ in range(len(num_per_type)):
            features[sum(num_per_type[0:typ]): sum(num_per_type[0:typ + 1]), typ] = 1
        return features

    @staticmethod
    def build_type(num_per_type: list) -> Tensor:
        """
        Build node or edge type tensor from list of numbers per type
        Args:
            num_per_type: list of numbers per type

        Returns:
            features: Tensor containing the type as number
        """
        total_num = sum(num_per_type)
        features = torch.zeros(total_num)
        for typ in range(len(num_per_type)):
            features[sum(num_per_type[0:typ]): sum(num_per_type[0:typ + 1])] = typ
        return features

    @staticmethod
    def split_data(trajectory_list: list, start_index=0) -> list:
        """
        Converts a list of trajectories (list of time step data) to a single sequential list of all time steps
        Args:
            trajectory_list: List of trajectories
            start_index: Where to start a trajectory default: 0, at the beginning
        Returns:
            data_list: One list of all time steps
        """
        data_list = []
        random.shuffle(trajectory_list)
        for trajectory in trajectory_list:
            for index, data in enumerate(trajectory):
                if index >= start_index:
                    data_list.append(data)

        return data_list

    @staticmethod
    def add_relative_mesh_positions(edge_attr: Tensor, edge_type: Tensor, input_mesh_edge_index: Tensor,
                                    initial_mesh_positions: Tensor) -> Tensor:
        """
        Adds the relative mesh positions to the mesh edges (in contrast to the world edges) and zero anywhere else.
        Refer to MGN by Pfaff et al. 2020 for more details.
        Args:
            edge_attr: Current edge features
            edge_type: Tensor containing the edges types
            input_mesh_edge_index: Mesh edge index tensor
            initial_mesh_positions: Initial positions of the mesh nodes "mesh coordinates"

        Returns:
            edge_attr: updated edge features
        """
        indices = torch.where(edge_type == 0)[0]  # type 2: mesh edges
        mesh_edge_index = input_mesh_edge_index
        mesh_attr = Preprocessing.get_relative_mesh_positions(mesh_edge_index, initial_mesh_positions).float()
        mesh_positions = torch.zeros(edge_attr.shape[0], mesh_attr.shape[1]).float()
        mesh_positions[indices, :] = mesh_attr
        edge_attr = torch.cat((edge_attr, mesh_positions), dim=1)
        return edge_attr

    @staticmethod
    def get_relative_mesh_positions(mesh_edge_index: Tensor, mesh_positions: Tensor) -> Tensor:
        """
        Transform the positions of the mesh into a relative position encoding along with the Euclidean distance in the edges
        Args:
            mesh_edge_index: Tensor containing the mesh edge indices
            mesh_positions: Tensor containing mesh positions

        Returns:
            edge_attr: Tensor containing the batched edge features
        """
        data = Data(pos=mesh_positions,
                    edge_index=mesh_edge_index)
        transforms = T.Compose([T.Cartesian(norm=False, cat=True), T.Distance(norm=False, cat=True)])
        data = transforms(data)
        return data.edge_attr

    @staticmethod
    def postprocessing(data: Data) -> Tuple[Data, Data]:
        """
        Task specific expansion of the given input graph. Adds different edge types based on neighborhood graphs.
        Convert the resulting graph into a Data object.

        Parameters
        ----------
            data: Data
                Basic graph without additional edge types

        Returns
        -------
            Tuple[Data, Data]
                Tuple containing the basic graph and the expanded graph.

        """
        mask = torch.where(data.node_type == NodeType.MESH)[0]
        obst_mask = torch.where(data.node_type == NodeType.COLLIDER)[0]
        point_index = data.point_index

        # Add world edges
        world_edges = torch_cluster.radius(data.pos[mask], data.pos[obst_mask], r=0.3, max_num_neighbors=100)
        row, col = world_edges[0], world_edges[1]
        row, col = row[row != col], col[row != col]
        world_edges = torch.stack([row, col], dim=0)
        world_edges[0, :] += len(mask)

        data.edge_index = torch.cat([data.edge_index, world_edges], dim=1)
        data.edge_type = torch.cat([data.edge_type, torch.tensor([2] * len(world_edges[0])).long()], dim=0)

        ext_edges = torch_cluster.radius(data.pos[mask], data.pos[mask], r=0.3, max_num_neighbors=100)
        row, col = ext_edges[0], ext_edges[1]
        row, col = row[row != col], col[row != col]
        ext_edges = torch.stack([row, col], dim=0)
        # TODO: Remove duplicates with transforms.remove_duplicated_edges
        ext_edges = Preprocessing.remove_duplicates_with_mesh_edges(data.edge_index, ext_edges)
        data.edge_index = torch.cat([data.edge_index, ext_edges], dim=1)
        data.edge_type = torch.cat([data.edge_type, torch.tensor([3] * len(ext_edges[0])).long()], dim=0)

        data_mgn = copy.deepcopy(data)
        old_edges = data_mgn.edge_type.shape[0]

        world_edges = torch_cluster.radius(data.pos[point_index:], data.pos[obst_mask], r=0.1, max_num_neighbors=100)
        row, col = world_edges[0], world_edges[1]
        row, col = row[row != col], col[row != col]
        world_edges = torch.stack([row, col], dim=0)
        world_edges[0, :] += len(mask)
        world_edges[1, :] += point_index

        data.edge_index = torch.cat([data.edge_index, world_edges, world_edges[[1, 0]]], dim=1)
        data.edge_type = torch.cat([data.edge_type, torch.tensor([4] * (len(world_edges[0]) * 2)).long()], dim=0)

        grounding_edges = torch_cluster.radius(data.pos[mask], data.pos[point_index:], r=0.1, max_num_neighbors=100)
        row, col = grounding_edges[0], grounding_edges[1]
        row, col = row[row != col], col[row != col]
        grounding_edges = torch.stack([row, col], dim=0)
        grounding_edges[0, :] += point_index
        data.edge_index = torch.cat([data.edge_index, grounding_edges, grounding_edges[[1, 0]]], dim=1)
        data.edge_type = torch.cat([data.edge_type, torch.tensor([5] * (len(grounding_edges[0]) * 2)).long()], dim=0)

        pc_edges = torch_cluster.radius(data.pos[point_index:], data.pos[point_index:], r=0.1, max_num_neighbors=100)
        row, col = pc_edges[0], pc_edges[1]
        row, col = row[row != col], col[row != col]
        pc_edges = torch.stack([row, col], dim=0)
        pc_edges[:, :] += point_index
        data.edge_index = torch.cat([data.edge_index, pc_edges], dim=1)
        data.edge_type = torch.cat([data.edge_type, torch.tensor([6] * len(pc_edges[0])).long()], dim=0)

        data.edge_index = torch.cat([data.edge_index, world_edges, world_edges[[1, 0]]], dim=1)
        data.edge_type = torch.cat([data.edge_type, torch.tensor([7] * (len(world_edges[0]) * 2)).long()], dim=0)

        values = [0] * 8
        for key in data.edge_type:
            values[int(key)] += 1

        data.edge_attr = Preprocessing.build_one_hot_features(values)
        mesh_edge_mask = torch.where(data.edge_type == 0)[0]
        data.edge_attr = Preprocessing.add_relative_mesh_positions(data.edge_attr,
                                                                   data.edge_type,
                                                                         data.edge_index[:, mesh_edge_mask],
                                                                   data.init_pos[mask])
        data_mgn.edge_attr = data.edge_attr[:old_edges]
        data_mgn.edge_type = data.edge_type[:old_edges]
        data_mgn.x = data_mgn.x[:point_index]
        data_mgn.pos = data_mgn.pos[:point_index]
        data_mgn.next_pos = data_mgn.next_pos[:point_index]
        data_mgn.node_type = data_mgn.node_type[:point_index]

        return data, data_mgn

    @staticmethod
    def remove_duplicates_with_mesh_edges(mesh_edges: Tensor, world_edges: Tensor) -> Tensor:
        """
        Removes the duplicates with the mesh edges have of the world edges that are created using a nearset neighbor search. (only MGN)
        To speed this up the adjacency matrices are used
        Args:
            mesh_edges: edge list of the mesh edges
            world_edges: edge list of the world edges

        Returns:
            new_world_edges: updated world edges without duplicates
        """
        import torch_geometric.utils as utils
        adj_mesh = utils.to_dense_adj(mesh_edges)
        if world_edges.shape[1] > 0:
            adj_world = utils.to_dense_adj(world_edges)
        else:
            adj_world = torch.zeros_like(adj_mesh)
        if adj_world.shape[1] < adj_mesh.shape[1]:
            padding_size = adj_mesh.shape[1] - adj_world.shape[1]
            padding_mask = torch.nn.ConstantPad2d((0, padding_size, 0, padding_size), 0)
            adj_world = padding_mask(adj_world)
        elif adj_world.shape[1] > adj_mesh.shape[1]:
            padding_size = adj_world.shape[1] - adj_mesh.shape[1]
            padding_mask = torch.nn.ConstantPad2d((0, padding_size, 0, padding_size), 0)
            adj_mesh = padding_mask(adj_mesh)
        new_adj = adj_world - adj_mesh
        new_adj[new_adj < 0] = 0
        new_world_edges = utils.dense_to_sparse(new_adj)[0]
        return new_world_edges
