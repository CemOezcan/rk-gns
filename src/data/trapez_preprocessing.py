import os
import pickle
from typing import Dict, Tuple
import torch_geometric.transforms as T

import torch
import torch_cluster
from torch import Tensor
from torch_geometric.data import Data
from tqdm import tqdm

from src.util.types import ConfigDict
from src.util.util import device


class TrapezPreprocessing:

    def __init__(self, split, path, raw):
        self.euclidian_distance = True

        self.hetero = False
        self.use_poisson = False
        self.split = split
        self.path = path
        self.raw = raw

        # dataset parameters
        self.dataset = 'coarse_full_graph_t'
        self.input_dataset = 'deformable_plate'# 'trapez_materials_contact_voxel'
        self.use_color = False
        self.use_mesh_coordinates = True

    def build_dataset_for_split(self):
        print(f"Generating {self.split} data")
        #with open(os.path.join(self.path, "data/trapez/input", self.input_dataset + "_" + self.split + ".pkl"), "rb") as file:
        with open(os.path.join(self.path, "data/deformable_plate/input", self.input_dataset + "_" + self.split + ".pkl"),
                      "rb") as file:
            rollout_data = pickle.load(file)
        trajectory_list = []

        for index, trajectory in enumerate(tqdm(rollout_data)):
            rollout_length = len(trajectory["nodes_grid"])
            data_list = []

            for timestep in range(rollout_length - 2):
                # get trajectory data for current timestep
                data_timestep = self.prepare_data_for_trajectory(trajectory, timestep)

                # create nearest neighbor graph with the given radius dict
                data = self.create_graph_from_raw(data_timestep)

                data_list.append(data)  # append object for timestep t to data_list

            trajectory_list.append(data_list)  # create list of trajectories with each trajectory being a list itself

        trajectory_list = self.squash_data(trajectory_list) if self.raw else self.convert_trajectory_to_data_list(trajectory_list)

        return trajectory_list

    def prepare_data_for_trajectory(self, data: Dict, timestep: int) -> Dict:
        # Transpose: edge list to sender, receiver list
        instance = dict()
        instance['pcd_pos'] = torch.tensor(data["pcd_points"][timestep])
        instance['target_pcd_pos'] = torch.tensor(data["pcd_points"][timestep + 1]).long()

        instance['mesh_pos'] = torch.tensor(data["nodes_grid"][timestep])
        instance['target_mesh_pos'] = torch.tensor(data["nodes_grid"][timestep + 1])
        instance['init_mesh_pos'] = torch.tensor(data["nodes_grid"][0])
        instance['mesh_edge_index'] = torch.tensor(data["edge_index_grid"].T).long()
        instance['mesh_cells'] = torch.tensor(data["triangles_grid"]).long()

        instance['target_collider_pos'] = torch.tensor(data["nodes_collider"][timestep + 1])
        instance['collider_pos'] = torch.tensor(data["nodes_collider"][timestep])
        instance['init_collider_pos'] = torch.tensor(data["nodes_collider"][0])
        instance['collider_edge_index'] = torch.tensor(data["edge_index_collider"].T).long()
        instance['collider_cells'] = torch.tensor(data["triangles_collider"]).long()

        return instance

    def create_graph_from_raw(self, input_data) -> Data:

        # dictionary for positions
        pos_dict = {'mesh': input_data['mesh_pos'], 'collider': input_data['collider_pos']}
        init_pos_dict = {'mesh': input_data['init_mesh_pos'], 'collider': input_data['init_collider_pos']}
        target_dict = {'mesh': input_data['target_mesh_pos'], 'collider': input_data['target_collider_pos']}

        # build nodes features (one hot)
        num_nodes = [values.shape[0] for values in pos_dict.values()]
        x = self.build_one_hot_features(num_nodes)
        node_type = self.build_type(num_nodes)

        # # used if poisson ratio needed as input feature, but atm incompatible with Imputation training
        poisson_ratio = torch.tensor([1.0])

        # index shift dict for edge index matrix
        index_shift_dict = {'mesh': 0, 'collider': num_nodes[0]}
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
                'u': poisson_ratio,
                'pos': pos.float(),
                'init_pos': init_pos,
                'edge_index': edge_index.long(),
                'edge_attr': edge_attr.float(),
                'cells': cells.long(),
                'y': target.float(),
                'y_old': input_data['mesh_pos'].float(),
                'node_type': node_type,
                'edge_type': edge_type,
                'cell_type': cell_type}

        return data

    def squash_data(self, trajectories):
        squashed_trajectories = list()
        for trajectory in trajectories:
            data_dict = dict()
            for key in trajectory[0].keys():
                data_dict[key] = torch.stack([data[key] for data in trajectory], dim=0)
            squashed_trajectories.append(data_dict)

        return squashed_trajectories


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
    def convert_trajectory_to_data_list(trajectory_list: list, start_index=0) -> list:
        """
        Converts a list of trajectories (list of time step data) to a single sequential list of all time steps
        Args:
            trajectory_list: List of trajectories
            start_index: Where to start a trajectory default: 0, at the beginning
        Returns:
            data_list: One list of all time steps
        """
        data_list = []
        for trajectory in trajectory_list:
            for index, data in enumerate(trajectory):
                if index >= start_index:
                    data_list.append(Data.from_dict(data))

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
        mesh_attr = TrapezPreprocessing.get_relative_mesh_positions(mesh_edge_index, initial_mesh_positions).float()
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





