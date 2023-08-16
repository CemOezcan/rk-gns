import copy
import math
import os
import pickle
import random
import numpy as np
import scipy
import torch
import torch_cluster
from torch_cluster import fps
import torch_geometric.transforms as T

from typing import Dict, Tuple, Union, List
from torch import Tensor
from torch_geometric.data import Data
from tqdm import tqdm

from src.util.types import NodeType, ConfigDict


class Preprocessing:
    """
    Class for preprocessing raw datasets into trajectories of system states.
    """

    def __init__(self, split: str, path: str, raw: bool, config: ConfigDict):
        """
        Initialization

        Parameters
        ----------
            split: str
                Name of the split.
            path: str
                Dataset path.
            raw: bool
                Whether to keep or split trajectories into individual data instances.
            config: ConfigDict
                Contains the configuration file.
        """
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

    def build_dataset_for_split(self) -> List[Union[List[Data], Data]]:
        """

        Returns
        -------

        """
        print(f'Generating {self.split} data')
        with open(self.directory, 'rb') as file:
            rollout_data = pickle.load(file)

        random.shuffle(rollout_data)
        rollout_data = rollout_data[:self.trajectories]

        trajectory_list = [self.compute(args) for args in rollout_data]

        trajectory_list = trajectory_list if self.raw else self.split_data(trajectory_list)

        return trajectory_list

    def compute(self, traj):
        rollout_length = len(traj['nodes_grid'])
        data_list = []

        for timestep in range(rollout_length - 2):
            data_timestep = self.extract_system_parameters(traj, timestep)
            data = self.create_graph(data_timestep)

            if not self.raw:
                data = Preprocessing.postprocessing(Data.from_dict(data))

            data_list.append(data)

        return data_list

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

        instance['mesh_pos'] = torch.tensor(data['pcd_points'][timestep])
        instance['target_mesh_pos'] = torch.tensor(data['pcd_points'][timestep + 1])
        instance['init_mesh_pos'] = torch.tensor(data['pcd_points'][0])

        instance['target_collider_pos'] = torch.tensor(data['nodes_collider'][timestep + 1])
        instance['collider_pos'] = torch.tensor(data['nodes_collider'][timestep])
        instance['init_collider_pos'] = torch.tensor(data['nodes_collider'][0])
        instance['collider_edge_index'] = torch.tensor(data['edge_index_collider'].T).long()
        instance['collider_cells'] = torch.tensor(data['triangles_collider']).long()

        ratio = (instance['init_mesh_pos'].shape[0] - instance['mesh_pos'].shape[0]) / instance['init_mesh_pos'].shape[0]

        count = min(100, int(100 - ratio * 100))
        soft = 100 - count

        nodes = count / instance['mesh_pos'].shape[0]
        index = fps(instance['mesh_pos'], ratio=nodes)

        new_index = list(set(range(int(instance['mesh_pos'].shape[0]))) - set(index))

        norm = torch.nn.Softmax()
        center = torch.mean(instance['collider_pos'], dim=0)
        radius = torch.pairwise_distance(instance['collider_pos'][0], center)
        center[1] = center[1] #- radius / 2
        dist = torch.pairwise_distance(instance['mesh_pos'][new_index], center)

        probs = norm(1 / dist ** 2)
        new_index = np.random.choice(new_index, size=soft, replace=False, p=probs)
        index = list(set(new_index).union(set(index)))[:100]

        instance['mesh_pos'] = instance['mesh_pos'][index]

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
        pos_dict = {'mesh': input_data['mesh_pos'], 'collider': input_data['collider_pos']}
        init_pos_dict = {'mesh': input_data['init_mesh_pos'], 'collider': input_data['init_collider_pos']}
        target_dict = {'mesh': input_data['target_mesh_pos'], 'collider': input_data['target_collider_pos']}

        # build nodes features (one hot)
        num_nodes = [values.shape[0] for values in pos_dict.values()]
        x = self.build_one_hot_features(num_nodes)
        node_type = self.build_type(num_nodes)

        # # used if poisson ratio needed as input feature, but atm incompatible with Imputation training
        poisson_ratio = input_data['poisson_ratio'] if self.use_poisson else torch.tensor([0.0]).reshape(-1, 1)
        target_poisson = input_data['poisson_ratio'].float()

        # target_poisson = list()
        # for ezbez in input_data['poisson_ratio']:
        #     if -0.1 < ezbez < 0.1:
        #         target_poisson.append([0, 1, 0])
        #     elif ezbez > 0:
        #         target_poisson.append([0, 0, 1])
        #     else:
        #         target_poisson.append([1, 0, 0])
        #
        # target_poisson = torch.tensor(target_poisson).float()

        # index shift dict for edge index matrix
        index_shift_dict = {'mesh': 0, 'collider': num_nodes[0]}
        collider_edges = torch.cat((input_data['collider_edge_index'], input_data['collider_edge_index'][[1, 0]]), dim=1)
        collider_edges[:, :] += index_shift_dict['collider']

        edge_index_dict = {('collider', 1, 'collider'): collider_edges}

        num_edges = [value.shape[1] for value in edge_index_dict.values()]
        edge_attr = self.build_one_hot_features(num_edges)
        edge_type = self.build_type(num_edges)

        collider_cells = input_data['collider_cells'][:, :] + index_shift_dict['collider']
        cells_dict = {'collider': collider_cells}

        num_cells = [value.shape[0] for value in cells_dict.values()]
        cell_type = self.build_type(num_cells)

        # create node positions tensor and edge_index from dicts
        pos = torch.cat(tuple(pos_dict.values()), dim=0)
        init_pos = torch.cat(tuple(init_pos_dict.values()), dim=0)
        target = torch.cat(tuple(target_dict.values()), dim=0)
        edge_index = torch.cat(tuple(edge_index_dict.values()), dim=1)
        cells = torch.cat(tuple(cells_dict.values()), dim=0)
        x = torch.cat([pos, x], dim=1)

        # create data object for torch
        data = {'x': x.float(),
                'poisson': target_poisson,
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

        # Add collision edges
        collision_edges = torch_cluster.radius(data.pos[mask], data.pos[obst_mask], r=0.3, max_num_neighbors=100)
        Preprocessing.add_edge_set(data, collision_edges, (len(mask), 0), 2, False)

        triangles = scipy.spatial.Delaunay(data.pos[mask])
        pc_edges = set()
        for simplex in triangles.simplices:
            pc_edges.update((simplex[i], simplex[j]) for i in range(-1, len(simplex)) for j in range(i + 1, len(simplex)))

        coll_set = set(collision_edges[1].tolist())
        short_dist_graph = torch_cluster.radius_graph(data.pos[mask], r=0.35, max_num_neighbors=100).tolist()
        short_edges = [(x, y) for x, y in zip(short_dist_graph[0], short_dist_graph[1]) if x in coll_set and y in coll_set]
        short_pc_edges = set(short_edges).intersection(set(pc_edges))

        long_dist_graph = torch_cluster.radius_graph(data.pos[mask], r=0.6, max_num_neighbors=100).tolist()
        long_edges = [(x, y) for x, y in zip(long_dist_graph[0], long_dist_graph[1]) if x not in coll_set or y not in coll_set]

        pc_edges_copy = set(long_edges).union(short_pc_edges).intersection(set(pc_edges))
        pc_edges = zip(*list(pc_edges_copy))


        # Convert edge indices to PyTorch tensor
        pc_edges = torch.tensor(list(pc_edges), dtype=torch.long)
        Preprocessing.add_edge_set(data, pc_edges, (0, 0), 3, False)

        values = [0] * 4
        for key in data.edge_type:
            values[int(key)] += 1

        data_mgn = copy.deepcopy(data)
        old_edges = data_mgn.edge_type.shape[0]

        data.edge_attr = Preprocessing.build_one_hot_features(values)
        data_mgn.edge_attr = data.edge_attr[:old_edges]
        data_mgn.edge_type = data.edge_type[:old_edges]
        data_mgn.x = data_mgn.x[:point_index]
        data_mgn.pos = data_mgn.pos[:point_index]
        data_mgn.next_pos = data_mgn.next_pos[:point_index]
        data_mgn.node_type = data_mgn.node_type[:point_index]

        return data_mgn, data_mgn

    @staticmethod
    def add_edge_set(data: Data, edges: Tensor, index_shift: Tuple[int], edge_type: int, bidirectional: bool, remove_duplicates: bool = False) -> None:
        """
        Adds a new set of edges to an existing graph
        Parameters
        ----------
            data: Data
                The current graph
            edges: Tensor
                Set of edges
            index_shift: Tuple[int]
                Shift edge indices
            edge_type: int
                Edge type for one-hot encoding
            bidirectional: bool
                Whether to expand edges to be bidirectional
            remove_duplicates: bool
                Whether to check existing edge set for duplicates and removing them if need be

        Returns
        -------
        Transformed graph

        """
        row, col = edges[0], edges[1]
        row, col = row[row != col], col[row != col]
        edges = torch.stack([row, col], dim=0)
        edges[0, :] += index_shift[0]
        edges[1, :] += index_shift[1]

        if remove_duplicates:
            edges = Preprocessing.get_unique_edges(data.edge_index, edges)

        indices = [data.edge_index, edges]
        factor = len(edges[0])

        if bidirectional:
            indices.append(edges[[1, 0]])
            factor *= 2

        data.edge_index = torch.cat(indices, dim=1)
        data.edge_type = torch.cat([data.edge_type, torch.tensor([edge_type] * factor).long()], dim=0)

    @staticmethod
    def get_unique_edges(e_1: Tensor, e_2: Tensor) -> Tensor:
        """
        Adds a new set of edges to the existing set while removing duplicates.

        Parameters
        ----------
            e_1: Tensor
                Current edge set
            e_2: Tensor
                New edges

        Returns
        -------
            Combined set of edges

        """
        e_1, e_2 = e_1.numpy(), e_2.numpy()
        e_1_set = set((i, j) for i, j in zip(*e_1))
        e_2_set = set((i, j) for i, j in zip(*e_2))

        unique_edges = e_2_set - e_1_set

        if len(unique_edges) == 0:
            return torch.tensor([[], []], dtype=torch.int64)
        else:
            return torch.tensor(list(unique_edges), dtype=torch.int64).t()