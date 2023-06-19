
import collections
import enum
import torch
import yaml
from torch_geometric.utils import scatter
import numpy as np
from typing import Dict, Any, List, Union, Optional
import numpy as np

device = torch.device('cpu')# torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def detach(tensor: torch.Tensor) -> np.array:

    if tensor.is_cuda:
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()


def read_yaml(config_name: str):
    with open(f'config/{config_name}.yaml', 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load_all(stream)
            for file in parsed_yaml:
                if file['name'] == 'DEFAULT':
                    return file
        except yaml.YAMLError as e:
            print(e)
            return None


def triangles_to_edges(faces, deform=False):
    """Computes mesh edges from triangles."""
    if not deform:
        # collect edges from triangles
        edges = torch.cat((faces[:, 0:2],
                           faces[:, 1:3],
                           torch.stack((faces[:, 2], faces[:, 0]), dim=1)), dim=0)
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single tf.int64
        receivers, _ = torch.min(edges, dim=1)
        senders, _ = torch.max(edges, dim=1)

        packed_edges = torch.stack((senders, receivers), dim=1)
        unique_edges = torch.unique(packed_edges, return_inverse=False, return_counts=False, dim=0)
        senders, receivers = torch.unbind(unique_edges, dim=1)
        senders = senders.to(torch.int64)
        receivers = receivers.to(torch.int64)

        two_way_connectivity = (torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0))
        return {'two_way_connectivity': two_way_connectivity, 'senders': senders, 'receivers': receivers}
    else:
        edges = torch.cat((faces[:, 0:2],
                           faces[:, 1:3],
                           faces[:, 2:4],
                           torch.stack((faces[:, 3], faces[:, 0]), dim=1)), dim=0)
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single tf.int64
        receivers, _ = torch.min(edges, dim=1)
        senders, _ = torch.max(edges, dim=1)

        packed_edges = torch.stack((senders, receivers), dim=1)
        unique_edges = torch.unique(packed_edges, return_inverse=False, return_counts=False, dim=0)
        senders, receivers = torch.unbind(unique_edges, dim=1)
        senders = senders.to(torch.int64)
        receivers = receivers.to(torch.int64)

        two_way_connectivity = (torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0))
        return {'two_way_connectivity': two_way_connectivity, 'senders': senders, 'receivers': receivers}


def get_from_nested_dict(dictionary: Dict[Any, Any], list_of_keys: List[Any],
                         raise_error: bool = False,
                         default_return: Optional[Any] = None) -> Any:
    """
    Utility function to traverse through a nested dictionary. For a given dict d and a list of keys [k1, k2, k3], this
    will return d.get(k1).get(k2).get(k3) if it exists, and default_return otherwise
    Args:
        dictionary: The dictionary to search through
        list_of_keys: List of keys in the order to traverse the dictionary by
        raise_error: Raises an error if any of the keys is missing in its respective subdict. If false, returns the
        default_return instead
        default_return: The thing to return if the dictionary does not contain the next key at some level

    Returns:

    """
    current_dict = dictionary
    for key in list_of_keys:
        if isinstance(current_dict, dict):  # still traversing dictionaries
            current_dict = current_dict.get(key, None)
        elif current_dict is None:  # key of sub-dictionary not found
            if raise_error:
                raise ValueError("Dict '{}' does not contain list_of_keys '{}'".format(dictionary, list_of_keys))
            else:
                return default_return
    return current_dict  # bottom level reached


#import networkx as nx
#import matplotlib.pyplot as plt

"""# Defining a Class
class GraphVisualization:

    def __init__(self):
        # visual is a list which stores all
        # the set of edges that constitutes a
        # graph
        self.visual = []
        self.nodes = []

    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addNode(self, id, pos):
        print(pos)
        self.nodes.append((id, pos))

    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)

    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list
    # nx.draw_networkx(G) - plots the graph
    # plt.show() - displays the graph
    def visualize(self):
        G = nx.Graph()
        for id, pos in self.nodes:
            G.add_node(id, pos=pos)
        pos = nx.get_node_attributes(G, 'pos')
        G.add_edges_from(self.visual)
        nx.draw(G, pos)
        plt.show()


# Driver code:
# G = GraphVisualization()
# for i in range(len(clustering_features[1])):
#     G.addNode(i + num_nodes, list(clustering_features[1][i][:2]))
# for i in range(len(neighbors)):
#     G.addEdge(int(senders[i]), int(receivers[i]))
#
# G.visualize()"""

