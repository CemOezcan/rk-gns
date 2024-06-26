import enum
import numpy as np

from typing import Dict, Any, List, Union, Iterable, Callable, Optional, Tuple, Generator
from torch import Tensor
from torch_geometric.data.batch import Batch
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.data.data import Data


"""
Custom class that redefines various types to increase clarity.
"""
Key = Union[str, int]  # for dictionaries, we usually use strings or ints as keys
ConfigDict = Dict[Key, Any]  # A (potentially nested) dictionary containing the "params" section of the .yaml file
RecordingDict = Dict[Key, Any]  # Alias for recording dicts
EntityDict = Dict[Key, Union[Dict, str]]  # potentially nested dictionary of entities
ScalarDict = Dict[Key, float]  # dict of scalar values
ValueDict = Dict[Key, Any]
Result = Union[List, int, float, np.ndarray]

InputBatch = Union[Dict[Key, Tensor], Tensor, Batch, Data, HeteroData, None]
OutputTensorDict = Dict[Key, Tensor]
AGGR_AGGR = 'aggregation'
CONCAT_AGGR = 'concatenation'

class NodeType(enum.IntEnum):
    MESH = 0
    COLLIDER = 1
    POINT = 2
    SHAPE = 3
