import numpy as np
import torch

from torch_geometric.data import Batch
from typing import Callable, Union, Tuple
from torch import nn, Tensor

from src.modules.gru.GRU import GRU
from src.modules.rkn.RKN import RKN, elup1
from src.util.types import NodeType


class Decoder(nn.Module):
    """Decodes node features from graph."""

    def __init__(self, output_size: int, node_type: str, latent_size: int, rnn_type: Union[bool, str], self_sup: bool = False):
        super().__init__()
        self.node_type = node_type
        self.recurrence = rnn_type is not False
        self.latent_size = latent_size
        self.self_sup = self_sup
        self.use_u = output_size == 1

        self.model = nn.Sequential(nn.LazyLinear(latent_size), nn.LeakyReLU(), nn.LazyLinear(output_size))

        if self.recurrence:
            self.rnn = get_RNN(rnn_type, self.latent_size)
            self.log_var_model = nn.Sequential(nn.LazyLinear(latent_size), nn.LeakyReLU(), nn.LazyLinear(output_size))

    def forward(self, graph: Batch) -> Tuple[Tensor, Union[None, Tensor]]:
        graph.u, post_var, h, c = self.rnn(graph)
        x_hat = self.transform_nodes(graph)

        ex = False
        if True in torch.isnan(x_hat):
            ex = True
            print("x_hat")
            for param in self.rnn.parameters():
                print(param.data)

        if True in torch.isinf(x_hat):
            ex = True
            print("x_hatinf")
            for param in self.rnn.parameters():
                print(param.data)


        mean = self.model(x_hat)
        # batch = self.norm(batch)
        if True in torch.isnan(mean):
            ex = True
            print("mean")
            for param in self.model.parameters():
                print(param.data)

        if True in torch.isinf(mean):
            ex = True
            print("meaninf")
            for param in self.model.parameters():
                print(param.data)

        if post_var is not None:
            log_var = self.log_var_model(torch.cat(post_var, dim=-1))
            var = elup1(log_var)
            c = torch.stack(c, dim=0)
            if True in torch.isnan(log_var):
                ex = True
                print("log_var")
                for param in self.log_var_model.parameters():
                    print(param.data)

            if True in torch.isinf(log_var):
                ex = True
                print("log_varinf")
                for param in self.log_var_model.parameters():
                    print(param.data)
        else:
            var = None



        if True in torch.isnan(var):
            ex = True
            print("var")

        if True in torch.isinf(var):
            ex = True
            print("varinf")

        y_hat = (mean, var)


        if ex:
            exit()

        return y_hat, (h, c)

    def transform_nodes(self, graph):
        if self.use_u:
            return graph.u

        mask = torch.where(graph[self.node_type].node_type == NodeType.MESH)[0]
        node_features = graph[self.node_type].x[mask]

        if self.self_sup:
            batch = graph[self.node_type].batch
            # TODO: integrate cov
            return torch.cat([node_features, graph.u[batch][mask]], dim=-1)

        return node_features

def get_RNN(rnn_type, latent_size):
    if str(rnn_type).lower() == 'gru':
        return GRU(latent_size)
    elif str(rnn_type).lower() == 'rkn':
        return RKN(int(latent_size / 2))
    else:
        raise NotImplementedError("Implement your RNN cells here!")
