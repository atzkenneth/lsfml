#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (©) 2023 Kenneth Atz (ETH Zurich)

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, Size, Tensor
from typing import Optional


def weights_init(m):
    """Xavier uniform weight initialization.

    :param m: A list of learnable linear PyTorch modules.
    :type m: [torch.nn.modules.linear.Linear]
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


class EGNN_sparse(MessagePassing):
    """torch geometric message-passing layer for 2D molecular graphs.
    """

    def __init__(self, feats_dim, m_dim=32, dropout=0.1, aggr="add", **kwargs):
        """Initialization of the 2D message passing layer.

        :param feats_dim: Node feature dimension.
        :type feats_dim: int
        :param m_dim: Meessage passing feature dimesnion, defaults to 32
        :type m_dim: int, optional
        :param dropout: Dropout value, defaults to 0.1
        :type dropout: float, optional
        :param aggr: Message aggregation type, defaults to "add"
        :type aggr: str, optional
        """
        assert aggr in {
            "add",
            "sum",
            "max",
            "mean",
        }, "pool method must be a valid option"

        kwargs.setdefault("aggr", aggr)
        super(EGNN_sparse, self).__init__(**kwargs)

        self.feats_dim = feats_dim
        self.m_dim = m_dim

        self.edge_input_dim = feats_dim * 2

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.edge_norm1 = nn.LayerNorm(m_dim)
        self.edge_norm2 = nn.LayerNorm(m_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            nn.SiLU(),
        )

        self.node_norm1 = nn.LayerNorm(feats_dim)
        self.node_norm2 = nn.LayerNorm(feats_dim)

        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        )

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
    ):
        """Forward pass in the mesaage passing fucntion.

        :param x: Node features.
        :type x: Tensor
        :param edge_index: Edge indices.
        :type edge_index: Adj
        :return: Updated node features.
        :rtype: Tensor
        """
        hidden_out = self.propagate(edge_index, x=x)

        return hidden_out

    def message(self, x_i, x_j):
        """Message passing.

        :param x_i: Node n_i.
        :type x_i: Tensor
        :param x_j: Node n_j.
        :type x_j: Tensor
        :return: Message m_ji
        :rtype: Tensor
        """
        m_ij = self.edge_mlp(torch.cat([x_i, x_j], dim=-1))
        return m_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """Overall propagation within the message passing. 

        :param edge_index: Edge indices.
        :type edge_index: Adj
        :return: Updated node features.
        :rtype: Tensor
        """
        # get input tensors
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute("message", coll_dict)
        aggr_kwargs = self.inspector.distribute("aggregate", coll_dict)
        update_kwargs = self.inspector.distribute("update", coll_dict)

        # get messages
        m_ij = self.message(**msg_kwargs)
        m_ij = self.edge_norm1(m_ij)

        # aggregate messages
        m_i = self.aggregate(m_ij, **aggr_kwargs)
        m_i = self.edge_norm2(m_i)

        # get updated node features
        hidden_feats = self.node_norm1(kwargs["x"])
        hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
        hidden_out = self.node_norm2(hidden_out)
        hidden_out = kwargs["x"] + hidden_out

        return self.update((hidden_out), **update_kwargs)


def fourier_encode_dist(x, num_encodings=4, include_self=True):
    """Encoding Euclidian diatomic distances into Fourier features.

    :param x: Distances in Angström.
    :type x: Tensor
    :param num_encodings: Number of sine and cosine functions, defaults to 4
    :type num_encodings: int, optional
    :param include_self: Option to include absolute distance, defaults to True
    :type include_self: bool, optional
    :return: Fourier features. 
    :rtype: Tensor
    """
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x


class EGNN_sparse3D(MessagePassing):
    """torch geometric message-passing layer for 3D molecular graphs.
    """

    def __init__(self, feats_dim, pos_dim=3, m_dim=32, dropout=0.1, fourier_features=16, aggr="add", **kwargs):
        """Initialization of the 3D message passing layer.

        :param feats_dim: Node feature dimension.
        :type feats_dim: int
        :param pos_dim: Dimension of the graph, defaults to 3
        :type pos_dim: int, optional
        :param m_dim: Meessage passing feature dimesnion, defaults to 32
        :type m_dim: int, optional
        :param dropout: Dropout value, defaults to 0.1
        :type dropout: float, optional
        :param fourier_features: Number of Fourier features, defaults to 16
        :type fourier_features: int, optional
        :param aggr: Message aggregation type, defaults to "add"
        :type aggr: str, optional
        """
        assert aggr in {
            "add",
            "sum",
            "max",
            "mean",
        }, "pool method must be a valid option"

        kwargs.setdefault("aggr", aggr)
        super(EGNN_sparse3D, self).__init__(**kwargs)

        # Model parameters
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.fourier_features = fourier_features

        self.edge_input_dim = (self.fourier_features * 2) + 1 + (feats_dim * 2)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.edge_norm1 = nn.LayerNorm(m_dim)
        self.edge_norm2 = nn.LayerNorm(m_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            nn.SiLU(),
        )

        self.node_norm1 = nn.LayerNorm(feats_dim)
        self.node_norm2 = nn.LayerNorm(feats_dim)

        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        )

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
    ):
        """Forward pass in the mesaage passing fucntion.

        :param x: Node features.
        :type x: Tensor
        :param edge_index: Edge indices.
        :type edge_index: Adj
        :return: Updated node features.
        :rtype: Tensor
        """
        coors, feats = x[:, : self.pos_dim], x[:, self.pos_dim :]
        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist = (rel_coors**2).sum(dim=-1, keepdim=True)

        if self.fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings=self.fourier_features)
            rel_dist = rel_dist.squeeze(1)
            # rel_dist = rearrange(rel_dist, "n () d -> n d")

        hidden_out = self.propagate(edge_index, x=feats, edge_attr=rel_dist)
        return torch.cat([coors, hidden_out], dim=-1)

    def message(self, x_i, x_j, edge_attr):
        """Message passing.

        :param x_i: Node n_i.
        :type x_i: Tensor
        :param x_j: Node n_j.
        :type x_j: Tensor
        :param edge_attr: Edge e_{ij}
        :type edge_attr: Tensor
        :return: Message m_ji
        :rtype: Tensor
        """
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return m_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """Overall propagation within the message passing. 

        :param edge_index: Edge indices.
        :type edge_index: Adj
        :return: Updated node features.
        :rtype: Tensor
        """
        # get input tensors
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute("message", coll_dict)
        aggr_kwargs = self.inspector.distribute("aggregate", coll_dict)
        update_kwargs = self.inspector.distribute("update", coll_dict)

        # get messages
        m_ij = self.message(**msg_kwargs)
        m_ij = self.edge_norm1(m_ij)

        # aggregate messages
        m_i = self.aggregate(m_ij, **aggr_kwargs)
        m_i = self.edge_norm2(m_i)

        # get updated node features
        hidden_feats = self.node_norm1(kwargs["x"])
        hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
        hidden_out = self.node_norm2(hidden_out)
        hidden_out = kwargs["x"] + hidden_out

        return self.update((hidden_out), **update_kwargs)
