#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (©) 2023, ETH Zurich

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, Size, Tensor


class DeltaNetAtomic(nn.Module):
    """Atomistic graph neural network (aGNN) for partial charge predictions."""

    def __init__(
        self,
        embedding_dim=128,
        n_kernels=2,
        n_mlp=3,
        mlp_dim=256,
        n_outputs=1,
        m_dim=64,
        initialize_weights=True,
        fourier_features=4,
        aggr="mean",
    ):
        """Initialization of aGNN.

        :param embeddings_dim: Embedding dimension of the input features (e.g. reaction conditions and atomic features), defaults to 64
        :type embeddings_dim: int, optional
        :param n_kernels: Number of message passing functions, defaults to 3
        :type n_kernels: int, optional
        :param n_mlp: Number of multi layer perceptrons, defaults to 3
        :type n_mlp: int, optional
        :param mlp_dim: Feature dimension of multi layer perceptrons, defaults to 256
        :type mlp_dim: int, optional
        :param out_dim: Output dimesion, defaults to 1
        :type out_dim: int, optional
        :param m_dim: Meessage passing feature dimesnion, defaults to 64
        :type m_dim: int, optional
        :param initialize_weights: initialize weights before training, defaults to True
        :type initialize_weights: bool, optional
        :param fourier_features: Number of Fourier features, defaults to 16
        :type fourier_features: int, optional
        :param aggr: Message aggregation type, defaults to "mean"
        :type aggr: str, optional
        """
        super(DeltaNetAtomic, self).__init__()

        self.pos_dim = 3
        self.m_dim = m_dim
        self.embedding_dim = embedding_dim
        self.n_kernels = n_kernels
        self.n_mlp = n_mlp
        self.mlp_dim = mlp_dim
        self.n_outputs = n_outputs
        self.initialize_weights = initialize_weights
        self.fourier_features = fourier_features
        self.aggr = aggr

        # Embedding
        self.embedding = nn.Embedding(num_embeddings=11, embedding_dim=self.embedding_dim)

        # Kernel
        self.kernel_dim = self.embedding_dim
        self.kernels = nn.ModuleList()
        for _ in range(self.n_kernels):
            self.kernels.append(
                EGNN_sparse(
                    feats_dim=self.kernel_dim,
                    pos_dim=self.pos_dim,
                    m_dim=self.m_dim,
                    fourier_features=self.fourier_features,
                    aggr=self.aggr,
                )
            )

        # MLP
        self.fnn = nn.ModuleList()
        input_fnn = self.kernel_dim * (self.n_kernels + 1)
        self.fnn.append(nn.Linear(input_fnn, mlp_dim))
        for _ in range(self.n_mlp - 1):
            self.fnn.append(nn.Linear(self.mlp_dim, self.mlp_dim))
        self.fnn.append(nn.Linear(self.mlp_dim, self.n_outputs))

        # Initialize weights
        if self.initialize_weights:
            self.kernels.apply(weights_init)
            self.fnn.apply(weights_init)
            nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, g_batch):
        """_summary_

        :param g_batch: _description_
        :type g_batch: _type_
        :return: _description_
        :rtype: _type_
        """
        # Embedding
        features = self.embedding(g_batch.atomids)
        features = torch.cat([g_batch.coords, features], dim=1)

        # Kernel
        feature_list = []
        feature_list.append(features[:, self.pos_dim :])

        for kernel in self.kernels:
            features = kernel(
                x=features,
                edge_index=g_batch.edge_index,
            )
            feature_list.append(features[:, self.pos_dim :])

        # Concat
        features = F.silu(torch.cat(feature_list, dim=1))

        # MLP 1
        for mlp in self.fnn[:-1]:
            features = F.silu(mlp(features))

        # Outputlayer
        features = self.fnn[-1](features).squeeze(1)

        return features


def weights_init(m):
    """Xavier uniform weight initialization.

    :param m: A list of learnable linear PyTorch modules.
    :type m: [torch.nn.modules.linear.Linear]
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


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


class EGNN_sparse(MessagePassing):
    """torch geometric message-passing layer for 3D molecular graphs."""

    def __init__(
        self, feats_dim, pos_dim=3, edge_attr_dim=0, m_dim=32, dropout=0.1, fourier_features=32, aggr="mean", **kwargs
    ):
        """Initialization of the 3D message passing layer.

        :param feats_dim: Node feature dimension.
        :type feats_dim: int
        :param pos_dim: Dimension of the graph, defaults to 3
        :type pos_dim: int, optional
        :param edge_attr_dim: Additional edge features (neglected in this implementation)
        :type edge_attr_dim: Tensor
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
        super(EGNN_sparse, self).__init__(**kwargs)

        # Model parameters
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.fourier_features = fourier_features

        self.edge_input_dim = (self.fourier_features * 2) + edge_attr_dim + 1 + (feats_dim * 2)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Edge layers
        self.edge_norm1 = nn.LayerNorm(m_dim)
        self.edge_norm2 = nn.LayerNorm(m_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            nn.SiLU(),
        )

        # Node layers
        self.node_norm1 = nn.LayerNorm(feats_dim)
        self.node_norm2 = nn.LayerNorm(feats_dim)

        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        )

        # Initialization
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
            rel_dist = rearrange(rel_dist, "n () d -> n d")

        hidden_out = self.propagate(
            edge_index,
            x=feats,
            edge_attr=rel_dist,
            coors=coors,
            rel_coors=rel_coors,
        )

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
        m_i = self.edge_norm1(m_i)

        # get updated node features
        hidden_feats = self.node_norm1(kwargs["x"])
        hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
        hidden_out = self.node_norm2(hidden_out)
        hidden_out = kwargs["x"] + hidden_out

        return self.update(hidden_out, **update_kwargs)
