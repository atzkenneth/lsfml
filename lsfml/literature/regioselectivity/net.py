#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (©) 2023 Kenneth Atz (ETH Zurich)

import torch
import torch.nn as nn

from lsfqml.lsfqml.publication.modules.gnn_blocks import (
    EGNN_sparse,
    EGNN_sparse3D,
    weights_init,
)


class Atomistic_EGNN(nn.Module):
    """Atomistic graph neural network (aGNN) for regioselectivity predictions. 
    """

    def __init__(self, n_kernels=3, mlp_dim=512, kernel_dim=64, embeddings_dim=64, qml=True, geometry=True):
        """Initialization of aGNN

        :param n_kernels: Number of message passing functions, defaults to 3
        :type n_kernels: int, optional
        :param mlp_dim: Feature dimension within the multi layer perceptrons, defaults to 512
        :type mlp_dim: int, optional
        :param kernel_dim: Feature dimension within the message passing fucntions, defaults to 64
        :type kernel_dim: int, optional
        :param embeddings_dim: Embedding dimension of the input features (e.g. reaction conditions and atomic features), defaults to 64
        :type embeddings_dim: int, optional
        :param qml: Option to include DFT-level partial charges, defaults to True
        :type qml: bool, optional
        :param geometry: Option to include steric information in the input graph, defaults to True
        :type geometry: bool, optional
        """
        super(Atomistic_EGNN, self).__init__()

        self.embeddings_dim = embeddings_dim
        self.m_dim = 16
        self.kernel_dim = kernel_dim
        self.n_kernels = n_kernels
        self.aggr = "add"
        self.pos_dim = 3
        self.mlp_dim = mlp_dim
        self.qml = qml
        self.geometry = geometry

        dropout = 0.1
        self.dropout = nn.Dropout(dropout)

        self.atom_em = nn.Embedding(num_embeddings=10, embedding_dim=self.embeddings_dim)
        self.ring_em = nn.Embedding(num_embeddings=2, embedding_dim=self.embeddings_dim)
        self.hybr_em = nn.Embedding(num_embeddings=4, embedding_dim=self.embeddings_dim)
        self.arom_em = nn.Embedding(num_embeddings=2, embedding_dim=self.embeddings_dim)

        if self.qml:
            self.chrg_em = nn.Linear(1, self.embeddings_dim)
            self.pre_egnn_mlp_input_dim = self.embeddings_dim * 5
            self.chrg_em.apply(weights_init)
        else:
            self.pre_egnn_mlp_input_dim = self.embeddings_dim * 4

        self.pre_egnn_mlp = nn.Sequential(
            nn.Linear(self.pre_egnn_mlp_input_dim, self.kernel_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.kernel_dim * 2, self.kernel_dim),
            nn.SiLU(),
            nn.Linear(self.kernel_dim, self.kernel_dim),
            nn.SiLU(),
        )

        self.kernels = nn.ModuleList()
        for _ in range(self.n_kernels):
            if self.geometry:
                self.kernels.append(
                    EGNN_sparse3D(
                        feats_dim=self.kernel_dim,
                        m_dim=self.m_dim,
                        aggr=self.aggr,
                    )
                )
            else:
                self.kernels.append(
                    EGNN_sparse(
                        feats_dim=self.kernel_dim,
                        m_dim=self.m_dim,
                        aggr=self.aggr,
                    )
                )

        self.post_egnn_mlp = nn.Sequential(
            nn.Linear(self.kernel_dim * self.n_kernels, self.mlp_dim),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_dim, 1),
            nn.Sigmoid(),
        )

        self.kernels.apply(weights_init)
        self.pre_egnn_mlp.apply(weights_init)
        self.post_egnn_mlp.apply(weights_init)
        nn.init.xavier_uniform_(self.atom_em.weight)
        nn.init.xavier_uniform_(self.ring_em.weight)
        nn.init.xavier_uniform_(self.hybr_em.weight)
        nn.init.xavier_uniform_(self.arom_em.weight)

    def forward(self, g_batch):
        """Forward pass of the atomistic GNN.

        :param g_batch: Input graph. 
        :type g_batch: class
        :return: Regioselectivity, 0 - 1 per atom. 
        :rtype: Tensor
        """
        if self.qml:
            features = self.pre_egnn_mlp(
                torch.cat(
                    [
                        self.atom_em(g_batch.atom_id),
                        self.ring_em(g_batch.ring_id),
                        self.hybr_em(g_batch.hybr_id),
                        self.arom_em(g_batch.arom_id),
                        self.chrg_em(g_batch.charges),
                    ],
                    dim=1,
                )
            )
        else:
            features = self.pre_egnn_mlp(
                torch.cat(
                    [
                        self.atom_em(g_batch.atom_id),
                        self.ring_em(g_batch.ring_id),
                        self.hybr_em(g_batch.hybr_id),
                        self.arom_em(g_batch.arom_id),
                    ],
                    dim=1,
                )
            )

        feature_list = []
        if self.geometry:
            features = torch.cat([g_batch.crds_3d, features], dim=1)
            for kernel in self.kernels:
                features = kernel(
                    x=features,
                    edge_index=g_batch.edge_index,
                )
                feature_list.append(features[:, self.pos_dim :])
        else:
            for kernel in self.kernels:
                features = kernel(x=features, edge_index=g_batch.edge_index)
                feature_list.append(features)

        features = torch.cat(feature_list, dim=1)
        features = self.post_egnn_mlp(features).squeeze(1)

        return features
