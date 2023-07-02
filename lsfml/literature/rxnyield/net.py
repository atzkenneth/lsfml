#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (©) 2023 Kenneth Atz (ETH Zurich)

import torch
import torch.nn as nn

from lsfml.modules.gmt import GraphMultisetTransformer
from lsfml.modules.gnn_blocks import (
    EGNN_sparse,
    EGNN_sparse3D,
    weights_init,
    scatter_sum,
)


class GraphTransformer(nn.Module):
    """Graph Transformer neural network (GTNN) for yield and binary predictions. 
    """
    
    def __init__(
        self,
        n_kernels=3,
        pooling_heads=8,
        mlp_dim=512,
        kernel_dim=64,
        embeddings_dim=64,
        qml=True,
        geometry=True,
    ):
        
        """Initialization of GTNN

        :param n_kernels: Number of message passing functions, defaults to 3
        :type n_kernels: int, optional
        :param pooling_heads: Number of Transformers, defaults to 8
        :type pooling_heads: int, optional
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
        super(GraphTransformer, self).__init__()

        self.embeddings_dim = embeddings_dim
        self.m_dim = 16
        self.kernel_dim = kernel_dim
        self.n_kernels = n_kernels
        self.aggr = "add"
        self.pos_dim = 3
        self.pooling_heads = pooling_heads
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
            nn.Linear(self.kernel_dim * self.n_kernels, self.kernel_dim),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.kernel_dim, self.kernel_dim),
            nn.SiLU(),
            nn.Linear(self.kernel_dim, self.kernel_dim),
            nn.SiLU(),
        )

        self.transformers = nn.ModuleList()
        for _ in range(self.pooling_heads):
            self.transformers.append(
                GraphMultisetTransformer(
                    in_channels=self.kernel_dim,
                    hidden_channels=self.kernel_dim,
                    out_channels=self.kernel_dim,
                    pool_sequences=["GMPool_G", "SelfAtt", "GMPool_I"],
                    num_heads=1,
                    layer_norm=True,
                )
            )

        self.lig_emb = nn.Embedding(num_embeddings=12, embedding_dim=self.embeddings_dim)
        self.sol_emb = nn.Embedding(num_embeddings=9, embedding_dim=self.embeddings_dim)
        self.rgn_emb = nn.Embedding(num_embeddings=2, embedding_dim=self.embeddings_dim)
        self.cat_emb = nn.Embedding(num_embeddings=4, embedding_dim=self.embeddings_dim)

        self.post_pooling_mlp_input_dim = self.kernel_dim * (self.pooling_heads) + self.embeddings_dim * 4

        self.post_pooling_mlp = nn.Sequential(
            nn.Linear(self.post_pooling_mlp_input_dim, self.mlp_dim),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_dim, 1),
        )

        self.transformers.apply(weights_init)
        self.kernels.apply(weights_init)
        self.pre_egnn_mlp.apply(weights_init)
        self.post_egnn_mlp.apply(weights_init)
        self.post_pooling_mlp.apply(weights_init)
        nn.init.xavier_uniform_(self.atom_em.weight)
        nn.init.xavier_uniform_(self.ring_em.weight)
        nn.init.xavier_uniform_(self.hybr_em.weight)
        nn.init.xavier_uniform_(self.arom_em.weight)
        nn.init.xavier_uniform_(self.lig_emb.weight)
        nn.init.xavier_uniform_(self.sol_emb.weight)
        nn.init.xavier_uniform_(self.rgn_emb.weight)
        nn.init.xavier_uniform_(self.cat_emb.weight)

    def forward(self, g_batch):
        """Forward pass of the GTNN.

        :param g_batch: Input graph. 
        :type g_batch: class
        :return: Prediction. 
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
        features = self.post_egnn_mlp(features)

        feature_list = []
        for transformer in self.transformers:
            feature_list.append(transformer(x=features, batch=g_batch.batch, edge_index=g_batch.edge_index))

        features = torch.cat(feature_list, dim=1)
        del feature_list

        conditions = torch.cat(
            [
                self.lig_emb(g_batch.lgnd_id),
                self.sol_emb(g_batch.slvn_id),
                self.rgn_emb(g_batch.rgnt_id),
                self.cat_emb(g_batch.clst_id),
            ],
            dim=1,
        )

        features = torch.cat([features, conditions], dim=1)
        features = self.post_pooling_mlp(features).squeeze(1)

        return features


class EGNN(nn.Module):
    """Graph neural network (GNN) using sum pooling for yield and binary predictions. 
    """

    def __init__(self, n_kernels=3, mlp_dim=512, kernel_dim=64, embeddings_dim=64, qml=True, geometry=True):
        """Initialization of GNN

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
        super(EGNN, self).__init__()

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
            nn.Linear(self.kernel_dim * self.n_kernels, self.kernel_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.kernel_dim * 2, self.kernel_dim * 2),
            nn.SiLU(),
            nn.Linear(self.kernel_dim * 2, self.kernel_dim * 2),
            nn.SiLU(),
        )

        self.lig_emb = nn.Embedding(num_embeddings=12, embedding_dim=self.embeddings_dim)
        self.sol_emb = nn.Embedding(num_embeddings=9, embedding_dim=self.embeddings_dim)
        self.rgn_emb = nn.Embedding(num_embeddings=2, embedding_dim=self.embeddings_dim)
        self.cat_emb = nn.Embedding(num_embeddings=4, embedding_dim=self.embeddings_dim)

        self.post_pooling_mlp_input_dim = self.kernel_dim * 2 + self.embeddings_dim * 4

        self.post_pooling_mlp = nn.Sequential(
            nn.Linear(self.post_pooling_mlp_input_dim, self.mlp_dim),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_dim, 1),
        )

        self.kernels.apply(weights_init)
        self.pre_egnn_mlp.apply(weights_init)
        self.post_egnn_mlp.apply(weights_init)
        self.post_pooling_mlp.apply(weights_init)
        nn.init.xavier_uniform_(self.atom_em.weight)
        nn.init.xavier_uniform_(self.ring_em.weight)
        nn.init.xavier_uniform_(self.hybr_em.weight)
        nn.init.xavier_uniform_(self.arom_em.weight)
        nn.init.xavier_uniform_(self.lig_emb.weight)
        nn.init.xavier_uniform_(self.sol_emb.weight)
        nn.init.xavier_uniform_(self.rgn_emb.weight)
        nn.init.xavier_uniform_(self.cat_emb.weight)

    def forward(self, g_batch):
        """Forward pass of the GNN.

        :param g_batch: Input graph. 
        :type g_batch: class
        :return: Prediction. 
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
        features = self.post_egnn_mlp(features)

        del feature_list

        features = scatter_sum(features, g_batch.batch, dim=0)

        conditions = torch.cat(
            [
                self.lig_emb(g_batch.lgnd_id),
                self.sol_emb(g_batch.slvn_id),
                self.rgn_emb(g_batch.rgnt_id),
                self.cat_emb(g_batch.clst_id),
            ],
            dim=1,
        )

        features = torch.cat([features, conditions], dim=1)
        features = self.post_pooling_mlp(features).squeeze(1)

        return features


class FNN(nn.Module):
    """Feed forward neural network (FNN) for yield and binary predictions. 
    """

    def __init__(self, fp_dim=256, mlp_dim=512, kernel_dim=64, embeddings_dim=64):
        """Initialization of FNN

        :param fp_dim: Input dimension of the ECFP descriptor, defaults to 256
        :type fp_dim: int, optional
        :param mlp_dim: Feature dimension within the multi layer perceptrons, defaults to 512
        :type mlp_dim: int, optional
        :param kernel_dim: Feature dimension within the message passing fucntions, defaults to 64
        :type kernel_dim: int, optional
        :param embeddings_dim: Embedding dimension of the input features (e.g. reaction conditions), defaults to 64
        :type embeddings_dim: int, optional
        """
        super(FNN, self).__init__()

        self.embeddings_dim = embeddings_dim
        self.kernel_dim = kernel_dim
        self.mlp_dim = mlp_dim
        self.fp_dim = fp_dim

        dropout = 0.1
        self.dropout = nn.Dropout(dropout)

        self.pre_egnn_mlp = nn.Sequential(
            nn.Linear(self.fp_dim, self.kernel_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.kernel_dim * 2, self.kernel_dim * 2),
            nn.SiLU(),
            nn.Linear(self.kernel_dim * 2, self.kernel_dim),
        )

        self.lig_emb = nn.Embedding(num_embeddings=12, embedding_dim=self.embeddings_dim)
        self.sol_emb = nn.Embedding(num_embeddings=9, embedding_dim=self.embeddings_dim)
        self.rgn_emb = nn.Embedding(num_embeddings=2, embedding_dim=self.embeddings_dim)
        self.cat_emb = nn.Embedding(num_embeddings=4, embedding_dim=self.embeddings_dim)

        self.post_pooling_mlp_input_dim = self.kernel_dim + self.embeddings_dim * 4

        self.post_pooling_mlp = nn.Sequential(
            nn.Linear(self.post_pooling_mlp_input_dim, self.mlp_dim),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_dim, self.mlp_dim),
            nn.SiLU(),
            nn.Linear(self.mlp_dim, 1),
        )

        self.post_pooling_mlp.apply(weights_init)
        nn.init.xavier_uniform_(self.lig_emb.weight)
        nn.init.xavier_uniform_(self.sol_emb.weight)
        nn.init.xavier_uniform_(self.rgn_emb.weight)
        nn.init.xavier_uniform_(self.cat_emb.weight)

    def forward(self, g_batch):
        """Forward pass of the FNN.

        :param g_batch: Input graph. 
        :type g_batch: class
        :return: Prediction. 
        :rtype: Tensor
        """
        features = self.pre_egnn_mlp(g_batch.ecfp_fp)

        conditions = torch.cat(
            [
                self.lig_emb(g_batch.lgnd_id),
                self.sol_emb(g_batch.slvn_id),
                self.rgn_emb(g_batch.rgnt_id),
                self.cat_emb(g_batch.clst_id),
            ],
            dim=1,
        )

        features = torch.cat([features, conditions], dim=1)
        features = self.post_pooling_mlp(features).squeeze(1)

        return features
