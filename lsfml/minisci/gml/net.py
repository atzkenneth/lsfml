import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch_scatter import scatter_sum
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import MessagePassing

# from torch_geometric.nn import GraphMultisetTransformer
# from torch_geometric.nn.aggr.gmt import GraphMultisetTransformer
from lsfml.minisci.gml.pygmt import GraphMultisetTransformer
from torch_geometric.typing import Adj, Size, Tensor


class GraphTransformer(nn.Module):
    def __init__(
        self,
        n_kernels=3,
        pooling_heads=8,
        mlp_dim=512,
        kernel_dim=128,
        embeddings_dim=128,
        geometry=True,
    ):
        super(GraphTransformer, self).__init__()

        self.embeddings_dim = embeddings_dim
        self.comnd_emb = 32
        self.m_dim = 16
        self.kernel_dim = kernel_dim
        self.n_kernels = n_kernels
        self.aggr = "add"
        self.pos_dim = 3
        self.pooling_heads = pooling_heads
        self.mlp_dim = mlp_dim
        self.out_dim = 1
        self.geometry = geometry

        # General embedding
        dropout = 0.1
        self.dropout = nn.Dropout(dropout)

        self.atom_em = nn.Embedding(num_embeddings=10, embedding_dim=self.embeddings_dim)
        self.ring_em = nn.Embedding(num_embeddings=2, embedding_dim=self.embeddings_dim)
        self.hybr_em = nn.Embedding(num_embeddings=4, embedding_dim=self.embeddings_dim)
        self.arom_em = nn.Embedding(num_embeddings=2, embedding_dim=self.embeddings_dim)

        # General pre egnn mlp
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

        # Molecule 1 - Kernel | post egnn mlp | Transformer
        self.kernels = nn.ModuleList()
        for _ in range(self.n_kernels):
            if self.geometry == True:
                self.kernels.append(
                    EGNN_sparse3D(
                        feats_dim=self.kernel_dim,
                        m_dim=self.m_dim,
                        aggr=self.aggr,
                    )
                )
            elif self.geometry == False:
                self.kernels.append(
                    EGNN_sparse(
                        feats_dim=self.kernel_dim,
                        m_dim=self.m_dim,
                        aggr=self.aggr,
                    )
                )

        post_egnn_mlp_inp_dim = self.kernel_dim * (self.n_kernels + 1)

        self.post_egnn_mlp = nn.Sequential(
            nn.Linear(post_egnn_mlp_inp_dim, self.kernel_dim),
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

        # Molecule 2 - Kernel | post egnn mlp | Transformer
        self.kernels2 = nn.ModuleList()
        for _ in range(self.n_kernels):
            if self.geometry == True:
                self.kernels2.append(
                    EGNN_sparse3D(
                        feats_dim=self.kernel_dim,
                        m_dim=self.m_dim,
                        aggr=self.aggr,
                    )
                )
            elif self.geometry == False:
                self.kernels2.append(
                    EGNN_sparse(
                        feats_dim=self.kernel_dim,
                        m_dim=self.m_dim,
                        aggr=self.aggr,
                    )
                )

        post_egnn_mlp2_inp_dim = self.kernel_dim * (self.n_kernels + 1)

        self.post_egnn_mlp2 = nn.Sequential(
            nn.Linear(post_egnn_mlp2_inp_dim, self.kernel_dim),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.kernel_dim, self.kernel_dim),
            nn.SiLU(),
            nn.Linear(self.kernel_dim, self.kernel_dim),
            nn.SiLU(),
        )

        self.transformer2 = GraphMultisetTransformer(
            in_channels=self.kernel_dim,
            hidden_channels=self.kernel_dim,
            out_channels=self.kernel_dim,
            pool_sequences=["GMPool_G", "SelfAtt", "GMPool_I"],
            num_heads=1,
            layer_norm=True,
        )

        # Concatenating conditions and finaling post pooling mlp
        self.con_lin = nn.Linear(10, self.comnd_emb * 2)

        self.rea_em = nn.Embedding(num_embeddings=1, embedding_dim=self.comnd_emb)
        self.so1_em = nn.Embedding(num_embeddings=2, embedding_dim=self.comnd_emb)
        self.so2_em = nn.Embedding(num_embeddings=1, embedding_dim=self.comnd_emb)
        self.cat_em = nn.Embedding(num_embeddings=7, embedding_dim=self.comnd_emb)
        self.add_em = nn.Embedding(num_embeddings=4, embedding_dim=self.comnd_emb)
        self.atm_em = nn.Embedding(num_embeddings=2, embedding_dim=self.comnd_emb)

        self.post_pooling_mlp_input_dim = self.kernel_dim * (self.pooling_heads + 1) + self.comnd_emb * 8

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

        # Initializing weights
        self.transformers.apply(weights_init)
        self.transformer2.apply(weights_init)
        self.kernels.apply(weights_init)
        self.kernels2.apply(weights_init)
        self.pre_egnn_mlp.apply(weights_init)
        self.post_egnn_mlp.apply(weights_init)
        self.post_egnn_mlp2.apply(weights_init)
        self.post_pooling_mlp.apply(weights_init)
        self.con_lin.apply(weights_init)
        nn.init.xavier_uniform_(self.atom_em.weight)
        nn.init.xavier_uniform_(self.ring_em.weight)
        nn.init.xavier_uniform_(self.hybr_em.weight)
        nn.init.xavier_uniform_(self.arom_em.weight)

    def forward(self, g_batch, g_batch2):
        # General embedding
        features1 = self.pre_egnn_mlp(
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

        features2 = self.pre_egnn_mlp(
            torch.cat(
                [
                    self.atom_em(g_batch2.atom_id),
                    self.ring_em(g_batch2.ring_id),
                    self.hybr_em(g_batch2.hybr_id),
                    self.arom_em(g_batch2.arom_id),
                ],
                dim=1,
            )
        )

        # Molecule 1 - Kernel | post egnn mlp | Transformer
        feature_list = []

        feature_list.append(features1)

        if self.geometry == True:
            features = torch.cat([g_batch.crds_3d, features1], dim=1)
            for kernel in self.kernels:
                features = kernel(
                    x=features,
                    edge_index=g_batch.edge_index,
                )
                feature_list.append(features[:, self.pos_dim :])
        elif self.geometry == False:
            features = features1
            for kernel in self.kernels:
                features = kernel(x=features, edge_index=g_batch.edge_index)
                feature_list.append(features)

        features = torch.cat(feature_list, dim=1)
        features = self.post_egnn_mlp(features)

        feature_list = []
        for transformer in self.transformers:
            feature_list.append(transformer(x=features, batch=g_batch.batch, edge_index=g_batch.edge_index))

        glob_features_1 = torch.cat(feature_list, dim=1)
        del feature_list

        # Molecule 2 - Kernel | post egnn mlp | Transformer
        feature_list2 = []

        feature_list2.append(features2)

        if self.geometry == True:
            features = torch.cat([g_batch2.crds_3d, features2], dim=1)
            for kernel in self.kernels2:
                features = kernel(
                    x=features,
                    edge_index=g_batch2.edge_index,
                )
                feature_list2.append(features[:, self.pos_dim :])
        elif self.geometry == False:
            features = features2
            for kernel in self.kernels2:
                features = kernel(x=features, edge_index=g_batch2.edge_index)
                feature_list2.append(features)

        features = torch.cat(feature_list2, dim=1)
        features = self.post_egnn_mlp2(features)

        glob_features_2 = self.transformer2(x=features, batch=g_batch2.batch, edge_index=g_batch2.edge_index)

        # Concatenating conditions and finaling post pooling mlp
        conditions = self.con_lin(g_batch.condtns)
        rea_em = self.rea_em(g_batch.rea_id.squeeze(1))
        so1_em = self.so1_em(g_batch.so1_id.squeeze(1))
        so2_em = self.so2_em(g_batch.so2_id.squeeze(1))
        cat_em = self.cat_em(g_batch.cat_id.squeeze(1))
        add_em = self.add_em(g_batch.add_id.squeeze(1))
        atm_em = self.atm_em(g_batch.atm_id.squeeze(1))

        features = torch.cat(
            [glob_features_1, glob_features_2, conditions, rea_em, so1_em, so2_em, cat_em, add_em, atm_em], dim=1
        )
        # features = torch.cat([glob_features_1, glob_features_2, rea_em, so1_em, so2_em, cat_em, add_em, atm_em], dim=1)
        # print(features.size(), glob_features_1.size(), glob_features_2.size(), conditions.size())
        features = self.post_pooling_mlp(features).squeeze(1)

        return features


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class EGNN_sparse(MessagePassing):
    def __init__(self, feats_dim, m_dim=32, dropout=0.1, aggr="add", **kwargs):
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
        hidden_out = self.propagate(edge_index, x=x)

        return hidden_out

    def message(self, x_i, x_j):
        m_ij = self.edge_mlp(torch.cat([x_i, x_j], dim=-1))
        return m_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
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
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x


class EGNN_sparse3D(MessagePassing):
    def __init__(self, feats_dim, pos_dim=3, m_dim=32, dropout=0.1, fourier_features=16, aggr="add", **kwargs):
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
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return m_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
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


if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    from net_utils import DataLSF, get_rxn_ids

    # data
    tran_ids, eval_ids, test_ids = get_rxn_ids()
    train_data = DataLSF(rxn_ids=tran_ids)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)

    # model
    model = GraphTransformer(
        n_kernels=3,
        pooling_heads=4,
        mlp_dim=512,
        kernel_dim=128,
        embeddings_dim=128,
        geometry=True,
    )

    print(
        sum(p.numel() for p in model.parameters()),
    )

    # model.train()

    for g, g2 in train_loader:
        p = model(g, g2)
        print(g.edge_index.size())
