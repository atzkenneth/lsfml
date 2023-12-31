#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (©) 2023, ETH Zurich

import random

import h5py
import numpy as np
import torch
from torch_geometric.data import Data
from lsfml.modules.pygdataset import Dataset

random.seed(2)


def get_rxn_ids(
    data,
):
    """Generates the data set split into training, validation and test sets.

    :param data: Path to h5 file, including preprocessed data, defaults to "../../data/literature_regio.h5"
    :type data: str, optional
    :return: Reaction IDs for training, validation and test split
    :rtype: list[str]
    """
    # Load data from h5 file
    h5f = h5py.File(data)

    # Load all rxn keys
    rxn_ids = list(h5f.keys())
    random.shuffle(rxn_ids)

    # Define subset of rxn keys
    tran_ids = rxn_ids[: int(len(rxn_ids) / 2)]
    eval_ids = rxn_ids[int(len(rxn_ids) / 4) * 3 :]
    test_ids = rxn_ids[int(len(rxn_ids) / 2) : int(len(rxn_ids) / 4) * 3]

    return tran_ids, eval_ids, test_ids


class DataLSF(Dataset):
    """Generates the desired graph objects (2D, 3D, QM) from reading the h5 files."""

    def __init__(
        self,
        rxn_ids,
        data,
        graph_dim,
    ):
        """Initialization.

        :param rxn_ids: Reaction IDs from the given split (train, eval, test)
        :type rxn_ids: list[str]
        :param data: Path to h5 file, including preprocessed data, defaults to "../../data/literature_regio.h5"
        :type data: str, optional
        :param graph_dim: Indicating 2D or 3D graph structure ("edge_2d" or "edge_3d"), defaults to "edge_2d"
        :type graph_dim: str, optional
        """
        # Define inputs
        self.graph_dim = graph_dim
        self.rxn_ids = rxn_ids

        # Load data from h5 file
        self.h5f = h5py.File(data)

        # Generate dict (int to rxn keys)
        nums = list(range(0, len(self.rxn_ids)))
        self.idx2rxn = {}
        for x in range(len(self.rxn_ids)):
            self.idx2rxn[nums[x]] = self.rxn_ids[x]

        print("\nLoader initialized:")
        print(f"Number of reactions loaded: {len(self.rxn_ids)}")
        print(f"Chosen graph_dim (edge_2d of edge_3d): {self.graph_dim}")

    def __getitem__(self, idx):
        """Loop over data.

        :param idx: Reaction ID
        :type idx: str
        :return: Input graph for the neural network.
        :rtype: torch_geometric.loader.dataloader.DataLoader
        """
        # int to rxn_id
        rxn_id = self.idx2rxn[idx]

        # Molecule
        atom_id = np.array(self.h5f[str(rxn_id)]["atom_id"])
        ring_id = np.array(self.h5f[str(rxn_id)]["ring_id"])
        hybr_id = np.array(self.h5f[str(rxn_id)]["hybr_id"])
        arom_id = np.array(self.h5f[str(rxn_id)]["arom_id"])
        charges = np.array(self.h5f[str(rxn_id)]["charges"])
        crds_3d = np.array(self.h5f[str(rxn_id)]["crds_3d"])
        pot_trg = np.array(self.h5f[str(rxn_id)]["pot_trg"])
        # print(idx, rxn_id, atom_id)

        # Edge IDs with desired dimension
        edge_index = np.array(self.h5f[str(rxn_id)][self.graph_dim])

        # Tragets
        rxn_trg = np.array(self.h5f[str(rxn_id)]["reg_trg"])

        num_nodes = torch.LongTensor(atom_id).size(0)

        graph_data = Data(
            atom_id=torch.LongTensor(atom_id),
            pot_trg=torch.LongTensor(pot_trg),
            ring_id=torch.LongTensor(ring_id),
            hybr_id=torch.LongTensor(hybr_id),
            arom_id=torch.LongTensor(arom_id),
            charges=torch.FloatTensor(charges),
            crds_3d=torch.FloatTensor(crds_3d),
            rxn_trg=torch.FloatTensor(rxn_trg),
            edge_index=torch.LongTensor(edge_index),
            num_nodes=num_nodes,
            rxn_id=rxn_id,
        )

        return graph_data

    def __len__(self):
        """Get length

        :return: length
        :rtype: int
        """
        return len(self.rxn_ids)
