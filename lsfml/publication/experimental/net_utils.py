#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (©) 2022 Kenneth Atz (ETH Zurich), David F. Nippa (F. Hoffmann-La Roche Ltd) & Alex T. Müller (F. Hoffmann-La Roche Ltd)

import random

import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset

random.seed(2)


def get_rxn_ids(
    data="../data/experimental_rxndata.h5",
    split="random",
    eln="ELN036496-147",
    testset="1",
):
    """Generates the data set split into training, validation and test sets. 

    :param data: Path to h5 file, including preprocessed data, defaults to "../data/experimental_rxndata.h5"
    :type data: str, optional
    :param split: Type of split (eln or random), defaults to "random"
    :type split: str, optional
    :param eln: Substrate number for substrate-based split, defaults to "ELN036496-147"
    :type eln: str, optional
    :param testset: Type of testset, defaults to "1"
    :type testset: str, optional
    :return: Reaction IDs for training, validation and test split
    :rtype: list[str]
    """

    # Load data from h5 file
    h5f = h5py.File(data)

    # Load all rxn keys
    rxn_ids = list(h5f.keys())
    random.shuffle(rxn_ids)

    # Define subset of rxn keys
    if split == "random":
        if testset == "1":
            tran_ids = rxn_ids[: int(len(rxn_ids) / 2)]
            eval_ids = rxn_ids[int(len(rxn_ids) / 4) * 3 :]
            test_ids = rxn_ids[int(len(rxn_ids) / 2) : int(len(rxn_ids) / 4) * 3]
        if testset == "2":
            tran_ids = rxn_ids[: int(len(rxn_ids) / 2)]
            eval_ids = rxn_ids[int(len(rxn_ids) / 2) : int(len(rxn_ids) / 4) * 3]
            test_ids = rxn_ids[int(len(rxn_ids) / 4) * 3 :]
        if testset == "3":
            tran_ids = rxn_ids[int(len(rxn_ids) / 2) :]
            eval_ids = rxn_ids[: int(len(rxn_ids) / 4)]
            test_ids = rxn_ids[int(len(rxn_ids) / 4) : int(len(rxn_ids) / 2)]
        if testset == "4":
            tran_ids = rxn_ids[int(len(rxn_ids) / 2) :]
            eval_ids = rxn_ids[int(len(rxn_ids) / 4) : int(len(rxn_ids) / 2)]
            test_ids = rxn_ids[: int(len(rxn_ids) / 4)]
    elif split == "eln":
        rxn_ids_train = [x for x in rxn_ids if eln not in x]
        tran_ids = rxn_ids_train[int(len(rxn_ids_train) / 3) :]
        eval_ids = rxn_ids_train[: int(len(rxn_ids_train) / 3)]
        test_ids = [x for x in rxn_ids if eln in x]

    return tran_ids, eval_ids, test_ids


class DataLSF(Dataset):
    """Generates the desired graph objects (2D, 3D, QM) from reading the h5 files. 
    """

    def __init__(
        self,
        rxn_ids,
        data="../data/experimental_rxndata.h5",
        data_substrates="../data/experimental_substrates.h5",
        target="binary",  
        graph_dim="edge_2d", 
        fingerprint="ecfp4_2", 
        conformers=["a", "b", "c", "d", "e"],
    ):
        """Initialization.

        :param rxn_ids: Reactions IDs from the given split (train, eval, test)
        :type rxn_ids: list[str]
        :param data: Path to h5 file, including preprocessed data, defaults to "../data/experimental_rxndata.h5"
        :type data: str, optional
        :param data: Path to h5 file, including preprocessed data, defaults to "../data/experimental_substrates.h5"
        :type data_substrates: str, optional
        :param target: Target type (binary or mono), defaults to "binary"
        :type target: str, optional
        :param graph_dim: Indicating 2D or 3D graph structure ("edge_2d" or "edge_3d"), defaults to "edge_2d"
        :type target: str, optional
        :param fingerprint: Indicating fingerprint type (ecfp4_2 or None), defaults to "ecfp4_2"
        :type target: str, optional
        :param conformers: List of conformers keys, defaults to ["a", "b", "c", "d", "e"]
        :type target: list[str], optional
        """

        # Define inputs
        self.target = target
        self.graph_dim = graph_dim
        self.fingerprint = fingerprint
        self.conformers = conformers
        self.rxn_ids = rxn_ids

        # Load data from h5 file
        self.h5f = h5py.File(data)
        self.h5f_subs = h5py.File(data_substrates)

        # Generate dict (int to rxn keys)
        nums = list(range(0, len(self.rxn_ids)))
        self.idx2rxn = {}
        for x in range(len(self.rxn_ids)):
            self.idx2rxn[nums[x]] = self.rxn_ids[x]

        print("\nLoader initialized:")
        print(f"Number of reactions loaded: {len(self.rxn_ids)}")
        print(f"Chosen target (binary or mono): {self.target}")
        print(f"Chosen graph_dim (edge_2d of edge_3d): {self.graph_dim}")
        print(f"Chosen fingerprint (ecfp4_2 of ecfp6_1): {self.fingerprint}")

    def __getitem__(self, idx):
        """Loop over data.

        :param idx: Reaction ID
        :type idx: str
        :return: Input graph for the neural network. 
        :rtype: torch_geometric.loader.dataloader.DataLoader
        """

        # int to rxn_id
        rxn_id = self.idx2rxn[idx]

        sbst_rxn = rxn_id.split("_")[0]
        sbst_rxn = sbst_rxn.split("-")[-1]

        # Pick random conformer
        conformer = random.choice(self.conformers)

        # Molecule
        atom_id = np.array(self.h5f_subs[sbst_rxn][f"atom_id_{conformer}"])
        ring_id = np.array(self.h5f_subs[sbst_rxn][f"ring_id_{conformer}"])
        hybr_id = np.array(self.h5f_subs[sbst_rxn][f"hybr_id_{conformer}"])
        arom_id = np.array(self.h5f_subs[sbst_rxn][f"arom_id_{conformer}"])
        charges = np.array(self.h5f_subs[sbst_rxn][f"charges_{conformer}"])
        crds_3d = np.array(self.h5f_subs[sbst_rxn][f"crds_3d_{conformer}"])

        if self.fingerprint is not None:
            ecfp_fp = np.array(self.h5f[rxn_id][self.fingerprint])
        else:
            ecfp_fp = np.array([])

        # Edge IDs with desired dimension
        edge_index = np.array(self.h5f_subs[sbst_rxn][f"{self.graph_dim}_{conformer}"])

        # Conditions
        lgnd_id = np.array(self.h5f[rxn_id]["lgnd_id"])
        slvn_id = np.array(self.h5f[rxn_id]["slvn_id"])

        # Tragets
        if self.target == "binary":
            rxn_trg = np.array(self.h5f[rxn_id]["mono_id"])
        elif self.target == "mono":
            mo_frct = np.array(self.h5f[rxn_id]["mo_frct"])
            di_frct = np.array(self.h5f[rxn_id]["di_frct"])
            rxn_trg = np.array(float(mo_frct) + float(di_frct))

        num_nodes = torch.LongTensor(atom_id).size(0)

        graph_data = Data(
            atom_id=torch.LongTensor(atom_id),
            ring_id=torch.LongTensor(ring_id),
            hybr_id=torch.LongTensor(hybr_id),
            arom_id=torch.LongTensor(arom_id),
            lgnd_id=torch.LongTensor(lgnd_id),
            slvn_id=torch.LongTensor(slvn_id),
            charges=torch.FloatTensor(charges),
            ecfp_fp=torch.FloatTensor(ecfp_fp),
            crds_3d=torch.FloatTensor(crds_3d),
            rxn_trg=torch.FloatTensor(rxn_trg),
            edge_index=torch.LongTensor(edge_index),
            num_nodes=num_nodes,
        )

        return graph_data

    def __len__(self):
        """Return length

        :return: length
        :rtype: int
        """
        return len(self.rxn_ids)
