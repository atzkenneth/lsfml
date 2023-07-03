#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (©) 2022 Kenneth Atz (ETH Zurich), David F. Nippa (F. Hoffmann-La Roche Ltd) & Alex T. Müller (F. Hoffmann-La Roche Ltd)

import glob

import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.utils.undirected import to_undirected

from lsfml.qml.prod import get_model
from lsfml.utils import get_dict_for_embedding, QML_ATOMTYPES

QMLMODEL = get_model(gpu=False)

QML_ATOMTYPE_DICT = get_dict_for_embedding(QML_ATOMTYPES)


def compare_charges(sdf):
    """Main function to compare predicted with calculated partial charges using DFT using reference structures in test_mols/.

    :param sdf: Path to SDF of a molecule.
    :type sdf: str
    :return: Mean absolute error between the predicted and calculated partial charges.
    :rtype: numpy.float64
    """
    mol = next(Chem.SDMolSupplier(sdf, removeHs=False))

    # props
    props = mol.GetPropsAsDict()

    # read charges
    dft = props["DFT:MULLIKEN_CHARGES"].split("|")
    dft = [float(x) for x in dft]
    dft = np.array(dft)

    #  get atomids and xyz coords
    qml_atomids = []
    crds_3d = []

    for idx, i in enumerate(mol.GetAtoms()):
        qml_atomids.append(QML_ATOMTYPE_DICT[i.GetSymbol()])
        crds_3d.append(list(mol.GetConformer().GetAtomPosition(idx)))

    qml_atomids = np.array(qml_atomids)
    crds_3d = np.array(crds_3d)

    # 3D graph for qml prediction
    qml_atomids = torch.LongTensor(qml_atomids)
    xyzs = torch.FloatTensor(crds_3d)
    edge_index = np.array(nx.complete_graph(qml_atomids.size(0)).edges())
    edge_index = to_undirected(torch.from_numpy(edge_index).t().contiguous())
    edge_index, _ = add_self_loops(edge_index, num_nodes=crds_3d.shape[0])

    qml_graph = Data(
        atomids=qml_atomids,
        coords=xyzs,
        edge_index=edge_index,
        num_nodes=qml_atomids.size(0),
    )

    # prediction
    charges = QMLMODEL(qml_graph).detach().numpy()

    # mae calculation
    return np.mean(np.abs(charges - dft))


if __name__ == "__main__":
    mol_files = sorted(glob.glob("test_mols/*sdf"))

    maes = []

    print("Partail charge prediction is conducted with the following mean absolute errors:")

    for sdf in mol_files:
        mae = compare_charges(sdf)
        print(sdf.split("/")[-1][:-4], mae)
