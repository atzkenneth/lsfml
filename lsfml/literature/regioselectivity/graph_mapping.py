#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (©) 2023 Kenneth Atz (ETH Zurich)

import h5py, os
import networkx as nx
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.utils.undirected import to_undirected
from tqdm import tqdm

from lsfml.qml.prod import get_model
from lsfml.utils import get_dict_for_embedding, AROMATOCITY, IS_RING, QML_ATOMTYPES, UTILS_PATH

QMLMODEL = get_model(gpu=False)

HYBRIDISATION_DICT = {"SP3": 0, "SP2": 1, "SP": 2, "UNSPECIFIED": 3, "S": 3}
AROMATOCITY_DICT = get_dict_for_embedding(AROMATOCITY)
IS_RING_DICT = get_dict_for_embedding(IS_RING)
QML_ATOMTYPE_DICT = get_dict_for_embedding(QML_ATOMTYPES)
ATOMTYPE_DICT = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4, "P": 5, "S": 6, "Cl": 7, "Br": 8, "I": 9}


def get_rms(smi, patt, repl):
    """Takes a SMILES-sting and replaces a substructure (patt) such as a functional group with another substructure (repl) 

    :param smi: SMILES-string.
    :type smi: str
    :param patt: Pattern to be replaced.
    :type patt: str
    :param repl: Pattern to replace. 
    :type repl: str
    :return: SMILES-string.
    :rtype: str
    """
    m = Chem.MolFromSmiles(smi)
    rms = Chem.MolToSmiles(AllChem.ReplaceSubstructs(m, patt, repl)[0])

    m = Chem.MolFromSmiles(rms)
    while rms != Chem.MolToSmiles(AllChem.ReplaceSubstructs(m, patt, repl)[0]):
        print(
            f"More than one Boron: {rms == Chem.MolToSmiles(AllChem.ReplaceSubstructs(m, patt, repl)[0])}"
            f", {rms}, {Chem.MolToSmiles(AllChem.ReplaceSubstructs(m, patt, repl)[0])}"
        )
        rms = Chem.MolToSmiles(AllChem.ReplaceSubstructs(m, patt, repl)[0])
        m = Chem.MolFromSmiles(rms)

    return rms


def get_regioselectivity(rms, seed):
    """Main function to generate the 2D and 3D molecular graphs and to extract the reagioselectivity given a SMILES-string and a seed for 3D conformer generation.

    :param rms: SMILES-string
    :type rms: str
    :param seed: Random seed for 3D conformer generation
    :type seed: int
    :return: tuple including all graph-relevant numpy arrays
    :rtype: tuple
    """
    mol_no_Hs = Chem.MolFromSmiles(rms)
    mol = Chem.rdmolops.AddHs(mol_no_Hs)

    atomids = []
    qml_atomids = []
    is_ring = []
    hyb = []
    arom = []
    crds_3d = []
    pot_trg = []

    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.UFFOptimizeMolecule(mol)

    for idx, i in enumerate(mol.GetAtoms()):
        atomids.append(ATOMTYPE_DICT[i.GetSymbol()])
        qml_atomids.append(QML_ATOMTYPE_DICT[i.GetSymbol()])
        is_ring.append(IS_RING_DICT[str(i.IsInRing())])
        hyb.append(HYBRIDISATION_DICT[str(i.GetHybridization())])
        arom.append(AROMATOCITY_DICT[str(i.GetIsAromatic())])
        crds_3d.append(list(mol.GetConformer().GetAtomPosition(idx)))

        nghbrs = [x.GetSymbol() for x in i.GetNeighbors()]
        if (i.GetSymbol() == "C") and ("H" in nghbrs):
            pot_trg.append(1)
        else:
            pot_trg.append(0)

    trg_atoms = []
    edge_dir1 = []
    edge_dir2 = []
    for idx, bond in enumerate(mol.GetBonds()):
        a2 = bond.GetEndAtomIdx()
        a1 = bond.GetBeginAtomIdx()
        edge_dir1.append(a1)
        edge_dir1.append(a2)
        edge_dir2.append(a2)
        edge_dir2.append(a1)

        if mol.GetAtoms()[a1].GetIsotope() == 2:
            trg_atoms.append(a2)

        if mol.GetAtoms()[a2].GetIsotope() == 2:
            trg_atoms.append(a1)

    edge_2d = torch.from_numpy(np.array([edge_dir1, edge_dir2]))

    target = [0 for i, x in enumerate(atomids)]
    for trg_atom in trg_atoms:
        target[trg_atom] = 1

    atomids = np.array(atomids)
    target = np.array(target)
    qml_atomids = np.array(qml_atomids)
    is_ring = np.array(is_ring)
    hyb = np.array(hyb)
    arom = np.array(arom)
    crds_3d = np.array(crds_3d)
    pot_trg = np.array(pot_trg)

    # 3D graph for qml and qml prediction
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

    charges = QMLMODEL(qml_graph).unsqueeze(1).detach().numpy()

    # Get edges for 3d graph
    radius = 4  # 5.29177 = 10 a0
    edge1 = []
    edge2 = []
    for i in range(len(atomids)):
        for j in range(len(atomids)):
            if i != j:
                dist = np.linalg.norm(crds_3d[j] - crds_3d[i])
                if dist <= radius:
                    edge1.append(i)
                    edge2.append(j)
                    edge1.append(j)
                    edge2.append(i)

    edge_3d = torch.from_numpy(np.array([edge1, edge2]))

    return (
        atomids,
        is_ring,
        hyb,
        arom,
        charges,
        edge_2d,
        edge_3d,
        crds_3d,
        pot_trg,
        target,
    )


if __name__ == "__main__":

    # Read csv
    df = pd.read_csv(os.path.join(UTILS_PATH, "data/literature_rxndata.csv"), encoding="unicode_escape")
    smiles = list(df["product_1_smiles"])
    rxn_yield = list(df["product_1_yield"])

    print(f"Initial number of reactions: {len(smiles)}")
    rxn_yield = [0.0 if np.isnan(x) else x for x in rxn_yield]
    smiles = [s for i, s in enumerate(smiles) if rxn_yield[i] >= 0.3]
    print(f"Number of reactions after removing yields <= 30%: {len(smiles)}")
    smiles = [s for i, s in enumerate(smiles) if len(str(smiles[i])) >= 5]
    print(f"Number of reactions after removing nan SMILES: {len(smiles)}")
    uniques = list(set(smiles))
    print(f"Number of reactions after removing duplicate SMILES: {len(uniques)}")

    repl = Chem.MolFromSmiles("[2H]")
    patt = Chem.MolFromSmarts("B2OC(C)(C)C(O2)(C)C")

    wins = 0
    loss = 0
    rxn_key = 0

    all_smiles = []
    h5_path = os.path.join(UTILS_PATH, "data/literature_regio.h5")

    with h5py.File(h5_path, "w") as lsf_container:

        for smi in tqdm(uniques):

            try:

                rms = get_rms(smi, patt, repl)

                if "[2H]" in rms:

                    (
                        atom_id,
                        ring_id,
                        hybr_id,
                        arom_id,
                        charges,
                        edge_2d,
                        edge_3d,
                        crds_3d,
                        pot_trg,
                        reg_trg,
                    ) = get_regioselectivity(rms, 0xF10D)

                    # Create group in h5 for this id
                    lsf_container.create_group(str(rxn_key))

                    # Molecule
                    lsf_container[str(rxn_key)].create_dataset("atom_id", data=atom_id)
                    lsf_container[str(rxn_key)].create_dataset("ring_id", data=ring_id)
                    lsf_container[str(rxn_key)].create_dataset("hybr_id", data=hybr_id)
                    lsf_container[str(rxn_key)].create_dataset("arom_id", data=arom_id)
                    lsf_container[str(rxn_key)].create_dataset("edge_2d", data=edge_2d)
                    lsf_container[str(rxn_key)].create_dataset("edge_3d", data=edge_3d)
                    lsf_container[str(rxn_key)].create_dataset("charges", data=charges)
                    lsf_container[str(rxn_key)].create_dataset("crds_3d", data=crds_3d)
                    lsf_container[str(rxn_key)].create_dataset("pot_trg", data=pot_trg)
                    lsf_container[str(rxn_key)].create_dataset("reg_trg", data=reg_trg)

                    all_smiles.append(rms)

                    wins += 1
                    rxn_key += 1

                else:
                    print(f"No boron in product or unconventional boron in product: {rms}, {smi}")

            except:
                loss += 1

    print(f"Reactions sucessfully transformed: {wins}; Reactions failed {loss}")

    df = pd.DataFrame(
        {
            "all_smiles": all_smiles,
        }
    )
    df.to_csv(os.path.join(UTILS_PATH, "data/literature_regio.csv"), sep=",", encoding="utf-8", index=False)
