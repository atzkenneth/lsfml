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
from rdkit.Chem import AllChem, rdMolDescriptors
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.utils.undirected import to_undirected
from tqdm import tqdm

from lsfml.qml.prod import get_model
from lsfml.utils import get_dict_for_embedding, HYBRIDISATIONS, AROMATOCITY, IS_RING, ATOMTYPES, QML_ATOMTYPES, UTILS_PATH

QMLMODEL = get_model(gpu=False)


HYBRIDISATION_DICT = get_dict_for_embedding(HYBRIDISATIONS)
AROMATOCITY_DICT = get_dict_for_embedding(AROMATOCITY)
IS_RING_DICT = get_dict_for_embedding(IS_RING)
ATOMTYPE_DICT = get_dict_for_embedding(ATOMTYPES)
QML_ATOMTYPE_DICT = get_dict_for_embedding(QML_ATOMTYPES)

sol_dict = {
    "O1CCCC1": 0,
    "C=1C=C(C=CC1C)C": 1,
    "N#CC": 2,
    "CCCCCC": 3,
    "O(C)C1CCCC1": 4,
    "O(C)C(C)(C)C": 5,
    "C1CCCCC1": 6,
    "C1CCCCCCC1": 7,
    "ClCCCl": 8,
}

rea_dict = {
    "O1B(OC(C)(C)C1(C)C)B2OC(C)(C)C(O2)(C)C": 0,
    "O1BOC(C)(C)C1(C)C": 1,
}

cat_dict = {
    "[O-]1(C)[Ir+]234([O-](C)[Ir+]1567[CH]=8CC[CH]7=[CH]6CC[CH]85)[CH]=9CC[CH]4=[CH]3CC[CH]92": 0,
    "[Cl-]1[Ir+]234([Cl-][Ir+]1567[CH]=8CC[CH]7=[CH]6CC[CH]85)[CH]=9CC[CH]4=[CH]3CC[CH]92": 1,
    "[OH-]1[Ir+]234([OH-][Ir+]1567[CH]=8CC[CH]7=[CH]6CC[CH]85)[CH]=9CC[CH]4=[CH]3CC[CH]92": 2,
    "O1=C([CH-]C(=O[Ir+]1234[CH]=5CC[CH]4=[CH]3CC[CH]52)C)C": 3,
}

lig_dict = {
    "N=1C=CC(=CC1C=2N=CC=C(C2)C(C)(C)C)C(C)(C)C": 0,
    "O=C1C=CC=2C=CC=C(C3=CN=C(C=C3)C=4N=CC=CC4)C2N1": 1,
    "N=1C=C(C(=C2C=CC3=C(N=CC(=C3C)C)C12)C)C": 2,
    "O=S(=O)([O-])CC=1C=NC(=CC1)C2=NC=C(C=C2)C.CCCC[N+](CCCC)(CCCC)CCCC": 3,
    "O=C(NC=1C=CC=CC1C=2C=NC(=CC2)C3=NC=CC=C3)NC4CCCCC4": 4,
    "N=1C=CC=CC1N2B(NC=3C=CC=CC32)B4NC=5C=CC=CC5N4C6=NC=CC=C6": 5,
    "O=C(NC1=CC=CC2=C1NC(=C2C)C)C=3C=NC(=CC3)C4=NC=CC=C4": 6,
    "N=1C=CC=C2C=CC=3C=CC(=NC3C12)C": 7,
    "O(C1=CC=CC(=C1C=2C(OC)=CC=CC2P(C=3C=C(C=C(C3)C)C)C=4C=C(C=C(C4)C)C)P(C=5C=C(C=C(C5)C)C)C=6C=C(C=C(C6)C)C)C": 8,
    "N=1C=CC(=CC1C=2N=CC=C(C2)C)C": 9,
    "FC(F)(F)C1OB(OC1)C=2C=CC=CC2C=3C=NC(=CC3)C4=NC=CC=C4": 10,
    "N=1C=CC=CC1C=2N=CC=CC2": 11,
}


def get_info_from_smi(smi, radius):
    """Main function for extracting relevant reaction conditions and generating the 2D and 3D molecular graphs given a SMILES-string.

    :param smi: SMILES-string
    :type smi: str
    :return: tuple including all graph-relevant numpy arrays
    :rtype: tuple
    """

    # Get mol objects from smiles
    mol_no_Hs = Chem.MolFromSmiles(smi)
    mol = Chem.rdmolops.AddHs(mol_no_Hs)

    ecfp4_fp = np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=256))

    atomids = []
    qml_atomids = []
    is_ring = []
    hyb = []
    arom = []
    crds_3d = []

    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    AllChem.UFFOptimizeMolecule(mol)

    for idx, i in enumerate(mol.GetAtoms()):
        atomids.append(ATOMTYPE_DICT[i.GetSymbol()])
        qml_atomids.append(QML_ATOMTYPE_DICT[i.GetSymbol()])
        is_ring.append(IS_RING_DICT[str(i.IsInRing())])
        hyb.append(HYBRIDISATION_DICT[str(i.GetHybridization())])
        arom.append(AROMATOCITY_DICT[str(i.GetIsAromatic())])
        crds_3d.append(list(mol.GetConformer().GetAtomPosition(idx)))

    atomids = np.array(atomids)
    qml_atomids = np.array(qml_atomids)
    is_ring = np.array(is_ring)
    hyb = np.array(hyb)
    arom = np.array(arom)
    crds_3d = np.array(crds_3d)

    # Edges for covalent bonds in sdf file
    edge_dir1 = []
    edge_dir2 = []
    for idx, bond in enumerate(mol.GetBonds()):
        a2 = bond.GetEndAtomIdx()
        a1 = bond.GetBeginAtomIdx()
        edge_dir1.append(a1)
        edge_dir1.append(a2)
        edge_dir2.append(a2)
        edge_dir2.append(a1)

    edge_2d = torch.from_numpy(np.array([edge_dir1, edge_dir2]))

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

    # get edges for 3d graph
    edge1 = []
    edge2 = []
    for i in range(len(atomids)):
        for j in range(len(atomids)):
            if i != j:
                dist = np.linalg.norm(crds_3d[j] - crds_3d[i])
                if dist <= radius:
                    edge1.append(i)
                    edge2.append(j)

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
        ecfp4_fp,
    )


if __name__ == "__main__":

    df = pd.read_csv(os.path.join(UTILS_PATH, "data/literature_rxndata.csv"), encoding="unicode_escape")

    # Rxn id
    rxn_id = list(df["rxn_id"])

    # Substrate
    educt = list(df["stmat_1_smiles"])

    # Molecular conditions
    catalyst = list(df["catalyst_1_smiles"])
    catalyst_eq = list(df["catalyst_1_eq"])
    ligand = list(df["ligand_1_smiles"])
    ligand_eq = list(df["ligand_1_eq"])
    reagent = list(df["reagent_1_smiles"])
    reagent_eq = list(df["reagent_1_eq"])
    solvent = list(df["solvent_1_smiles"])
    solvent_ratio = list(df["solvent_1_fraction"])

    # Targets
    trg = list(df["product_1_yield"])

    # Get molecule-dict for short rxids
    print("Calculating properties for all substartes")

    wins = 0
    loss = 0

    print(f"Transforming {len(rxn_id)} reactions into h5 format")

    h5_path = os.path.join(UTILS_PATH, "data/literature_rxndata.h5")

    with h5py.File(h5_path, "w") as lsf_container:

        for idx, rxn_key in enumerate(tqdm(rxn_id)):

            try:

                (
                    atom_id,
                    ring_id,
                    hybr_id,
                    arom_id,
                    charges,
                    edge_2d,
                    edge_3d,
                    crds_3d,
                    ecfp4_2,
                ) = get_info_from_smi(educt[idx], 4)

                rgnt_id = rea_dict[reagent[idx]]
                lgnd_id = lig_dict[ligand[idx]]
                clst_id = cat_dict[catalyst[idx]]
                slvn_id = sol_dict[solvent[idx]]
                trg_rxn = trg[idx]

                if np.isnan(trg_rxn):
                    trg_rxn = 0.0

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
                lsf_container[str(rxn_key)].create_dataset("ecfp4_2", data=[ecfp4_2])

                # Conditions
                lsf_container[str(rxn_key)].create_dataset("rgnt_id", data=[int(rgnt_id)])
                lsf_container[str(rxn_key)].create_dataset("lgnd_id", data=[int(lgnd_id)])
                lsf_container[str(rxn_key)].create_dataset("clst_id", data=[int(clst_id)])
                lsf_container[str(rxn_key)].create_dataset("slvn_id", data=[int(slvn_id)])

                # Traget
                lsf_container[str(rxn_key)].create_dataset("trg_rxn", data=[trg_rxn])

                wins += 1

            except:
                loss += 1

    print(f"Reactions sucessfully transformed: {wins}; Reactions failed: {loss}")
