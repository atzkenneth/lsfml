#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (©) 2023 Kenneth Atz (ETH Zurich)

import h5py
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
from lsfml.utils import get_dict_for_embedding, get_fp_from_smi, HYBRIDISATIONS, AROMATOCITY, IS_RING, ATOMTYPES, QML_ATOMTYPES, UTILS_PATH

QMLMODEL = get_model(gpu=False)


HYBRIDISATION_DICT = get_dict_for_embedding(HYBRIDISATIONS)
AROMATOCITY_DICT = get_dict_for_embedding(AROMATOCITY)
IS_RING_DICT = get_dict_for_embedding(IS_RING)
ATOMTYPE_DICT = get_dict_for_embedding(ATOMTYPES)
QML_ATOMTYPE_DICT = get_dict_for_embedding(QML_ATOMTYPES)


def get_info_from_smi(smi, randomseed):
    """Main function for extracting relevant reaction conditions and generating the 2D and 3D molecular graphs given a SMILES-string and a seed for 3D conformer generation.

    :param smi: SMILES-string
    :type smi: str
    :param randomseed: random seed
    :type randomseed: int
    :return: tuple including all graph-relevant numpy arrays
    :rtype: tuple
    """
    # Get mol objects from smiles
    mol_no_Hs = Chem.MolFromSmiles(smi)
    mol = Chem.rdmolops.AddHs(mol_no_Hs)

    atomids = []
    qml_atomids = []
    is_ring = []
    hyb = []
    arom = []
    crds_3d = []

    AllChem.EmbedMolecule(mol, randomSeed=randomseed)
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
    )


if __name__ == "__main__":

    df = pd.read_csv(os.path.join(UTILS_PATH, "data/experimental_rxndata.csv"))

    # Rxn id
    rxn_id = list(df["rxn_id"])

    # Substrate
    educt = list(df["educt"])

    # Non-molecular conditions
    rxn_scale_mol = list(df["rxn_scale_mol"])
    rxn_temp_C = list(df["rxn_temp_C"])
    rxn_time_h = list(df["rxn_time_h"])
    rxn_atm = list(df["rxn_atm"])
    rxn_c_moll = list(df["rxn_c_moll"])

    # Molecular conditions
    catalyst = list(df["catalyst"])
    catalyst_eq = list(df["catalyst_eq"])
    ligand = list(df["ligand"])
    ligand_eq = list(df["ligand_eq"])
    reagent = list(df["reagent"])
    reagent_eq = list(df["reagent_eq"])
    solvent = list(df["solvent"])
    solvent_ratio = list(df["solvent_ratio"])

    # Targets
    yes_no = list(df["yes_no"])
    mono_bo = list(df["mono_bo"])
    di_bo = list(df["di_bo"])
    non_bo = list(df["non_bo"])

    # Embedding of molecular conditions
    rea_dict = get_dict_for_embedding(reagent)
    lig_dict = get_dict_for_embedding(ligand)
    cat_dict = get_dict_for_embedding(catalyst)
    sol_dict = get_dict_for_embedding(solvent)

    print("Liands in data set:", lig_dict)
    print("Solvents in data set:", sol_dict)

    # Get molecule-dict for short rxids

    unique_substraes = {}

    for idx, rxn_key in enumerate(rxn_id):

        short_rxn_key = rxn_key.split("_")[0]
        short_rxn_key = short_rxn_key.split("-")[-1]

        if short_rxn_key not in unique_substraes:

            unique_substraes[short_rxn_key] = educt[idx]

        else:
            pass

    print(f"Calculating properties for {len(unique_substraes)} unique substartes")

    with h5py.File("../data/experimental_substrates.h5", "w") as lsf_container1:

        for rxn_key in tqdm(unique_substraes):

            (
                atom_id_a,
                ring_id_a,
                hybr_id_a,
                arom_id_a,
                charges_a,
                edge_2d_a,
                edge_3d_a,
                crds_3d_a,
            ) = get_info_from_smi(unique_substraes[rxn_key], 0xF00A)

            (
                atom_id_b,
                ring_id_b,
                hybr_id_b,
                arom_id_b,
                charges_b,
                edge_2d_b,
                edge_3d_b,
                crds_3d_b,
            ) = get_info_from_smi(unique_substraes[rxn_key], 0xF00B)

            (
                atom_id_c,
                ring_id_c,
                hybr_id_c,
                arom_id_c,
                charges_c,
                edge_2d_c,
                edge_3d_c,
                crds_3d_c,
            ) = get_info_from_smi(unique_substraes[rxn_key], 0xF00C)

            (
                atom_id_d,
                ring_id_d,
                hybr_id_d,
                arom_id_d,
                charges_d,
                edge_2d_d,
                edge_3d_d,
                crds_3d_d,
            ) = get_info_from_smi(unique_substraes[rxn_key], 0xF00D)

            (
                atom_id_e,
                ring_id_e,
                hybr_id_e,
                arom_id_e,
                charges_e,
                edge_2d_e,
                edge_3d_e,
                crds_3d_e,
            ) = get_info_from_smi(unique_substraes[rxn_key], 0xF00E)

            # Substrate ID
            lsf_container1.create_group(rxn_key)

            # Molecule
            lsf_container1[rxn_key].create_dataset("atom_id_a", data=atom_id_a)
            lsf_container1[rxn_key].create_dataset("ring_id_a", data=ring_id_a)
            lsf_container1[rxn_key].create_dataset("hybr_id_a", data=hybr_id_a)
            lsf_container1[rxn_key].create_dataset("arom_id_a", data=arom_id_a)
            lsf_container1[rxn_key].create_dataset("charges_a", data=charges_a)
            lsf_container1[rxn_key].create_dataset("edge_2d_a", data=edge_2d_a)
            lsf_container1[rxn_key].create_dataset("edge_3d_a", data=edge_3d_a)
            lsf_container1[rxn_key].create_dataset("crds_3d_a", data=crds_3d_a)
            lsf_container1[rxn_key].create_dataset("atom_id_b", data=atom_id_b)
            lsf_container1[rxn_key].create_dataset("ring_id_b", data=ring_id_b)
            lsf_container1[rxn_key].create_dataset("hybr_id_b", data=hybr_id_b)
            lsf_container1[rxn_key].create_dataset("arom_id_b", data=arom_id_b)
            lsf_container1[rxn_key].create_dataset("charges_b", data=charges_b)
            lsf_container1[rxn_key].create_dataset("edge_2d_b", data=edge_2d_b)
            lsf_container1[rxn_key].create_dataset("edge_3d_b", data=edge_3d_b)
            lsf_container1[rxn_key].create_dataset("crds_3d_b", data=crds_3d_b)
            lsf_container1[rxn_key].create_dataset("atom_id_c", data=atom_id_c)
            lsf_container1[rxn_key].create_dataset("ring_id_c", data=ring_id_c)
            lsf_container1[rxn_key].create_dataset("hybr_id_c", data=hybr_id_c)
            lsf_container1[rxn_key].create_dataset("arom_id_c", data=arom_id_c)
            lsf_container1[rxn_key].create_dataset("charges_c", data=charges_c)
            lsf_container1[rxn_key].create_dataset("edge_2d_c", data=edge_2d_c)
            lsf_container1[rxn_key].create_dataset("edge_3d_c", data=edge_3d_c)
            lsf_container1[rxn_key].create_dataset("crds_3d_c", data=crds_3d_c)
            lsf_container1[rxn_key].create_dataset("atom_id_d", data=atom_id_d)
            lsf_container1[rxn_key].create_dataset("ring_id_d", data=ring_id_d)
            lsf_container1[rxn_key].create_dataset("hybr_id_d", data=hybr_id_d)
            lsf_container1[rxn_key].create_dataset("arom_id_d", data=arom_id_d)
            lsf_container1[rxn_key].create_dataset("charges_d", data=charges_d)
            lsf_container1[rxn_key].create_dataset("edge_2d_d", data=edge_2d_d)
            lsf_container1[rxn_key].create_dataset("edge_3d_d", data=edge_3d_d)
            lsf_container1[rxn_key].create_dataset("crds_3d_d", data=crds_3d_d)
            lsf_container1[rxn_key].create_dataset("atom_id_e", data=atom_id_e)
            lsf_container1[rxn_key].create_dataset("ring_id_e", data=ring_id_e)
            lsf_container1[rxn_key].create_dataset("hybr_id_e", data=hybr_id_e)
            lsf_container1[rxn_key].create_dataset("arom_id_e", data=arom_id_e)
            lsf_container1[rxn_key].create_dataset("charges_e", data=charges_e)
            lsf_container1[rxn_key].create_dataset("edge_2d_e", data=edge_2d_e)
            lsf_container1[rxn_key].create_dataset("edge_3d_e", data=edge_3d_e)
            lsf_container1[rxn_key].create_dataset("crds_3d_e", data=crds_3d_e)

    wins = 0
    loss = 0

    print(f"Transforming {len(rxn_id)} reactions into h5 format")

    with h5py.File("../data/experimental_rxndata.h5", "w") as lsf_container:

        for idx, rxn_key in enumerate(tqdm(rxn_id)):

            try:

                rgnt_id = rea_dict[reagent[idx]]
                lgnd_id = lig_dict[ligand[idx]]
                clst_id = cat_dict[catalyst[idx]]
                slvn_id = sol_dict[solvent[idx]]

                # Create group in h5 for this id
                lsf_container.create_group(rxn_key)

                # Add all parameters as datasets to the created group
                if ligand_eq[idx] == "none":
                    lgnd_eq = 0.0
                else:
                    lgnd_eq = ligand_eq[idx]

                # Molecule
                ecfp4_2 = get_fp_from_smi(educt[idx])
                lsf_container[rxn_key].create_dataset("ecfp4_2", data=[ecfp4_2])

                # Conditions
                lsf_container[rxn_key].create_dataset("rgnt_id", data=[int(rgnt_id)])
                lsf_container[rxn_key].create_dataset("lgnd_id", data=[int(lgnd_id)])
                lsf_container[rxn_key].create_dataset("clst_id", data=[int(clst_id)])
                lsf_container[rxn_key].create_dataset("slvn_id", data=[int(slvn_id)])
                lsf_container[rxn_key].create_dataset("rgnt_eq", data=[float(reagent_eq[idx])])
                lsf_container[rxn_key].create_dataset("lgnd_eq", data=[float(lgnd_eq)])
                lsf_container[rxn_key].create_dataset("clst_eq", data=[float(catalyst_eq[idx])])
                lsf_container[rxn_key].create_dataset("rxn_scl", data=[float(rxn_scale_mol[idx])])
                lsf_container[rxn_key].create_dataset("rxn_con", data=[float(rxn_c_moll[idx])])
                lsf_container[rxn_key].create_dataset("rxn_tmp", data=[float(rxn_temp_C[idx])])
                lsf_container[rxn_key].create_dataset("rxn_tme", data=[float(rxn_time_h[idx])])

                # Tragets
                lsf_container[rxn_key].create_dataset("mono_id", data=[yes_no[idx]])
                lsf_container[rxn_key].create_dataset("mo_frct", data=[[mono_bo[idx]]])
                lsf_container[rxn_key].create_dataset("di_frct", data=[[di_bo[idx]]])
                lsf_container[rxn_key].create_dataset("no_frct", data=[[non_bo[idx]]])

                wins += 1

            except:
                loss += 1

    print(f"Reactions sucessfully transformed: {wins}; Reactions failed: {loss}")
