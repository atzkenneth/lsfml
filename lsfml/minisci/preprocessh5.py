from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem

import torch, h5py
import pandas as pd
from collections import Counter
from tqdm import tqdm
import numpy as np
import networkx as nx

from torch_geometric.data import Data
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import add_self_loops

# from lsfqml.roche_experiments.qml.prod import get_model
# QMLMODEL = get_model(gpu=False)


def get_dict_for_embedding(list):
    """Creates a dict x: 0 - x: N for x in List with len(list(set(List))) = N"""
    list_dict = {}
    list_counter = Counter(list)
    for idx, x in enumerate(list_counter):
        list_dict[x] = idx
    return list_dict


HYBRIDISATIONS = [
    "SP3",
    "SP2",
    "SP",
    "UNSPECIFIED",
    "S",
]
AROMATOCITY = [
    "True",
    "False",
]
IS_RING = [
    "True",
    "False",
]
ATOMTYPES = [
    "H",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
]  # "Si", "B",
QML_ATOMTYPES = [
    "X",
    "H",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
]

HYBRIDISATION_DICT = get_dict_for_embedding(HYBRIDISATIONS)
AROMATOCITY_DICT = get_dict_for_embedding(AROMATOCITY)
IS_RING_DICT = get_dict_for_embedding(IS_RING)
ATOMTYPE_DICT = get_dict_for_embedding(ATOMTYPES)
QML_ATOMTYPE_DICT = get_dict_for_embedding(QML_ATOMTYPES)


def get_fp_from_smi(smi):
    """Get ECFP from SMILES"""
    mol_no_Hs = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol_no_Hs)

    return np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=256))


def get_3dG_from_smi(smi, randomSeed):
    # get mol objects from smiles
    mol_no_Hs = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol_no_Hs)

    atomids = []
    qml_atomids = []
    is_ring = []
    hyb = []
    arom = []
    crds_3d = []
    tokeep = []

    AllChem.EmbedMolecule(mol, randomSeed)  # 0xf00d
    AllChem.UFFOptimizeMolecule(mol)

    for idx, i in enumerate(mol.GetAtoms()):
        atomids.append(ATOMTYPE_DICT[i.GetSymbol()])
        qml_atomids.append(QML_ATOMTYPE_DICT[i.GetSymbol()])
        is_ring.append(IS_RING_DICT[str(i.IsInRing())])
        hyb.append(HYBRIDISATION_DICT[str(i.GetHybridization())])
        arom.append(AROMATOCITY_DICT[str(i.GetIsAromatic())])
        crds_3d.append(list(mol.GetConformer().GetAtomPosition(idx)))
        if (
            (ATOMTYPE_DICT[i.GetSymbol()] == 1)
            and (IS_RING_DICT[str(i.IsInRing())] == 0)
            and (AROMATOCITY_DICT[str(i.GetIsAromatic())] == 0)
            and (HYBRIDISATION_DICT[str(i.GetHybridization())] == 1)
        ):
            # print(i.GetSymbol(), str(i.IsInRing()), str(i.GetIsAromatic()),
            # str(i.GetHybridization()), i.GetTotalNumHs(), i.GetExplicitValence())
            tokeep.append(1)
        else:
            tokeep.append(0)

    atomids = np.array(atomids)
    qml_atomids = np.array(qml_atomids)
    is_ring = np.array(is_ring)
    hyb = np.array(hyb)
    arom = np.array(arom)
    crds_3d = np.array(crds_3d)
    tokeep = np.array(tokeep)

    # edges for covalent bonds in sdf file
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

    # qml_graph = Data(
    #     atomids=qml_atomids,
    #     coords=xyzs,
    #     edge_index=edge_index,
    #     num_nodes=qml_atomids.size(0),
    # )

    # charges = QMLMODEL(qml_graph).unsqueeze(1).detach().numpy()

    # get edges for 3d graph
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

    """# print out test xyz file with charges
    print(mol.GetNumAtoms(), "\n")
    for idx, i in enumerate(mol.GetAtoms()):
        xyz = list(mol.GetConformer().GetAtomPosition(idx))
        print(i.GetSymbol(), xyz[0], xyz[1], xyz[2], str(i.GetIsAromatic()), str(i.GetHybridization()), charges[idx])"""

    return (
        atomids,
        is_ring,
        hyb,
        arom,
        # charges,
        edge_2d,
        edge_3d,
        crds_3d,
        tokeep,
    )


if __name__ == "__main__":
    name = "hte_only"  # hte_only, plus_lit
    df = pd.read_csv(f"data/20230603_alkylation_rxndata_full.csv", sep=",", encoding="unicode_escape")  # *hte.h5

    # read csv id
    rxn_id = list(df["rxn_id"])
    startingmat_1_smiles = list(df["startingmat_1_smiles"])
    startingmat_2_smiles = list(df["startingmat_2_smiles"])
    reagent_1_smiles = list(df["reagent_1_smiles"])
    solvent_1_smiles = list(df["solvent_1_smiles"])
    solvent_2_smiles = list(df["solvent_2_smiles"])
    reagent_1_eq = list(df["reagent_1_eq"])
    startingmat_2_eq = list(df["startingmat_2_eq"])
    concentration_moll = list(df["concentration_moll"])
    product_mono = list(df["product_mono"])
    product_di = list(df["product_di"])
    product_non = list(df["product_non"])
    binary = list(df["binary"])

    print(f"Number of individual reactions: {len(rxn_id)}")
    # acids = list(set(startingmat_2_smiles))

    rea_dict = get_dict_for_embedding(reagent_1_smiles)
    so1_dict = get_dict_for_embedding(solvent_1_smiles)
    so2_dict = get_dict_for_embedding(solvent_2_smiles)
    print(
        "Distinct reagent and solvent types (should be one each): ",
        list(rea_dict.keys()),
        list(so1_dict.keys()),
        list(so2_dict.keys()),
    )
    product_all = [float(product_mono[i] + product_di[i]) for i, x in enumerate(product_mono)]

    # Get unique substrates
    startingmat_1_unique = list(set(startingmat_1_smiles))
    startingmat_2_unique = list(set(startingmat_2_smiles))
    print(
        f"Number of unique reactions: {len(rxn_id)} \nNumber of substrates: {len(startingmat_1_unique)}, \nNumber of carboxylic acids: {len(startingmat_2_unique)}"
    )

    # Get the rxn keys for the two starting materials
    rxn_smi_dict = {}
    for i, rxn in enumerate(rxn_id):
        rxn_smi_dict[rxn] = [startingmat_1_smiles[i], startingmat_2_smiles[i]]

    torch.save(rxn_smi_dict, f"data/rxn_smi_dict_{name}.pt")

    # Get 3D graph data
    print(
        f"\nGenerating 3D conformers and QM predictions for {len(startingmat_1_unique)} substrates and saving them into h5 format:"
    )

    with h5py.File(f"data/lsf_rxn_substrate_{name}.h5", "w") as lsf_container1:
        for smi in tqdm(startingmat_1_unique):
            (
                atom_id_1_a,
                ring_id_1_a,
                hybr_id_1_a,
                arom_id_1_a,
                # charges_1_a,
                edge_2d_1_a,
                edge_3d_1_a,
                crds_3d_1_a,
                to_keep_1_a,
            ) = get_3dG_from_smi(smi, 0xF00A)

            (
                atom_id_1_b,
                ring_id_1_b,
                hybr_id_1_b,
                arom_id_1_b,
                # charges_1_b,
                edge_2d_1_b,
                edge_3d_1_b,
                crds_3d_1_b,
                to_keep_1_b,
            ) = get_3dG_from_smi(smi, 0xF00B)

            (
                atom_id_1_c,
                ring_id_1_c,
                hybr_id_1_c,
                arom_id_1_c,
                # charges_1_c,
                edge_2d_1_c,
                edge_3d_1_c,
                crds_3d_1_c,
                to_keep_1_c,
            ) = get_3dG_from_smi(smi, 0xF00C)

            (
                atom_id_1_d,
                ring_id_1_d,
                hybr_id_1_d,
                arom_id_1_d,
                # charges_1_d,
                edge_2d_1_d,
                edge_3d_1_d,
                crds_3d_1_d,
                to_keep_1_d,
            ) = get_3dG_from_smi(smi, 0xF00D)

            (
                atom_id_1_e,
                ring_id_1_e,
                hybr_id_1_e,
                arom_id_1_e,
                # charges_1_e,
                edge_2d_1_e,
                edge_3d_1_e,
                crds_3d_1_e,
                to_keep_1_e,
            ) = get_3dG_from_smi(smi, 0xF00E)

            # Substrate ID
            lsf_container1.create_group(smi)

            # Molecule
            lsf_container1[smi].create_dataset("atom_id_1_a", data=atom_id_1_a)
            lsf_container1[smi].create_dataset("ring_id_1_a", data=ring_id_1_a)
            lsf_container1[smi].create_dataset("hybr_id_1_a", data=hybr_id_1_a)
            lsf_container1[smi].create_dataset("arom_id_1_a", data=arom_id_1_a)
            # lsf_container1[smi].create_dataset("charges_1_a", data=charges_1_a)
            lsf_container1[smi].create_dataset("edge_2d_1_a", data=edge_2d_1_a)
            lsf_container1[smi].create_dataset("edge_3d_1_a", data=edge_3d_1_a)
            lsf_container1[smi].create_dataset("crds_3d_1_a", data=crds_3d_1_a)
            lsf_container1[smi].create_dataset("to_keep_1_a", data=to_keep_1_a)
            lsf_container1[smi].create_dataset("atom_id_1_b", data=atom_id_1_b)
            lsf_container1[smi].create_dataset("ring_id_1_b", data=ring_id_1_b)
            lsf_container1[smi].create_dataset("hybr_id_1_b", data=hybr_id_1_b)
            lsf_container1[smi].create_dataset("arom_id_1_b", data=arom_id_1_b)
            # lsf_container1[smi].create_dataset("charges_1_b", data=charges_1_b)
            lsf_container1[smi].create_dataset("edge_2d_1_b", data=edge_2d_1_b)
            lsf_container1[smi].create_dataset("edge_3d_1_b", data=edge_3d_1_b)
            lsf_container1[smi].create_dataset("crds_3d_1_b", data=crds_3d_1_b)
            lsf_container1[smi].create_dataset("to_keep_1_b", data=to_keep_1_b)
            lsf_container1[smi].create_dataset("atom_id_1_c", data=atom_id_1_c)
            lsf_container1[smi].create_dataset("ring_id_1_c", data=ring_id_1_c)
            lsf_container1[smi].create_dataset("hybr_id_1_c", data=hybr_id_1_c)
            lsf_container1[smi].create_dataset("arom_id_1_c", data=arom_id_1_c)
            # lsf_container1[smi].create_dataset("charges_1_c", data=charges_1_c)
            lsf_container1[smi].create_dataset("edge_2d_1_c", data=edge_2d_1_c)
            lsf_container1[smi].create_dataset("edge_3d_1_c", data=edge_3d_1_c)
            lsf_container1[smi].create_dataset("crds_3d_1_c", data=crds_3d_1_c)
            lsf_container1[smi].create_dataset("to_keep_1_c", data=to_keep_1_c)
            lsf_container1[smi].create_dataset("atom_id_1_d", data=atom_id_1_d)
            lsf_container1[smi].create_dataset("ring_id_1_d", data=ring_id_1_d)
            lsf_container1[smi].create_dataset("hybr_id_1_d", data=hybr_id_1_d)
            lsf_container1[smi].create_dataset("arom_id_1_d", data=arom_id_1_d)
            # lsf_container1[smi].create_dataset("charges_1_d", data=charges_1_d)
            lsf_container1[smi].create_dataset("edge_2d_1_d", data=edge_2d_1_d)
            lsf_container1[smi].create_dataset("edge_3d_1_d", data=edge_3d_1_d)
            lsf_container1[smi].create_dataset("crds_3d_1_d", data=crds_3d_1_d)
            lsf_container1[smi].create_dataset("to_keep_1_d", data=to_keep_1_d)
            lsf_container1[smi].create_dataset("atom_id_1_e", data=atom_id_1_e)
            lsf_container1[smi].create_dataset("ring_id_1_e", data=ring_id_1_e)
            lsf_container1[smi].create_dataset("hybr_id_1_e", data=hybr_id_1_e)
            lsf_container1[smi].create_dataset("arom_id_1_e", data=arom_id_1_e)
            # lsf_container1[smi].create_dataset("charges_1_e", data=charges_1_e)
            lsf_container1[smi].create_dataset("edge_2d_1_e", data=edge_2d_1_e)
            lsf_container1[smi].create_dataset("edge_3d_1_e", data=edge_3d_1_e)
            lsf_container1[smi].create_dataset("crds_3d_1_e", data=crds_3d_1_e)
            lsf_container1[smi].create_dataset("to_keep_1_e", data=to_keep_1_e)

    h5f1 = h5py.File(f"data/lsf_rxn_substrate_{name}.h5")
    print(f"Successfully transformed {len(list(h5f1.keys()))} substrates")

    print(
        f"\nGenerating 3D conformers and QM predictions for {len(startingmat_2_unique)} carboxylic acids and saving them into h5 format:"
    )

    with h5py.File(f"data/lsf_rxn_carbacids_{name}.h5", "w") as lsf_container2:
        for smi in tqdm(startingmat_2_unique):
            (
                atom_id_2_a,
                ring_id_2_a,
                hybr_id_2_a,
                arom_id_2_a,
                # charges_2_a,
                edge_2d_2_a,
                edge_3d_2_a,
                crds_3d_2_a,
                to_keep_2_a,
            ) = get_3dG_from_smi(smi, 0xF00A)

            (
                atom_id_2_b,
                ring_id_2_b,
                hybr_id_2_b,
                arom_id_2_b,
                # charges_2_b,
                edge_2d_2_b,
                edge_3d_2_b,
                crds_3d_2_b,
                to_keep_2_b,
            ) = get_3dG_from_smi(smi, 0xF00B)

            (
                atom_id_2_c,
                ring_id_2_c,
                hybr_id_2_c,
                arom_id_2_c,
                # charges_2_c,
                edge_2d_2_c,
                edge_3d_2_c,
                crds_3d_2_c,
                to_keep_2_c,
            ) = get_3dG_from_smi(smi, 0xF00C)

            (
                atom_id_2_d,
                ring_id_2_d,
                hybr_id_2_d,
                arom_id_2_d,
                # charges_2_d,
                edge_2d_2_d,
                edge_3d_2_d,
                crds_3d_2_d,
                to_keep_2_d,
            ) = get_3dG_from_smi(smi, 0xF00D)

            (
                atom_id_2_e,
                ring_id_2_e,
                hybr_id_2_e,
                arom_id_2_e,
                # charges_2_e,
                edge_2d_2_e,
                edge_3d_2_e,
                crds_3d_2_e,
                to_keep_2_e,
            ) = get_3dG_from_smi(smi, 0xF00E)

            # Substrate ID
            lsf_container2.create_group(smi)

            # Molecule
            lsf_container2[smi].create_dataset("atom_id_2_a", data=atom_id_2_a)
            lsf_container2[smi].create_dataset("ring_id_2_a", data=ring_id_2_a)
            lsf_container2[smi].create_dataset("hybr_id_2_a", data=hybr_id_2_a)
            lsf_container2[smi].create_dataset("arom_id_2_a", data=arom_id_2_a)
            # lsf_container2[smi].create_dataset("charges_2_a", data=charges_2_a)
            lsf_container2[smi].create_dataset("edge_2d_2_a", data=edge_2d_2_a)
            lsf_container2[smi].create_dataset("edge_3d_2_a", data=edge_3d_2_a)
            lsf_container2[smi].create_dataset("crds_3d_2_a", data=crds_3d_2_a)
            lsf_container2[smi].create_dataset("atom_id_2_b", data=atom_id_2_b)
            lsf_container2[smi].create_dataset("ring_id_2_b", data=ring_id_2_b)
            lsf_container2[smi].create_dataset("hybr_id_2_b", data=hybr_id_2_b)
            lsf_container2[smi].create_dataset("arom_id_2_b", data=arom_id_2_b)
            # lsf_container2[smi].create_dataset("charges_2_b", data=charges_2_b)
            lsf_container2[smi].create_dataset("edge_2d_2_b", data=edge_2d_2_b)
            lsf_container2[smi].create_dataset("edge_3d_2_b", data=edge_3d_2_b)
            lsf_container2[smi].create_dataset("crds_3d_2_b", data=crds_3d_2_b)
            lsf_container2[smi].create_dataset("atom_id_2_c", data=atom_id_2_c)
            lsf_container2[smi].create_dataset("ring_id_2_c", data=ring_id_2_c)
            lsf_container2[smi].create_dataset("hybr_id_2_c", data=hybr_id_2_c)
            lsf_container2[smi].create_dataset("arom_id_2_c", data=arom_id_2_c)
            # lsf_container2[smi].create_dataset("charges_2_c", data=charges_2_c)
            lsf_container2[smi].create_dataset("edge_2d_2_c", data=edge_2d_2_c)
            lsf_container2[smi].create_dataset("edge_3d_2_c", data=edge_3d_2_c)
            lsf_container2[smi].create_dataset("crds_3d_2_c", data=crds_3d_2_c)
            lsf_container2[smi].create_dataset("atom_id_2_d", data=atom_id_2_d)
            lsf_container2[smi].create_dataset("ring_id_2_d", data=ring_id_2_d)
            lsf_container2[smi].create_dataset("hybr_id_2_d", data=hybr_id_2_d)
            lsf_container2[smi].create_dataset("arom_id_2_d", data=arom_id_2_d)
            # lsf_container2[smi].create_dataset("charges_2_d", data=charges_2_d)
            lsf_container2[smi].create_dataset("edge_2d_2_d", data=edge_2d_2_d)
            lsf_container2[smi].create_dataset("edge_3d_2_d", data=edge_3d_2_d)
            lsf_container2[smi].create_dataset("crds_3d_2_d", data=crds_3d_2_d)
            lsf_container2[smi].create_dataset("atom_id_2_e", data=atom_id_2_e)
            lsf_container2[smi].create_dataset("ring_id_2_e", data=ring_id_2_e)
            lsf_container2[smi].create_dataset("hybr_id_2_e", data=hybr_id_2_e)
            lsf_container2[smi].create_dataset("arom_id_2_e", data=arom_id_2_e)
            # lsf_container2[smi].create_dataset("charges_2_e", data=charges_2_e)
            lsf_container2[smi].create_dataset("edge_2d_2_e", data=edge_2d_2_e)
            lsf_container2[smi].create_dataset("edge_3d_2_e", data=edge_3d_2_e)
            lsf_container2[smi].create_dataset("crds_3d_2_e", data=crds_3d_2_e)

    h5f2 = h5py.File(f"data/lsf_rxn_carbacids_{name}.h5")
    print(f"Successfully transformed {len(list(h5f2.keys()))} carboxylic acids")

    print(f"\nTransforming {len(rxn_id)} reactions into h5 format")

    with h5py.File(f"data/lsf_rxn_conditions_{name}.h5", "w") as lsf_container:
        for idx, rxn_key in enumerate(tqdm(rxn_id)):
            smi1 = startingmat_1_smiles[idx]
            smi2 = startingmat_2_smiles[idx]
            rgt_eq = float(reagent_1_eq[idx])
            sm2_eq = float(startingmat_2_eq[idx])
            conc_m = float(concentration_moll[idx])

            # print(rxn_key, smi1, smi2, rgt_eq, sm2_eq, conc_m)

            ecfp4_2_1 = get_fp_from_smi(smi1)
            ecfp4_2_2 = get_fp_from_smi(smi2)

            # Create group in h5 for this ids
            lsf_container.create_group(rxn_key)

            # Molecule ECFP
            lsf_container[rxn_key].create_dataset("ecfp4_2_1", data=[ecfp4_2_1])
            lsf_container[rxn_key].create_dataset("ecfp4_2_2", data=[ecfp4_2_2])

            # Conditions
            lsf_container[rxn_key].create_dataset("rgt_eq", data=[rgt_eq])
            lsf_container[rxn_key].create_dataset("sm2_eq", data=[sm2_eq])
            lsf_container[rxn_key].create_dataset("conc_m", data=[conc_m])

            # Traget
            lsf_container[rxn_key].create_dataset("trg_yld", data=[product_all[idx]])
            lsf_container[rxn_key].create_dataset("trg_bin", data=[binary[idx]])

    h5f = h5py.File(f"data/lsf_rxn_conditions_{name}.h5")
    print(f"Successfully transformed {len(list(h5f.keys()))} conditions")
