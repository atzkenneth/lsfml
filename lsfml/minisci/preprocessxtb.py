from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem

import os
import pandas as pd
from collections import Counter
from tqdm import tqdm
import numpy as np

def get_dict_for_embedding(list):
    """Creates a dict x: 0 - x: N for x in List with len(list(set(List))) = N"""
    list_dict = {}
    list_counter = Counter(list)
    for idx, x in enumerate(list_counter):
        list_dict[x] = idx
    return list_dict

if __name__ == '__main__':

    df = pd.read_csv("data/20221113_alkylation_rxndata.csv", sep=',', encoding='unicode_escape')

    # read csv id
    rxn_id  = list(df["rxn_id"])  
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

    rea_dict = get_dict_for_embedding(reagent_1_smiles)
    so1_dict = get_dict_for_embedding(solvent_1_smiles)
    so2_dict = get_dict_for_embedding(solvent_2_smiles)
    print("Distinct reagent and solvent types (should be one each): ", list(rea_dict.keys()), list(so1_dict.keys()), list(so2_dict.keys()))
    product_all = [float(product_mono[i] + product_di[i]) for i, x in enumerate(product_mono)]

    # Get the rxn keys for the two starting materials
    rxn_smi1_dict = {}
    for i, rxn in enumerate(rxn_id):

        short_rxn = rxn_id[i].split("-")[1]
        short_rxn = short_rxn.split("_")[0]

        if short_rxn not in rxn_smi1_dict:
            rxn_smi1_dict[short_rxn] = startingmat_1_smiles[i]
        else:
            pass

    print(f"Number of substrates: {len(rxn_smi1_dict.keys())}")

    rxn_smi2_dict = {}
    for i, rxn in enumerate(rxn_id):

        short_rxn = rxn_id[i].split("_")[-1]

        if short_rxn not in rxn_smi2_dict:
            rxn_smi2_dict[short_rxn] = startingmat_2_smiles[i]
        else:
            pass

    print(f"Number of carboxylic acids: {len(rxn_smi2_dict.keys())}")

    # Get 3D graph data
    print(f"\nGenerating 3D conformers and QM predictions for {len(rxn_smi1_dict.keys())} substrates and saving them:")
    randomSeeds = [0xf00a, 0xf00b, 0xf00c, 0xf00d, 0xf00e]

    os.makedirs("confomers/", exist_ok=True)

    for rxn in rxn_smi1_dict:
        for randomSeed in randomSeeds:
            os.makedirs(f"confomers/{rxn}_{randomSeed}/", exist_ok=True)
            smi = rxn_smi1_dict[rxn]
            mol_no_Hs = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol_no_Hs)
            AllChem.EmbedMolecule(mol, randomSeed) # 0xf00d
            AllChem.UFFOptimizeMolecule(mol)
            w = Chem.SDWriter(f"confomers/{rxn}_{randomSeed}/{rxn}_{randomSeed}.sdf")
            w.write(mol)
            w.close()
