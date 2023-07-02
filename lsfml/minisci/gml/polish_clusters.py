from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import argparse, torch, configparser, os, time
import numpy as np
import pandas as pd
from tqdm import tqdm

from lsfqml.clustering.cluster import SaveXlsxFromFrame

def filter_df(fname):

    df = pd.read_csv(f"labelled_data/{fname}.csv")

    smiles = df["smi_passed"]

    rings = []
    no_ps = []

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        nrings = rdMolDescriptors.CalcNumRings(mol)
        rings.append(nrings)

        if ("p" in smi) or ("P" in smi):
            no_ps.append(1)
        else:
            no_ps.append(0)  

    df["rings"] = rings
    df["no_ps"] = no_ps

    df = df.drop(df[df.rings <= 2].index)
    df = df.drop(df[df.no_ps >= 1].index)

    out_name = f"labelled_data/{fname}_filtered.xlsx"
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    df.to_csv(f"labelled_data/{fname}_filtered.csv", index=False,)

    SaveXlsxFromFrame(
        df,
        out_name,
        molCols=["smi_passed",],
        size=(300, 300),
    )





if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("-csv", type=str, default="labels_350_cluster_2")
    args = parser.parse_args()

    filter_df(args.csv)