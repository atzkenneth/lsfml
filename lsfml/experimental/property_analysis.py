#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (©) 2023 Kenneth Atz, & Gisbert Schneider (ETH Zurich)

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw

import pandas as pd
import numpy as np
import os, xlsxwriter
from tqdm import tqdm
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from lsfml.utils import UTILS_PATH

fontsize = 22


def get_hist_property(wgt, rot, hba, hbd, psa, rng, sp3, ste, name, bins):
    """Get histogram of molecular properties.

    Args:
        wgt (list): Molecular weight.
        rot (list): Rotatable bonds.
        hba (list): Hydrogen bond acceptors.
        hbd (list): Hydrogen bond acceptors.
        psa (list): Polar surface area.
        rng (list): Rings.
        sp3 (list): Fraction sp3.
        ste (list): Stereogenic centers.
        name (str): File name.
        bins (int): Number of bins.
    """
    fig = plt.figure(figsize=(40, 16))
    gs = GridSpec(nrows=2, ncols=4)
    gs.update(wspace=0.4, hspace=0.2)

    ax = fig.add_subplot(111)
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.tick_params(labelcolor="w", top=False, bottom=False, left=False, right=False)
    ax.set_ylabel("Number of molecules", fontsize=fontsize + 8, labelpad=60)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(wgt, density=False, facecolor="royalblue", bins=bins)
    ax1.tick_params(axis="x", labelsize=fontsize)
    ax1.tick_params(axis="y", labelsize=fontsize)
    ax1.set_xlabel(str("Molecular weight / g/mol"), fontsize=fontsize + 4)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(rot, density=False, facecolor="royalblue", bins=bins)
    ax2.tick_params(axis="x", labelsize=fontsize)
    ax2.tick_params(axis="y", labelsize=fontsize)
    ax2.set_xlabel(str("Rotatable bonds / $N$"), fontsize=fontsize + 4)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(hba, density=False, facecolor="royalblue", bins=bins)
    ax3.tick_params(axis="x", labelsize=fontsize)
    ax3.tick_params(axis="y", labelsize=fontsize)
    ax3.set_xlabel(str("Hydrogen bond acceptors / $N$"), fontsize=fontsize + 4)

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.hist(hbd, density=False, facecolor="royalblue", bins=bins)
    ax4.tick_params(axis="x", labelsize=fontsize)
    ax4.tick_params(axis="y", labelsize=fontsize)
    ax4.set_xlabel(str("Hydrogen bond donors / $N$"), fontsize=fontsize + 4)

    ax5 = fig.add_subplot(gs[1, 0])
    ax5.hist(psa, density=False, facecolor="royalblue", bins=bins)
    ax5.tick_params(axis="x", labelsize=fontsize)
    ax5.tick_params(axis="y", labelsize=fontsize)
    ax5.set_xlabel(str("Polar surface area / $A^2$"), fontsize=fontsize + 4)

    ax6 = fig.add_subplot(gs[1, 1])
    ax6.hist(rng, density=False, facecolor="royalblue", bins=bins)
    ax6.tick_params(axis="x", labelsize=fontsize)
    ax6.tick_params(axis="y", labelsize=fontsize)
    ax6.set_xlabel(str("Rings / $N$"), fontsize=fontsize + 4)

    ax7 = fig.add_subplot(gs[1, 2])
    ax7.hist(sp3, density=False, facecolor="royalblue", bins=bins)
    ax7.tick_params(axis="x", labelsize=fontsize)
    ax7.tick_params(axis="y", labelsize=fontsize)
    ax7.set_xlabel(str("Fraction sp3"), fontsize=fontsize + 4)

    ax8 = fig.add_subplot(gs[1, 3])
    ax8.hist(ste, density=False, facecolor="royalblue", bins=bins)
    ax8.tick_params(axis="x", labelsize=fontsize)
    ax8.tick_params(axis="y", labelsize=fontsize)
    ax8.set_xlabel(str("Stereogenic centers / $N$"), fontsize=fontsize + 4)

    out_name = os.path.join("analysis/figures/", name + ".png")
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    plt.savefig(out_name, dpi=200)
    plt.clf()

    return


def get_hist_temp_time(temps, times, scale, concs):
    """Get historgram of reaction conditions.

    Args:
        temps (list): Reaction temperature.
        times (list): Reaction time.
        scale (list): Reaction scale.
        concs (list): Reaction concentration.
    """
    fig = plt.figure(figsize=(40, 10))
    gs = GridSpec(nrows=1, ncols=4)
    gs.update(wspace=0.4, hspace=0.2)

    ax = fig.add_subplot(111)
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.tick_params(labelcolor="w", top=False, bottom=False, left=False, right=False)
    ax.set_ylabel("Number of reactions", fontsize=fontsize + 8, labelpad=60)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(temps, density=False, facecolor="royalblue", bins=30)
    ax1.tick_params(axis="x", labelsize=fontsize)
    ax1.tick_params(axis="y", labelsize=fontsize)
    ax1.set_xlabel(str("Temperarure / Celsius"), fontsize=fontsize + 4)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(times, density=False, facecolor="royalblue", bins=30)
    ax2.tick_params(axis="x", labelsize=fontsize)
    ax2.tick_params(axis="y", labelsize=fontsize)
    ax2.set_xlabel(str("Time / hour"), fontsize=fontsize + 4)

    scale = [x * 1000 for x in scale]
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(scale, density=False, facecolor="royalblue", bins=30)
    ax3.tick_params(axis="x", labelsize=fontsize)
    ax3.tick_params(axis="y", labelsize=fontsize)
    ax3.set_xlabel(str("Scale / mol"), fontsize=fontsize + 4)

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.hist(concs, density=False, facecolor="royalblue", bins=30)
    ax4.tick_params(axis="x", labelsize=fontsize)
    ax4.tick_params(axis="y", labelsize=fontsize)
    ax4.set_xlabel(str("Concentration / mol/L"), fontsize=fontsize + 4)

    out_name = os.path.join("analysis/figures/histogram_non_molecular_conditions.png")
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    plt.savefig(out_name, dpi=300)
    plt.clf()

    return


def get_propertiest(smiles):
    """Calculating molecular properties from a list of SMILES.

    Args:
        smiles (list): SMILES strings

    Returns:
        lists: Molecular properties.
    """
    wgt = []
    rot = []
    hba = []
    hbd = []
    psa = []
    rng = []
    sp3 = []
    ste = []

    for i, smi in enumerate(tqdm(smiles)):
        try:
            mol = Chem.MolFromSmiles(smi)
            wgt.append(rdMolDescriptors.CalcExactMolWt(mol))
            rot.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
            hba.append(rdMolDescriptors.CalcNumHBA(mol))
            hbd.append(rdMolDescriptors.CalcNumHBD(mol))
            psa.append(rdMolDescriptors.CalcTPSA(mol))
            rng.append(rdMolDescriptors.CalcNumRings(mol))
            sp3.append(rdMolDescriptors.CalcFractionCSP3(mol))
            ste.append(rdMolDescriptors.CalcNumAtomStereoCenters(mol))
        except:
            pass

    return wgt, rot, hba, hbd, psa, rng, sp3, ste


def get_csv_summary(df, key, with_mol_img=False):
    """Save csv from data frame.

    Args:
        df (pandas data frame): Data frame consisting of SMILES strings and their IDs.
        key (str): key to access SMILES strings.
        with_mol_img (bool, optional): Save mol img to xlsx file. Defaults to False.
    """
    smiles_set = list(set(df[key]))
    smiles_list = list(df[key])

    summary_dict = {}

    for smi in smiles_set:
        summary_dict[smi] = []

    yes_no = list(df["yes_no"])

    for idx, x in enumerate(smiles_list):
        summary_dict[x].append(yes_no[idx])

    smls = []
    wins = []
    loss = []
    totl = []

    for x in summary_dict:
        smls.append(x)
        wins.append(summary_dict[x].count(1))
        loss.append(summary_dict[x].count(0))
        totl.append(summary_dict[x].count(0) + summary_dict[x].count(1))

    df_tmp = pd.DataFrame(
        {
            "smiles": smls,
            "num_reactions_worked": wins,
            "num_reactions_failed": loss,
            "num_reactions_total": totl,
        }
    )

    df_tmp.sort_values(by="num_reactions_worked", ascending=False, inplace=True, ignore_index=True)

    out_name = "analysis/summary/summary_" + str(key) + ".xlsx"
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    df_tmp.to_csv(
        "analysis/summary/summary_" + str(key) + ".csv",
        index=False,
    )

    if with_mol_img:
        SaveXlsxFromFrame(
            df_tmp,
            out_name,
            molCols=[
                "smiles",
            ],
            size=(300, 300),
        )

    return


def SaveXlsxFromFrame(frame, outFile, molCols=["ROMol"], size=(300, 300)):
    """Generating xlsx file with drawings.

    Args:
        frame (pandas data frame): Data frame consisting of SMILES strings and their IDs and other properties.
        outFile (str): Name of the files saved.
        molCols (list, optional): Columns from which SMILES are saved as drwings. Defaults to ["ROMol"].
        size (tuple, optional): Size of the drawings. Defaults to (300, 300).
    """
    cols = list(frame.columns)

    dataTypes = dict(frame.dtypes)

    workbook = xlsxwriter.Workbook(outFile)  # New workbook
    worksheet = workbook.add_worksheet()  # New work sheet
    worksheet.set_column("A:A", size[0] / 6.0)  # column width

    # Write first row with column names
    c2 = 0
    molCol_names = [f"{x}_img" for x in molCols]
    for x in molCol_names + cols:
        worksheet.write_string(0, c2, x)
        c2 += 1

    c = 1
    for _, row in tqdm(frame.iterrows(), total=len(frame)):
        for k, molCol in enumerate(molCols):
            image_data = BytesIO()

            # none can not be visualized as molecule
            if row[molCol] == "none":
                pass
            else:
                img = Draw.MolToImage(Chem.MolFromSmiles(row[molCol]), size=size)
                img.save(image_data, format="PNG")
                worksheet.set_row(c, height=size[1])  # looks like height is not in px?
                worksheet.insert_image(c, k, "f", {"image_data": image_data})

        c2 = len(molCols)
        for x in cols:
            if str(dataTypes[x]) == "object":
                # string length is limited in xlsx
                worksheet.write_string(c, c2, str(row[x])[:32000])
            elif ("float" in str(dataTypes[x])) or ("int" in str(dataTypes[x])):
                if (row[x] != np.nan) or (row[x] != np.inf):
                    worksheet.write_number(c, c2, row[x])
            elif "datetime" in str(dataTypes[x]):
                worksheet.write_datetime(c, c2, row[x])
            c2 += 1
        c += 1

    workbook.close()
    image_data.close()


def get_hist_equiv(ctls_eq, lgnd_eq, rgnt_eq):
    """Plot hoistogram.

    Args:
        ctls_eq (str): Equivalents of Catalyst.
        lgnd_eq (str): Equivalents of Ligand.
        rgnt_eq (str): Equivalents of Reagent.
    """
    fig = plt.figure(figsize=(30, 14))
    gs = GridSpec(nrows=1, ncols=3)
    gs.update(wspace=0.4, hspace=0.2)

    ax = fig.add_subplot(111)
    ax.spines["top"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.tick_params(labelcolor="w", top=False, bottom=False, left=False, right=False)
    ax.set_xlabel("Equivalents / %", fontsize=fontsize + 8, labelpad=60)
    ax.set_ylabel("Number of reactions", fontsize=fontsize + 8, labelpad=60)

    ctls_eq = [x * 100 for x in ctls_eq]
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(ctls_eq, density=False, facecolor="royalblue", bins=30)
    ax1.tick_params(axis="x", labelsize=fontsize)
    ax1.tick_params(axis="y", labelsize=fontsize)
    ax1.set_xlabel(str("Catalyst"), fontsize=fontsize + 4)

    lgnd_eq = [x * 100 for x in lgnd_eq]
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(lgnd_eq, density=False, facecolor="royalblue", bins=30)
    ax2.tick_params(axis="x", labelsize=fontsize)
    ax2.tick_params(axis="y", labelsize=fontsize)
    ax2.set_xlabel(str("Ligand"), fontsize=fontsize + 4)

    rgnt_eq = [x * 100 for x in rgnt_eq]
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(rgnt_eq, density=False, facecolor="royalblue", bins=30)
    ax3.tick_params(axis="x", labelsize=fontsize)
    ax3.tick_params(axis="y", labelsize=fontsize)
    ax3.set_xlabel(str("Reagent"), fontsize=fontsize + 4)

    out_name = os.path.join("analysis/figures/histogram_equivalents.png")
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    plt.savefig(out_name, dpi=300)
    plt.clf()

    return


def yield_hist(yields, name):
    """Plot histrogram of reaction yields.

    Args:
        yields (list): List of reaction yields.
        name (str): Name of output file.

    Yields:
        _type_: _description_
    """
    plt.figure(figsize=(8, 8))
    yields = [float(x) * 100 for x in yields]
    plt.hist(yields, density=False, color="royalblue", bins=20)
    plt.ylabel("Occurrence", fontsize=18)
    plt.xlabel("Reaction yield / %", fontsize=18)
    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)
    plt.savefig(name, dpi=600)
    plt.show()
    plt.clf()

    return


if __name__ == "__main__":
    # read csv
    df = pd.read_csv(os.path.join(UTILS_PATH, "data/experimental_rxndata.csv"))
    smiles = list(set(df["educt"]))
    print(f"Number of different educts: {len(list(set(smiles)))}")

    # gets all properties from smiles list
    print("Calculating property distribution and generating histogram:")
    wgt, rot, hba, hbd, psa, rng, sp3, ste = get_propertiest(smiles)

    # plot histogram of properteies from all smiles
    get_hist_property(wgt, rot, hba, hbd, psa, rng, sp3, ste, "property_all", bins=15)

    # get csv summaries
    print("Generating csv and xls files:")
    get_csv_summary(df, "educt", with_mol_img=True)
    get_csv_summary(df, "catalyst", with_mol_img=False)
    get_csv_summary(df, "reagent", with_mol_img=True)
    get_csv_summary(df, "solvent", with_mol_img=True)
    get_csv_summary(df, "ligand", with_mol_img=True)
    get_csv_summary(df, "rxn_atm", with_mol_img=False)

    # get histogram of time/temp
    print("Generating rxn histogram:")
    temps = list(df["rxn_temp_C"])
    times = list(df["rxn_time_h"])
    scale = list(df["rxn_scale_mol"])
    concs = list(df["rxn_c_moll"])
    get_hist_temp_time(temps, times, scale, concs)

    # get histogram of equivalents
    ctls_eq = list(df["catalyst_eq"])
    lgnd_eq = list(df["ligand_eq"])
    rgnt_eq = list(df["reagent_eq"])
    get_hist_equiv(ctls_eq, lgnd_eq, rgnt_eq)

    # yield hist
    yields = [1 - x for x in list(df["non_bo"])]
    yield_hist(yields, "analysis/figures/hist_rxn_yield.png")

    yields = [x for x in yields if x > 0]
    yield_hist(yields, "analysis/figures/hist_rxn_yield_pos.png")

    print("All Done!")
