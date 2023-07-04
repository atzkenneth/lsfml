#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (©) 2023 Kenneth Atz, & Gisbert Schneider (ETH Zurich)

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ertl import identify_functional_groups
from rdkit import Chem
from tqdm import tqdm

PARAMS = [2, 40, "fg_analysis/"]
FOLDER_NAME = PARAMS[2]
os.makedirs(FOLDER_NAME, exist_ok=True)


def get_smiles_2_rxn_dict(smiles, yes_no):
    """Creates a dict for SMILES-strings to sum of successful reactions.

    :param smiles: SMILES-strings
    :type smiles: list[str]
    :param yes_no: Reaction outcomes.
    :type yes_no: list[int]
    :return: Dict SMILES-strings (str): sum of successful reactions (int).
    :rtype: dict
    """
    smiles_2_rxn_dict = {}

    for i, smi in enumerate(tqdm(smiles)):
        if smi not in smiles_2_rxn_dict:
            smiles_2_rxn_dict[smi] = 0
            smiles_2_rxn_dict[smi] += yes_no[i]

        elif smi in smiles_2_rxn_dict:
            smiles_2_rxn_dict[smi] += yes_no[i]

    return smiles_2_rxn_dict


def get_smiles_2_negative_rxn_dict(smiles, yes_no):
    """Creates a dict for SMILES-strings to sum of failed reactions.

    :param smiles: SMILES-strings
    :type smiles: list[str]
    :param yes_no: Reaction outcomes.
    :type yes_no: list[int]
    :return: Dict SMILES-strings (str): sum of failed reactions (int).
    :rtype: dict
    """
    smiles_2_rxn_dict = {}

    for i, smi in enumerate(tqdm(smiles)):
        failure = abs(yes_no[i] - 1)

        if smi not in smiles_2_rxn_dict:
            smiles_2_rxn_dict[smi] = 0
            smiles_2_rxn_dict[smi] += failure

        elif smi in smiles_2_rxn_dict:
            smiles_2_rxn_dict[smi] += failure

    return smiles_2_rxn_dict


def plt_barplot(sorted_model_dict, model_dict_std, name, ylabel, xlabel, ylim, imgsize1, imgsize2):
    """Creates a bar plot given a list of mean values and standard deviations.

    :param sorted_model_dict: Dict for mean values.
    :type sorted_model_dict: dict
    :param model_dict_std: Dict for standard deviations.
    :type model_dict_std: dict
    :param name: File name.
    :type name: str
    :param ylabel: Y-axis label.
    :type ylabel: str
    :param xlabel: X-axis label.
    :type xlabel: str
    :param ylim: Y-axis limit.
    :type ylim: boolean
    :param imgsize1: Figure size 1.
    :type imgsize1: int
    :param imgsize2: Figure size 2.
    :type imgsize2: int
    """
    plt.figure(figsize=(imgsize1, imgsize2))

    keys = list(sorted_model_dict.keys())
    accs = []
    stds = []

    for k in sorted_model_dict:
        accs.append(np.mean(np.array(sorted_model_dict[k])))
        stds.append(np.mean(np.array(model_dict_std[k])))

    plt.bar(keys, accs, color="lightskyblue")  # grey
    plt.errorbar(keys, accs, yerr=stds, color="black", ls="none", elinewidth=2.5)
    plt.tick_params(axis="x", labelsize=30, rotation=90)
    plt.tick_params(axis="y", labelsize=34)
    plt.xlabel(f"\n{xlabel}", fontsize=34)
    plt.ylabel(f"\n{ylabel}", fontsize=34)

    if ylim:
        bottom, top = plt.xlim()
        plt.ylim((min(accs) - 10, max(accs) + 3))
    plt.tight_layout()
    plt.savefig(f"{FOLDER_NAME}{name}.png", dpi=400)
    plt.clf()


def plt_barplot_two_colors(
    sorted_model_dict, model_dict_std, name, ylabel, xlabel, ylim, imgsize1, imgsize2, dict_to_check
):
    """Creates a bar plot given a list of mean values and standard deviations. Additionally, specific bars can be colored in a second color.

    :param sorted_model_dict: Dict for mean values.
    :type sorted_model_dict: dict
    :param model_dict_std: Dict for standard deviations.
    :type model_dict_std: dict
    :param name: File name.
    :type name: str
    :param ylabel: Y-axis label.
    :type ylabel: str
    :param xlabel: X-axis label.
    :type xlabel: str
    :param ylim: Y-axis limit.
    :type ylim: boolean
    :param imgsize1: Figure size 1.
    :type imgsize1: int
    :param imgsize2: Figure size 2.
    :type imgsize2: int
    :param dict_to_check: Name of keys for second color.
    :type dict_to_check: str
    """
    plt.figure(figsize=(imgsize1, imgsize2))

    keys = list(sorted_model_dict.keys())
    accs = []
    stds = []

    for k in sorted_model_dict:
        accs.append(np.mean(np.array(sorted_model_dict[k])))
        stds.append(np.mean(np.array(model_dict_std[k])))

    barlist = plt.bar(keys, accs, color="lightskyblue")

    for (
        i,
        k,
    ) in enumerate(keys):
        if k not in dict_to_check:
            barlist[i].set_color("orange")

    plt.errorbar(keys, accs, yerr=stds, color="black", ls="none", elinewidth=2.5)
    plt.tick_params(axis="x", labelsize=30, rotation=90)
    plt.tick_params(axis="y", labelsize=34)
    plt.xlabel(f"\n{xlabel}", fontsize=34)
    plt.ylabel(f"\n{ylabel}", fontsize=34)

    if ylim:
        bottom, top = plt.xlim()
        plt.ylim((min(accs) - 10, max(accs) + 3))
    plt.tight_layout()
    plt.savefig(f"{FOLDER_NAME}{name}.png", dpi=400)
    plt.clf()


def plt_barplot_with_adapt_color(
    sorted_model_dict, model_dict_std, name, ylabel, xlabel, ylim, imgsize1, imgsize2, bar_color
):
    """Creates a bar plot given a list of mean values and standard deviations. Additionally, the color can be specified.

    :param sorted_model_dict: Dict for mean values.
    :type sorted_model_dict: dict
    :param model_dict_std: Dict for standard deviations.
    :type model_dict_std: dict
    :param name: File name.
    :type name: str
    :param ylabel: Y-axis label.
    :type ylabel: str
    :param xlabel: X-axis label.
    :type xlabel: str
    :param ylim: Y-axis limit.
    :type ylim: boolean
    :param imgsize1: Figure size 1.
    :type imgsize1: int
    :param imgsize2: Figure size 2.
    :type imgsize2: int
    :param bar_color: Color.
    :type bar_color: str
    """
    plt.figure(figsize=(imgsize1, imgsize2))

    keys = list(sorted_model_dict.keys())

    accs = []
    stds = []

    for k in sorted_model_dict:
        accs.append(np.mean(np.array(sorted_model_dict[k])))
        stds.append(np.mean(np.array(model_dict_std[k])))

    plt.errorbar(keys, accs, yerr=stds, color="black", ls="none", elinewidth=2.5)
    plt.tick_params(axis="x", labelsize=30, rotation=90)
    plt.tick_params(axis="y", labelsize=34)
    plt.xlabel(f"\n{xlabel}", fontsize=34)
    plt.ylabel(f"\n{ylabel}", fontsize=34)

    if ylim:
        bottom, top = plt.xlim()
        plt.ylim((min(accs) - 10, max(accs) + 3))
    plt.tight_layout()
    plt.savefig(f"{FOLDER_NAME}{name}.png", dpi=400)
    plt.clf()


def get_fg_occurence(smiles):
    """Counts successful reactions per functional group in a dict of SMILES-srings to reaction outcome.

    :param smiles: SMILES-strings (str): reaction outcome {int}
    :type smiles: dict
    :return: dict functional group (str): number of successful reactions (int).
    :rtype: dict
    """
    fg_dict = {}

    uniques = list(set(smiles))

    for u in uniques:
        try:
            m = Chem.MolFromSmiles(u)
            fgs = identify_functional_groups(m)

            tmp_fgs = []

            for f in fgs:
                tmp_fgs.append(f[PARAMS[0]])

            for fg in tmp_fgs:
                if fg not in fg_dict:
                    fg_dict[fg] = 1

                elif fg in fg_dict:
                    fg_dict[fg] += 1
        except:
            print("skipping:", u)

    return fg_dict


def get_fg_tollerance(smiles_2_rxn_dict):
    """Counts functional groups in a list of SMILES-srings.

    :param smiles: SMILES-strings
    :type smiles: list[str]
    :return: dict functional group (str): number of successful reactions (int).
    :rtype: dict
    """
    fg_tollerance_dict = {}

    for smi in tqdm(smiles_2_rxn_dict):
        rxn = smiles_2_rxn_dict[smi]

        try:
            m = Chem.MolFromSmiles(smi)
            fgs = identify_functional_groups(m)

            tmp_fgs = []

            for f in fgs:
                tmp_fgs.append(f[PARAMS[0]])

            for fg in tmp_fgs:
                if fg not in fg_tollerance_dict:
                    fg_tollerance_dict[fg] = rxn

                elif fg in fg_tollerance_dict:
                    fg_tollerance_dict[fg] += rxn

        except:
            print("skipping:", smi)

    return fg_tollerance_dict


if __name__ == "__main__":
    # Smiles from lsf-space
    df = pd.read_csv("../data/experimental_rxndata.csv")
    smiles = list(df["educt"])
    yes_no = list(df["yes_no"])
    yes_no = [float(x) for x in yes_no]

    # Get dicts for absoluet and relative success/failure rate of the individual FGs
    print(
        f"\nExtracting all functional groups from the substrates present in the {len(smiles)} "
        "reactions of the experimental data set:"
    )
    smiles_2_rxn_dict = get_smiles_2_rxn_dict(smiles, yes_no)
    smiles_2_negative_rxn_dict = get_smiles_2_negative_rxn_dict(smiles, yes_no)

    print(
        f"\nCalculating the relative success/failure rate of the {len(smiles_2_rxn_dict)} individual "
        "functioan groups:"
    )
    fg_tollerance_dict = get_fg_tollerance(smiles_2_rxn_dict)
    fg_intollerance_dict = get_fg_tollerance(smiles_2_negative_rxn_dict)

    print("\nCreating the five plots:")

    # Green barplot
    sorted_fgs = sorted(fg_tollerance_dict, key=lambda k: fg_tollerance_dict[k], reverse=True)
    sorted_fg_dict = {}
    model_dict_std = {}
    for k in sorted_fgs:
        sorted_fg_dict[k] = fg_tollerance_dict[k]
        model_dict_std[k] = 0

    plt_barplot_with_adapt_color(
        sorted_fg_dict,
        model_dict_std,
        "lsf_space_success",
        "Successful reactions / $N$",
        "Functional group / SMILES",
        None,
        22,
        14,
        "palegreen",
    )
    print(
        f"1. Plotted the functional groups for the unique {len(smiles_2_rxn_dict)} substrates by success "
        "(absolute number)."
    )

    # Red barplot
    sorted_fgs = sorted(fg_intollerance_dict, key=lambda k: fg_intollerance_dict[k], reverse=True)
    sorted_fg_dict = {}
    model_dict_std = {}
    for k in sorted_fgs:
        sorted_fg_dict[k] = fg_intollerance_dict[k]
        model_dict_std[k] = 0

    plt_barplot_with_adapt_color(
        sorted_fg_dict,
        model_dict_std,
        "lsf_space_failure",
        "Failed reactions / $N$",
        "Functional group / SMILES",
        None,
        22,
        14,
        "lightcoral",
    )
    print(
        f"2. Plotted the functional groups for the unique {len(smiles_2_rxn_dict)} substrates by failure "
        "(absolute number)."
    )

    # Barplot of LSF-space
    fg_dict = get_fg_occurence(smiles)
    sorted_fgs = sorted(fg_dict, key=lambda k: fg_dict[k], reverse=True)
    sorted_fg_dict = {}
    model_dict_std = {}
    for k in sorted_fgs:
        sorted_fg_dict[k] = fg_dict[k]
        model_dict_std[k] = 0

    plt_barplot(
        sorted_fg_dict,
        model_dict_std,
        "lsf_space_number",
        "Occurence in LSF-space library / $N$",
        "Functional group / SMILES",
        None,
        22,
        14,
    )
    dict_to_cherck = sorted_fg_dict
    print(
        f"3. Plotted the {len(list(fg_dict.keys()))} unique functional groups for the {len(smiles_2_rxn_dict)} "
        "substrates by occurence (absolute number)."
    )

    # Bar plot percentage of failed
    percent_dict = {}

    for fg in fg_dict:
        percent_dict[fg] = fg_intollerance_dict[fg] / (fg_dict[fg] * 24)

    sorted_fgs = sorted(percent_dict, key=lambda k: percent_dict[k], reverse=True)
    sorted_fg_dict = {}
    model_dict_std = {}
    for k in sorted_fgs:
        sorted_fg_dict[k] = percent_dict[k]
        model_dict_std[k] = 0

    plt_barplot_with_adapt_color(
        sorted_fg_dict,
        model_dict_std,
        "lsf_space_failed_fraction",
        "Fraction of failed reactions",
        "Functional group / SMILES",
        None,
        22,
        14,
        "lightskyblue",
    )
    print(
        f"4. Plotted the functional groups from the unique {len(smiles_2_rxn_dict)} substrates by "
        "failure (relative number)."
    )

    # Barplot of Drug-space
    drug_data = pd.read_csv("../clustering/cluster_analysis/filtered_list/filtered_list.csv")
    smiles = list(drug_data["smiles_list"])
    fg_dict = get_fg_occurence(smiles)

    occurences = []
    for k in fg_dict:
        occurences.append(fg_dict[k])
    top_100 = sorted(occurences)[-PARAMS[1]]

    sorted_fgs = sorted(fg_dict, key=lambda k: fg_dict[k], reverse=True)
    sorted_fg_dict = {}
    model_dict_std = {}
    for k in sorted_fgs:
        if fg_dict[k] >= top_100:
            sorted_fg_dict[k] = fg_dict[k]
            model_dict_std[k] = 0

    plt_barplot_two_colors(
        sorted_fg_dict,
        model_dict_std,
        "drug_lsf_space_comparison",
        "Occurence in drug-space library / $N$",
        "Functional group / SMILES",
        None,
        22,
        14,
        dict_to_cherck,
    )
    print(
        f"5. Plotted the {len(sorted_fg_dict)} / {len(fg_dict)} most abundant functional groups of the "
        "drug space library and highlighted the once not present in the LSF space library in orange."
    )
