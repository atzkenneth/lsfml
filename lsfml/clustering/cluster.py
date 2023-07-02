#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (©) 2023 Kenneth Atz (ETH Zurich)

import argparse
import os
from io import BytesIO

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlsxwriter
from matplotlib.gridspec import GridSpec
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdMolDescriptors
from rdkit.Chem.rdchem import Mol
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from tqdm import tqdm

fontsize = 22


def neutralize_atoms(mol: Mol) -> Mol:
    """Neutralizes a given SMILES-string.

    :param mol: RDKit molecule object
    :type mol: rdkit.Chem.rdchem.Mol
    :return: RDKit molecule object
    :rtype: rdkit.Chem.rdchem.Mol
    """
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


def get_hist(fname, wgt, rot, hba, hbd, psa, rng, sp3, ste, name, bins):
    """Creates a histogram plot for given molecular properties.

    :param fname: File name.
    :param wgt: Molecular weight.
    :param rot: Number of rotatable bonds.
    :param hba: Number of hydrogen bond acceptors.
    :param hbd: Number of hydrogen bond donors.
    :param psa: Polar surface areas.
    :param rng: Number of rings
    :param sp3: Fraction of sp3 centers.
    :param ste: Number of stereogenic centers.
    :param name: Name used to save the plot. 
    :param bins: Number of bins for each histogram. 
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
    ax1.hist(wgt, density=False, facecolor="royalblue", bins=30)
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
    ax5.hist(psa, density=False, facecolor="royalblue", bins=30)
    ax5.tick_params(axis="x", labelsize=fontsize)
    ax5.tick_params(axis="y", labelsize=fontsize)
    ax5.set_xlabel(str("Polar surface area / $A^2$"), fontsize=fontsize + 4)

    ax6 = fig.add_subplot(gs[1, 1])
    ax6.hist(rng, density=False, facecolor="royalblue", bins=bins)
    ax6.tick_params(axis="x", labelsize=fontsize)
    ax6.tick_params(axis="y", labelsize=fontsize)
    ax6.set_xlabel(str("Rings / $N$"), fontsize=fontsize + 4)

    ax7 = fig.add_subplot(gs[1, 2])
    ax7.hist(sp3, density=False, facecolor="royalblue", bins=30)
    ax7.tick_params(axis="x", labelsize=fontsize)
    ax7.tick_params(axis="y", labelsize=fontsize)
    ax7.set_xlabel(str("Fraction sp3"), fontsize=fontsize + 4)

    ax8 = fig.add_subplot(gs[1, 3])
    ax8.hist(ste, density=False, facecolor="royalblue", bins=bins)
    ax8.tick_params(axis="x", labelsize=fontsize)
    ax8.tick_params(axis="y", labelsize=fontsize)
    ax8.set_xlabel(str("Stereogenic centers / $N$"), fontsize=fontsize + 4)

    out_name = os.path.join(f"{fname}/figures/", name + ".png")
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    plt.savefig(out_name, dpi=200)
    plt.clf()

    return


def get_propertiest(smiles):
    """Calculates molecular properties given a list of SMILES-stings.

    :param smiles: SMILES-sting.
    :return: List of molecular properties. 
    :rtype: tuple(lists)
    """

    wgt, rot, hba, hbd, psa, rng, sp3, ste = [], [], [], [], [], [], [], []

    for smi in smiles:
        try:
            smi = smi.split(".")[0]
            mol = Chem.MolFromSmiles(smi)
            neutralize_atoms(mol)
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


def get_filtered_smiles_and_fps(smiles, drugid, drname, max_mwt, min_mwt):
    """Filters SMILES-list based on different filter criteria (e.g. molecular weight). 

    :param smiles: SMILES-strings.
    :param drugid: ID of SMILES-strings.
    :param drname: Name of SMILES-strings.
    :param max_mwt: Upper molecular weight cut off. 
    :param min_mwt: Lower molecular weight cut off. 
    :return: SMILES-strings and their 2 IDs as lists. 
    :rtype: tuple(lists)
    """

    smi_passed, drug_id, fps, drug_name = [], [], [], []

    for i, smi in enumerate(smiles):
        try:
            smi = smi.split(".")[0]
            mol = Chem.MolFromSmiles(smi)
            neutralize_atoms(mol)
            mwt = rdMolDescriptors.CalcExactMolWt(mol)
            if (mwt <= max_mwt) and (mwt >= min_mwt):
                fps.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=256))
                smi_passed.append(smi)
                drug_id.append(drugid[i])
                drug_name.append(drname[i])
        except:
            pass
    return fps, drug_id, drug_name, smi_passed


def get_similarity_matrix(fps):
    """Creates a similarity matrix given a list of fingerprint verctors. 

    :param fps: List of fingerprint verctors.
    :return: Similarity matrix. 
    :rtype: np.ndarray
    """

    nfps = len(fps)
    matrix = []

    for i in range(0, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        matrix.append(sims)

    return np.array(matrix)


def get_clusters(matrix, n_clusters):
    """Creates clusters given a similarity matrix.

    :param matrix: Similarity matrix.
    :param n_clusters: np.ndarray
    :return: Cluster labels.
    :rtype: list[int]
    """

    clustering = AgglomerativeClustering(linkage="ward", n_clusters=n_clusters)
    clustering.fit(matrix)

    return list(clustering.labels_)


def get_coloured_pca(fname, matrix, labels, name, two_d=False, pointsize=30):
    """Creates colored principal component analysis (PCA) plot given a similarity matrix and cluster labels. 

    :param fname: File name. 
    :param matrix: Similarity matrix.
    :param labels: Cluster labels.
    :param name: Name to save the plot.
    :param two_d: Option to plot clusters for PC1 and PC2 only , defaults to False
    :type two_d: bool, optional
    :param pointsize: Point size in the scatter plot, defaults to 10
    :type pointsize: int, optional
    """

    pca = PCA(n_components=2)
    pca.fit(matrix)
    expl_variance = pca.explained_variance_ratio_
    pca2 = pca.transform(matrix)
    pca2 = np.array(pca2)

    if two_d:
        labels = get_clusters(pca2, 8)

    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)
    bounds = np.linspace(0, 8, 9)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fontsize = 15

    plt.figure(figsize=(9, 9))
    plt.scatter(
        pca2[:, 0],
        pca2[:, 1],
        c=labels,
        s=pointsize,
        cmap=cmap,
        norm=norm,
        label=f"Expl. variance: {str(expl_variance[0] * 100)[:4]} %\nand {str(expl_variance[1] * 100)[:3]} %",
    )
    plt.legend(
        loc="best",
        fontsize=fontsize + 5,
        handletextpad=0,
        handlelength=0,
        markerscale=0,
    )
    plt.tick_params(axis="x", labelsize=fontsize + 5)
    plt.tick_params(axis="y", labelsize=fontsize + 5)
    plt.xlabel(str("Principal component 1"), fontsize=fontsize + 5)
    plt.ylabel(str("Principal component 2"), fontsize=fontsize + 5)

    out_name = os.path.join(f"{fname}/figures/{name}")
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    plt.savefig(f"{out_name}.png", dpi=400)
    plt.savefig(f"{out_name}_tp.png", dpi=400, transparent=True)
    plt.savefig(f"{out_name}.pdf", dpi=400)
    plt.clf()


def SaveXlsxFromFrame(frame, outFile, molCols=["ROMol"], size=(300, 300)):
    """Saves a csv file as xlsx file with molecules as figures. 

    :param frame: Data frame.
    :param outFile: Name of output file. 
    :param molCols: Column where the molecules can be found (can be >1), defaults to ["ROMol"]
    :type molCols: list, optional
    :param size: Size of the Figures added, defaults to (300, 300)
    :type size: tuple, optional
    """

    cols = list(frame.columns)

    dataTypes = dict(frame.dtypes)

    workbook = xlsxwriter.Workbook(outFile)
    worksheet = workbook.add_worksheet()
    worksheet.set_column("A:A", size[0] / 6.0)

    c2 = 0
    molCol_names = [f"{x}_img" for x in molCols]
    for x in molCol_names + cols:
        worksheet.write_string(0, c2, x)
        c2 += 1

    c = 1
    for _, row in tqdm(frame.iterrows(), total=len(frame)):
        for k, molCol in enumerate(molCols):
            image_data = BytesIO()
            img = Draw.MolToImage(Chem.MolFromSmiles(row[molCol]), size=size)
            img.save(image_data, format="PNG")

            worksheet.set_row(c, height=size[1])
            worksheet.insert_image(c, k, "f", {"image_data": image_data})

        c2 = len(molCols)
        for x in cols:
            if str(dataTypes[x]) == "object":
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


def save_clusters_to_csv_cosine(fname, smi_passed, drug_id, drug_name, matrix, num_clusters, name_1, name_2):
    """Saves clusters to a csv file where the molecules are sorted by their cosine distance to the centroid. 

    :param fname: File name. 
    :param smi_passed: List of SMILES-string. 
    :param drug_id: List of IDs of SMILES-stirngs. 
    :param drug_name: List of names of SMILES-stirngs. 
    :param matrix: Similarity matrix.
    :param num_clusters: Number of clusters.
    :param name_1: Column name used for IDs.
    :param name_2: Column name used for names.
    """

    for number in range(num_clusters):

        print(f"Ranking cluster {number} by cosine distance to cluster centroid:")

        smiles_cluster, drugid_cluster, drname_cluster, matrix_cluster = [], [], [], []

        for i, x in enumerate(labels):
            if x == number:
                smiles_cluster.append(smi_passed[i])
                drugid_cluster.append(drug_id[i])
                drname_cluster.append(drug_name[i])
                matrix_cluster.append(matrix[i])

        matrix_cluster = np.array(matrix_cluster)
        centroid = matrix_cluster.mean(axis=0)

        distances = [cosine(array, centroid) for array in matrix_cluster]

        wgt, rot, hba, hbd, psa, rng, sp3, ste = get_propertiest(smiles_cluster)
        get_hist(
            fname,
            wgt,
            rot,
            hba,
            hbd,
            psa,
            rng,
            sp3,
            ste,
            "property_cluster" + str(number),
            bins=10,
        )

        df = pd.DataFrame(
            {
                "smiles_list": smiles_cluster,
                name_1: drugid_cluster,
                name_2: drname_cluster,
                "distance_to_cluster_centriod": distances,
            }
        )

        df.sort_values(
            by="distance_to_cluster_centriod",
            ascending=True,
            inplace=True,
            ignore_index=True,
        )

        out_name = f"{fname}/clusters_cosine/cluster_{number}.xlsx"
        os.makedirs(os.path.dirname(out_name), exist_ok=True)
        df.to_csv(
            f"{fname}/clusters_cosine/cluster_{number}.csv",
            index=False,
        )

        SaveXlsxFromFrame(
            df,
            out_name,
            molCols=[
                "smiles_list",
            ],
            size=(300, 300),
        )


def df_to_csv(fname, drug_id, drug_name, smi_passed, name_1, name_2):
    """Saves given data frame to csv (used to save each cluster individually). 

    :param fname: File name.
    :param drug_id: List of IDs of SMILES-stirngs. 
    :param drug_name: List of names of SMILES-stirngs. 
    :param smi_passed: List of SMILES-string. 
    :param name_1: Column name used for IDs.
    :param name_2: Column name used for names.
    """

    df = pd.DataFrame(
        {
            "smiles_list": smi_passed,
            name_1: drug_id,
            name_2: drug_name,
        }
    )

    out_name = f"{fname}/filtered_list/filtered_list.csv"
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    df.to_csv(
        out_name,
        index=False,
    )

    SaveXlsxFromFrame(
        df,
        out_name[:-4] + ".xlsx",
        molCols=[
            "smiles_list",
        ],
        size=(300, 300),
    )


if __name__ == "__main__":

    # Initialize arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-xlsx", type=str, default="drugs_data")
    parser.add_argument("-min_mwt", type=int, default=200)
    parser.add_argument("-max_mwt", type=int, default=800)
    parser.add_argument("-id1", type=str, default="Drug_ID")
    parser.add_argument("-id2", type=str, default="Drug Name")
    args = parser.parse_args()

    # Make folder
    fname = "cluster_analysis"
    os.makedirs(fname, exist_ok=True)

    # Read xlsx
    drug_data = pd.read_excel(f"../data/{args.xlsx}.xlsx", engine="openpyxl")
    smiles = list(drug_data["SMILES"])
    drugid = list(drug_data[args.id1])
    drname = list(drug_data[args.id2])

    # define number of clusters
    number_of_clusters = 8

    # Gets all properties from smiles list
    print(f"Calculating molecular properties of {len(smiles)} drug molecules:")
    wgt, rot, hba, hbd, psa, rng, sp3, ste = get_propertiest(smiles)
    get_hist(fname, wgt, rot, hba, hbd, psa, rng, sp3, ste, "property_all", bins=20)

    # Apply weight-filters and get fingerprints from smiles
    fps, drug_id, drug_name, smi_passed = get_filtered_smiles_and_fps(
        smiles, drugid, drname, max_mwt=args.max_mwt, min_mwt=args.min_mwt
    )
    print(
        f"Applied weight filters ({args.min_mwt} - {args.max_mwt} g/mol): "
        f"{len(smiles)} drugs were reduced to {len(smi_passed)} drugs."
    )

    # Save filtered csv
    df_to_csv(fname, drug_id, drug_name, smi_passed, args.id1, args.id2)

    # Plots histogram of properteies from filtered smiles
    print(f"Calculating molecular properties of selected {len(smi_passed)} drug molecules:")
    wgt, rot, hba, hbd, psa, rng, sp3, ste = get_propertiest(smi_passed)
    get_hist(fname, wgt, rot, hba, hbd, psa, rng, sp3, ste, "property_filtered", bins=10)

    # Get similarity-matrix from fingerprints
    print(f"Calculating similarity matrix of {len(fps)} ECFP vectors:")
    matrix = get_similarity_matrix(fps)

    # Cluster similarity-matrix, get cluster labels and plot coloured PCA
    print(f"Cluster the calculated matrix into {number_of_clusters} clusters:")
    labels = get_clusters(matrix, number_of_clusters)
    get_coloured_pca(fname, matrix, labels, "pca_all_dimesnions", two_d=False, pointsize=30)
    get_coloured_pca(fname, matrix, labels, "pca_two_dimesnions", two_d=True, pointsize=30)

    # Save clusters to csv, ranked with cosine distance
    save_clusters_to_csv_cosine(
        fname,
        smi_passed,
        drug_id,
        drug_name,
        matrix,
        number_of_clusters,
        args.id1,
        args.id2,
    )
    print(f"The results of the clustering are saved here: {fname}")
