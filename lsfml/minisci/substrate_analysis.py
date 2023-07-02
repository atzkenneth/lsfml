import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

META_ORTHO = [1, 2, 3, 4, 5, 6, 8, 9, 11, 16, 19, 20, 21]
ORTHO_ONLY = [10, 12, 13, 14, 15, 17, 18, 22, 23]
INDA_PYRRO = [24, 26, 27]
matplotlib.rcParams["axes.linewidth"] = 2
matplotlib.rcParams["xtick.major.size"] = 2
matplotlib.rcParams["xtick.major.width"] = 2
matplotlib.rcParams["xtick.minor.size"] = 2
matplotlib.rcParams["xtick.minor.width"] = 2


def plt_barplot(sorted_rxn_dict, sorted_int_dict, name, ylabel, xlabel, imgsize1, imgsize2):
    plt.figure(figsize=(imgsize1, imgsize2))

    keys = list(sorted_rxn_dict.keys())
    if sorted_int_dict:
        keys = [sorted_int_dict[smi] for smi in keys]

    accs = []
    for smi in sorted_rxn_dict:
        accs.append(sorted_rxn_dict[smi])
    # Number of substrates with
    plt.bar(
        keys,
        accs,
        color="lightskyblue",
        label="High reaction yield (>35%): 9 (43%)\nMedium reaction yield (11-35%): 9 (43%)\nLow reaction yield (<11%): 3 (14%)",
    )  # grey
    plt.legend(loc="upper right", fontsize=16, handletextpad=0, handlelength=0)
    # plt.errorbar(keys, accs, yerr=stds, color="black", ls='none', elinewidth=2.5)
    plt.tick_params(axis="x", labelsize=20, rotation=90)
    plt.tick_params(axis="y", labelsize=20)
    plt.xlabel(f"\n{xlabel}", fontsize=20)
    plt.ylabel(f"\n{ylabel} / %", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"plots/{name}.png", dpi=400)
    plt.clf()


def box_plot(fg_lst_dict):
    medianprops = dict(linestyle="-", linewidth=4, color="royalblue")
    flierprops = dict(marker="o", markerfacecolor="royalblue", markersize=9, markeredgecolor="none")

    for fg in fg_lst_dict:
        print(fg, np.mean(np.array(fg_lst_dict[fg])), np.std(np.array(fg_lst_dict[fg])), fg_lst_dict[fg])

    plt.figure(figsize=(5, 8))
    # plt.figure(figsize=(5, 7))
    plt.boxplot(fg_lst_dict.values(), flierprops=flierprops, medianprops=medianprops)
    plt.tick_params(axis="x", labelsize=20, rotation=90)
    plt.tick_params(axis="y", labelsize=20)
    plt.xticks([1, 2, 3], list(fg_lst_dict.keys()), fontsize=20)
    # plt.xticks(rotation=90)
    plt.xticks(rotation=45, ha="center")
    plt.ylabel("Average reaction yield / %\n", fontsize=20)
    # plt.xlabel(f"N-heterocyclic substrates", fontsize=22)
    # plt.xlabel("$N$-hetero arenes", fontsize=22)
    plt.title("$N$-hetero arenes\n", fontsize=20, weight="bold")
    plt.tight_layout()
    plt.savefig("plots/substrate_boxplot.png", dpi=400)
    plt.clf()


if __name__ == "__main__":
    df = pd.read_csv("data/20221209_alkylation_full_hte_only.csv", sep=",", encoding="unicode_escape")

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

    # prd_all = list(np.array(product_mono) + np.array(product_di))
    product_all = [float(product_mono[i] + product_di[i]) for i, x in enumerate(product_mono)]
    product_all = [float(i) for i in product_all]

    rxn_smi1_dict = {}
    for idx, yld in enumerate(product_all):
        smi = startingmat_2_smiles[idx]

        if smi not in rxn_smi1_dict:
            tmplist = []
            tmplist.append(product_all[idx])
            rxn_smi1_dict[smi] = tmplist
        elif smi in rxn_smi1_dict:
            tmplist = rxn_smi1_dict[smi]
            tmplist.append(product_all[idx])
            rxn_smi1_dict[smi] = tmplist

    working_subst = []
    for smi in rxn_smi1_dict:
        if int(np.mean(np.array(rxn_smi1_dict[smi])) * 100) >= 10:
            working_subst.append(smi)
        else:
            pass

    rxn_smi2_dict = {}
    for idx, yld in enumerate(product_all):
        smi = startingmat_1_smiles[idx]
        subs_smi = startingmat_2_smiles[idx]

        if subs_smi in working_subst:
            if smi not in rxn_smi2_dict:
                tmplist = []
                tmplist.append(product_all[idx])
                rxn_smi2_dict[smi] = tmplist

            elif smi in rxn_smi2_dict:
                tmplist = rxn_smi2_dict[smi]
                tmplist.append(product_all[idx])
                rxn_smi2_dict[smi] = tmplist

    rxn_dict_mean = {}
    for smi in rxn_smi2_dict:
        rxn_dict_mean[smi] = np.mean(np.array(rxn_smi2_dict[smi])) * 100

    sorted_rxns = sorted(rxn_dict_mean, key=lambda k: rxn_dict_mean[k], reverse=True)
    sorted_rxn_dict = {}
    sorted_int_dict = {}
    counter = 1
    drug_ids = [11, 19, 21, 22, 24, 27]
    for smi in sorted_rxns:
        if counter not in drug_ids:
            sorted_rxn_dict[smi] = rxn_dict_mean[smi]
            sorted_int_dict[smi] = str(counter)
        counter += 1
        print(counter - 1, f"({int(rxn_dict_mean[smi] * 10) / 10}%)")
        # print(smi)

    plt_barplot(
        sorted_rxn_dict,
        sorted_int_dict,
        "substrate_conversion",
        "Average reaction yield",
        "$N$-heterocyclic substrates selected by \nmachine learning",
        8,
        6,
    )

    meta = []
    meor = []
    five = []

    for smi in sorted_rxn_dict:
        yld = sorted_rxn_dict[smi]
        # yld_list = rxn_smi2_dict[smi]
        idx = int(sorted_int_dict[smi])

        if idx in META_ORTHO:
            meta.append(yld)
            # cylc_o += yld_list

        elif idx in ORTHO_ONLY:
            meor.append(yld)
            # amides += yld_list

        elif idx in INDA_PYRRO:
            five.append(yld)
            # cylc_c += yld_list

    fg_lst_dict = {
        "Meta unsubstituted \npyridines": meta,
        "Meta substituted \npyridines": meor,
        "Five-membered \nrings": five,
    }

    box_plot(fg_lst_dict)
