import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams["axes.linewidth"] = 2
matplotlib.rcParams["xtick.major.size"] = 2
matplotlib.rcParams["xtick.major.width"] = 2
matplotlib.rcParams["xtick.minor.size"] = 2
matplotlib.rcParams["xtick.minor.width"] = 2


def plt_barplot(accs, keys, name, ylabel, xlabel, ylim, imgsize1, imgsize2):
    plt.figure(figsize=(imgsize1, imgsize2))
    plt.bar(keys, accs, color="lightskyblue")  # grey
    plt.tick_params(axis="x", labelsize=30, rotation=90)
    plt.tick_params(axis="y", labelsize=30)
    plt.xlabel(f"\n{xlabel}", fontsize=36)
    plt.ylabel(f"\n{ylabel}", fontsize=36)
    plt.xticks(rotation=45, ha="center")

    if ylim:
        bottom, top = plt.xlim()
        plt.ylim((min(accs) - 10, max(accs) + 3))
    plt.tight_layout()
    plt.savefig(f"gml/plots/scatters/{name}.png", dpi=400)
    plt.clf()


if __name__ == "__main__":
    informer = "eln036496"
    insilico = "eln044720"

    name = "hte_only"  # hte_only, plus_lit
    df = pd.read_csv(f"data/20221209_alkylation_full_{name}.csv", sep=",", encoding="unicode_escape")

    rxn_id = list(df["rxn_id"])
    binary = list(df["binary"])
    print(len(rxn_id))

    insilico = [x for x in rxn_id if "eln044720" in x]
    informer = [x for x in rxn_id if "eln036496" in x]
    bin_siic = [x for i, x in enumerate(binary) if "eln044720" in rxn_id[i]]

    print(len(informer), len(insilico), len(bin_siic))

    vals = [bin_siic.count(0), bin_siic.count(1)]
    print(bin_siic.count(0), bin_siic.count(1))
    keys = ["Failure", "Success"]
    plt_barplot(vals, keys, f"insilico_analysis", f"Reactions / #", "Reaction outcome", None, 7, 11)

    ####
    ####

    product_mono = list(df["product_mono"])
    product_di = list(df["product_di"])
    product_non = list(df["product_non"])
    product_all = [float(product_mono[i] + product_di[i]) for i, x in enumerate(product_mono)]
    product_all = [float(i) for i in product_all]
    yld_siic = [x for i, x in enumerate(product_all) if "eln044720" in rxn_id[i]]

    startingmat_1_smiles = list(df["startingmat_1_smiles"])
    smi_siic = [x for i, x in enumerate(startingmat_1_smiles) if "eln044720" in rxn_id[i]]

    rxn_smi1_dict = {}
    rxn_smi2_dict = {}
    for i, smi in enumerate(smi_siic):
        # print(smi, yld_siic[i])

        if smi not in rxn_smi1_dict:
            rxn_smi1_dict[smi] = 0
            rxn_smi2_dict[smi] = 0
        elif smi in rxn_smi1_dict:
            pass

        if bin_siic[i] > 0:
            rxn_smi1_dict[smi] += 1
        else:
            rxn_smi2_dict[smi] += 1
            pass

    smis = list(rxn_smi1_dict.keys())
    num_trans = list(rxn_smi1_dict.values())
    med = sum(i <= 16 and i >= 9 for i in num_trans)
    hig = sum(i >= 17 for i in num_trans)
    low = sum(i <= 8 for i in num_trans)
    print(hig, med, low)

    vals = [low, med, hig]
    keys = ["0-8", "9-16", "17-23"]
    plt_barplot(
        vals,
        keys,
        "insilico_count",
        "Substrates / #",
        "Successful transformations \nper substrate / #",
        None,
        9,
        11,
    )
