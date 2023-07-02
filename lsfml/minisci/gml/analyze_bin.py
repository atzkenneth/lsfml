import matplotlib.pyplot as plt
import torch, os
import numpy as np
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import torch, os
import numpy as np
from glob import glob

matplotlib.rcParams["axes.linewidth"] = 1
matplotlib.rcParams["xtick.major.size"] = 1
matplotlib.rcParams["xtick.major.width"] = 1
matplotlib.rcParams["xtick.minor.size"] = 1
matplotlib.rcParams["xtick.minor.width"] = 1

pre = ["Negative", "Positive"]
exp = ["Negative", "Positive"]


def get_heat(p, name, categories=2):
    p = f"results/config_{p}_"
    p11, p21, p31 = f"{p}1_1.pt", f"{p}1_2.pt", f"{p}1_3.pt"
    p12, p22, p32 = f"{p}2_1.pt", f"{p}2_2.pt", f"{p}2_3.pt"
    p13, p23, p33 = f"{p}3_1.pt", f"{p}3_2.pt", f"{p}3_3.pt"
    p14, p24, p34 = f"{p}4_1.pt", f"{p}4_2.pt", f"{p}4_3.pt"

    # first run
    data11 = torch.load(p11, map_location=torch.device("cpu"))
    data12 = torch.load(p12, map_location=torch.device("cpu"))
    data13 = torch.load(p13, map_location=torch.device("cpu"))
    data14 = torch.load(p14, map_location=torch.device("cpu"))
    ys = data11[2] + data12[2] + data13[2] + data14[2]
    ps = data11[3] + data12[3] + data13[3] + data14[3]
    ps = [p * 100 for p in ps]
    ps = [p if p < 100 else 100 for p in ps]
    ps = [p if p > 0 else 0 for p in ps]
    ys = [y * 100 for y in ys]
    ys = [float(y) for y in ys]
    ps = [float(p) for p in ps]
    heatmap, xedges, yedges = np.histogram2d(ys, ps, bins=categories, range=[[0, 100], [0, 100]])
    heatmap = np.array(heatmap)
    heatmap1 = heatmap.astype(int)

    tn, fp, fn, tp = heatmap1[0][0], heatmap1[0][1], heatmap1[1][0], heatmap1[1][1]

    if tp == 0:
        ppv1 = 0
        tpr1 = 0
    else:
        ppv1 = tp / (tp + fp)
        tpr1 = tp / (tp + fn)

    if tn == 0:
        npv1 = 0
        tnr1 = 0
    else:
        npv1 = tn / (tn + fn)
        tnr1 = tn / (tn + fp)

    f_score1 = (2 * ppv1 * tpr1) / (ppv1 + tpr1)
    # print(f"Combined: {ppv1 + npv1 + tpr1 + tnr1}, PPV: {ppv1}, NPV: {npv1}, TPR: {tpr1}, TNR: {tnr1}")

    # second run
    data21 = torch.load(p21, map_location=torch.device("cpu"))
    data22 = torch.load(p22, map_location=torch.device("cpu"))
    data23 = torch.load(p23, map_location=torch.device("cpu"))
    data24 = torch.load(p24, map_location=torch.device("cpu"))
    ys = data21[2] + data22[2] + data23[2] + data24[2]
    ps = data21[3] + data22[3] + data23[3] + data24[3]
    ps = [p * 100 for p in ps]
    ps = [p if p < 100 else 100 for p in ps]
    ps = [p if p > 0 else 0 for p in ps]
    ys = [y * 100 for y in ys]
    ys = [float(y) for y in ys]
    ps = [float(p) for p in ps]
    heatmap, xedges, yedges = np.histogram2d(ys, ps, bins=categories, range=[[0, 100], [0, 100]])
    heatmap = np.array(heatmap)
    heatmap2 = heatmap.astype(int)

    tn, fp, fn, tp = heatmap2[0][0], heatmap2[0][1], heatmap2[1][0], heatmap2[1][1]

    if tp == 0:
        ppv2 = 0
        tpr2 = 0
    else:
        ppv2 = tp / (tp + fp)
        tpr2 = tp / (tp + fn)

    if tn == 0:
        npv2 = 0
        tnr2 = 0
    else:
        npv2 = tn / (tn + fn)
        tnr2 = tn / (tn + fp)

    f_score2 = (2 * ppv2 * tpr2) / (ppv2 + tpr2)
    # print(f"Combined: {ppv2 + npv2 + tpr2 + tnr2}, PPV: {ppv2}, NPV: {npv2}, TPR: {tpr2}, TNR: {tnr2}")

    # third run
    data31 = torch.load(p31, map_location=torch.device("cpu"))
    data32 = torch.load(p32, map_location=torch.device("cpu"))
    data33 = torch.load(p33, map_location=torch.device("cpu"))
    data34 = torch.load(p34, map_location=torch.device("cpu"))
    ys = data31[2] + data32[2] + data33[2] + data34[2]
    ps = data31[3] + data32[3] + data33[3] + data34[3]
    ps = [p * 100 for p in ps]
    ps = [p if p < 100 else 100 for p in ps]
    ps = [p if p > 0 else 0 for p in ps]
    ys = [y * 100 for y in ys]
    ys = [float(y) for y in ys]
    ps = [float(p) for p in ps]
    heatmap, xedges, yedges = np.histogram2d(ys, ps, bins=categories, range=[[0, 100], [0, 100]])
    heatmap = np.array(heatmap)
    heatmap3 = heatmap.astype(int)

    tn, fp, fn, tp = heatmap3[0][0], heatmap3[0][1], heatmap3[1][0], heatmap3[1][1]

    if tp == 0:
        ppv3 = 0
        tpr3 = 0
    else:
        ppv3 = tp / (tp + fp)
        tpr3 = tp / (tp + fn)

    if tn == 0:
        npv3 = 0
        tnr3 = 0
    else:
        npv3 = tn / (tn + fn)
        tnr3 = tn / (tn + fp)

    f_score3 = (2 * ppv3 * tpr3) / (ppv3 + tpr3)
    # print(f"Combined: {ppv3 + npv3 + tpr3 + tnr3}, PPV: {ppv3}, NPV: {npv3}, TPR: {tpr3}, TNR: {tnr3}")

    # Combined heatmaps
    heatmaps = np.array([heatmap1, heatmap2, heatmap3])

    # print(heatmaps.shape)
    h_mean = np.mean(heatmaps, axis=0)
    h_std = np.std(heatmaps, axis=0)
    # h_mean = np.flip(h_mean.T, 0)
    # h_std = np.flip(h_std.T, 0)
    # print(h_mean, h_std)
    # print(h_mean[0][1], h_mean[1][0])
    percentage = int((h_mean[0][0] + h_mean[1][1]) / len(ys) * 1000) / 10
    percentage_std = int((h_std[0][0] + h_std[1][1]) / len(ys) * 1000) / 10

    # Start plot
    fig, ax = plt.subplots()
    im = ax.imshow(h_mean, cmap="Blues")  # PuOr

    # Get statistic values
    ppv = np.array([ppv1, ppv2, ppv3])
    npv = np.array([npv1, npv2, npv3])
    tpr = np.array([tpr1, tpr2, tpr3])
    tnr = np.array([tnr1, tnr2, tnr3])
    fsc = np.array([f_score1, f_score2, f_score3])

    ppv_mean, ppv_std = int(np.mean(ppv) * 1000) / 10, int(np.std(ppv) * 1000) / 10
    npv_mean, npv_std = int(np.mean(npv) * 1000) / 10, int(np.std(npv) * 1000) / 10
    tpr_mean, tpr_std = int(np.mean(tpr) * 1000) / 10, int(np.std(tpr) * 1000) / 10
    tnr_mean, tnr_std = int(np.mean(tnr) * 1000) / 10, int(np.std(tnr) * 1000) / 10
    f_score_mean, f_score_std = int(np.mean(fsc) * 1000) / 10, int(np.std(fsc) * 1000) / 10

    # print(f"{name}, PPV: {ppv_mean} ($\pm${ppv_std}), NPV: {npv_mean} ($\pm${npv_std}), TPR: {tpr_mean} ($\pm${tpr_std}), TNR: {tnr_mean} ($\pm${tnr_std}), Percentage: {percentage} ($\pm${percentage_std}), F-Score: {f_score_mean} ($\pm${f_score_std})")
    print(
        f"{name[:-4]} {f_score_mean} ($\pm${f_score_std}) & {ppv_mean} ($\pm${ppv_std}) & {tpr_mean} ($\pm${tpr_std}) & {percentage} ($\pm${percentage_std})"
    )

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(exp)))
    ax.set_yticks(np.arange(len(pre)))
    ax.set_xticklabels(exp)
    ax.set_yticklabels(pre)
    # ax.text(1.25, 1.8, f"Number of reactions: {len(ys)}\nCorrectly assigned: {percentage} (±{percentage_std})%\nPPV: {ppv_mean} (±{ppv_std}) %,  NPV: {npv_mean} (±{npv_std}) %\nTPR: {tpr_mean} (±{tpr_std}) %,  TNR: {tnr_mean} (±{tnr_std}) %", bbox={"facecolor": "white", "pad": 8}, fontsize=10)
    # ax.text(0.65, 2.2, f"Number of reactions: {len(ys)}\nCorrectly assigned: {percentage} (±{percentage_std})%\nPPV: {ppv_mean} (±{ppv_std}) %,\nTPR: {tpr_mean} (±{tpr_std}) %,\nF-Score: {f_score_mean} (±{f_score_std}) %", bbox={"facecolor": "white", "pad": 8}, fontsize=16)
    # ax.text(-0.8, 2.5, f"Correctly assigned reactions: {percentage} (±{percentage_std})%,\nF-Score: {f_score_mean} (±{f_score_std}) %,\nPPV: {ppv_mean} (±{ppv_std}) %, TPR: {tpr_mean} (±{tpr_std}) %.", fontsize=20)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(pre)):
        for j in range(len(exp)):
            # if (i == 0) and (j == 0):
            if i == j:
                text = ax.text(
                    j,
                    i,
                    f"{np.round(h_mean[i, j], 1)} (±{np.round(h_std[i, j], 1)})",
                    ha="center",
                    va="center",
                    color="w",
                    size=20,
                )
            else:
                text = ax.text(
                    j,
                    i,
                    f"{np.round(h_mean[i, j], 1)} (±{np.round(h_std[i, j], 1)})",
                    ha="center",
                    va="center",
                    color="navy",
                    size=20,
                )
    # ax.set_title("Reaction yield prediction of literature data (borylation)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.ax.set_ylabel("Reactions / #", fontsize=24)
    cbar.ax.tick_params(labelsize=20)
    cbar.minorticks_on()

    plt.tick_params(axis="x", labelsize=22)
    plt.tick_params(axis="y", labelsize=22)
    plt.ylabel(f"Experimental binary reaction outcome", fontsize=24)
    plt.xlabel(f"Predicted binary reaction outcome", fontsize=24)
    plt.gcf().set_size_inches(9, 8)
    # fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)
    # fig.subplots_adjust(top=0.3)
    # fig.subplots_adjust(right=0.3)
    # fig.subplots_adjust(left=0.3)
    plt.savefig(name[:-4] + "t.png", dpi=500, transparent=True)
    plt.savefig(name, dpi=500)
    plt.clf()

    return


if __name__ == "__main__":
    runs = [
        500,
    ]
    os.makedirs("plots/bin_heatmaps/", exist_ok=True)

    for i, run in enumerate(sorted(runs)):
        get_heat(p=run, name=f"plots/bin_heatmaps/std_heat_{run}.png")
