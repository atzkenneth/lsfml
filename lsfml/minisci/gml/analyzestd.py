import matplotlib
import matplotlib.pyplot as plt
import torch, os
import numpy as np
from glob import glob
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

matplotlib.rcParams["axes.linewidth"] = 1
matplotlib.rcParams["xtick.major.size"] = 1
matplotlib.rcParams["xtick.major.width"] = 1
matplotlib.rcParams["xtick.minor.size"] = 1
matplotlib.rcParams["xtick.minor.width"] = 1

# exp = ["<1", "1-11", ">11-35", ">35"]
# pre = ["<1", "1-11", ">11-35", ">35"]
exp = ["0-25", "25-50", "50-75", "75-100"]
pre = ["0-25", "25-50", "50-75", "75-100"]
VMAX = 500
COLOR_CUT = 250


def get_bins(ys, a=0.25, b=0.50, c=0.75):  # a=0.01, b=0.11, c=0.35
    binned = []

    for i in ys:
        if i <= a:
            binned.append(0)
        elif (i <= b) and (i > a):
            binned.append(1)
        elif (i <= c) and (i > b):
            binned.append(2)
        elif i > c:
            binned.append(3)

    print(binned.count(0), binned.count(1), binned.count(2), binned.count(3))

    return binned


def get_heat(p, name, categories=4):
    p = f"results/config_{p}_"

    p1a, p2a, p3a = f"{p}1_1.pt", f"{p}1_2.pt", f"{p}1_3.pt"
    p1b, p2b, p3b = f"{p}2_1.pt", f"{p}2_2.pt", f"{p}2_3.pt"
    p1c, p2c, p3c = f"{p}3_1.pt", f"{p}3_2.pt", f"{p}3_3.pt"
    p1d, p2d, p3d = f"{p}4_1.pt", f"{p}4_2.pt", f"{p}4_3.pt"

    # first run
    data_a = torch.load(p1a, map_location=torch.device("cpu"))
    data_b = torch.load(p1b, map_location=torch.device("cpu"))
    data_c = torch.load(p1c, map_location=torch.device("cpu"))
    data_d = torch.load(p1d, map_location=torch.device("cpu"))
    (
        ys_a,
        ps_a,
    ) = (
        data_a[2],
        data_a[3],
    )
    (
        ys_b,
        ps_b,
    ) = (
        data_b[2],
        data_b[3],
    )
    (
        ys_c,
        ps_c,
    ) = (
        data_c[2],
        data_c[3],
    )
    (
        ys_d,
        ps_d,
    ) = (
        data_d[2],
        data_d[3],
    )
    ys = ys_a + ys_b + ys_c + ys_d
    ys = [float(x) for x in ys]
    ps = ps_a + ps_b + ps_c + ps_d
    ps = [float(x) for x in ps]
    ps = [0 if x <= 0.025 else x for x in ps]
    ys = get_bins(ys)
    ps = get_bins(ps)
    ps = [p / 3 * 100 for p in ps]
    ys = [y / 3 * 100 for y in ys]
    heatmap, xedges, yedges = np.histogram2d(ps, ys, bins=categories)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap = np.array(heatmap)
    # heatmap1 = np.flip(heatmap.T, 0).astype(int)
    heatmap1 = heatmap.T.astype(int)
    pcc1 = pearsonr(ps, ys)
    mae1 = mean_absolute_error(ps, ys)
    zer1 = (heatmap1[0][0] + heatmap1[1][1] + heatmap1[2][2] + heatmap1[3][3]) / len(ys) * 100
    one1 = (
        (heatmap1[0][1] + heatmap1[1][0] + heatmap1[1][2] + heatmap1[2][1] + heatmap1[2][3] + heatmap1[3][2])
        / len(ys)
        * 100
    )
    two1 = (heatmap1[0][2] + heatmap1[1][3] + heatmap1[2][0] + heatmap1[3][1]) / len(ys) * 100
    thr1 = (heatmap1[0][3] + heatmap1[0][3]) / len(ys) * 100

    # second run
    data_a = torch.load(p2a, map_location=torch.device("cpu"))
    data_b = torch.load(p2b, map_location=torch.device("cpu"))
    data_c = torch.load(p2c, map_location=torch.device("cpu"))
    data_d = torch.load(p2d, map_location=torch.device("cpu"))
    (
        ys_a,
        ps_a,
    ) = (
        data_a[2],
        data_a[3],
    )
    (
        ys_b,
        ps_b,
    ) = (
        data_b[2],
        data_b[3],
    )
    (
        ys_c,
        ps_c,
    ) = (
        data_c[2],
        data_c[3],
    )
    (
        ys_d,
        ps_d,
    ) = (
        data_d[2],
        data_d[3],
    )
    ys = ys_a + ys_b + ys_c + ys_d
    ys = [float(x) for x in ys]
    ps = ps_a + ps_b + ps_c + ps_d
    ps = [float(x) for x in ps]
    ps = [0 if x <= 0.025 else x for x in ps]
    ys = get_bins(ys)
    ps = get_bins(ps)
    ps = [p / 3 * 100 for p in ps]
    ys = [y / 3 * 100 for y in ys]
    heatmap, xedges, yedges = np.histogram2d(ps, ys, bins=categories)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap = np.array(heatmap)
    # heatmap2 = np.flip(heatmap.T, 0).astype(int)
    heatmap2 = heatmap.T.astype(int)
    pcc2 = pearsonr(ps, ys)
    mae2 = mean_absolute_error(ps, ys)
    zer2 = (heatmap2[0][0] + heatmap2[1][1] + heatmap2[2][2] + heatmap2[3][3]) / len(ys) * 100
    one2 = (
        (heatmap2[0][1] + heatmap2[1][0] + heatmap2[1][2] + heatmap2[2][1] + heatmap2[2][3] + heatmap2[3][2])
        / len(ys)
        * 100
    )
    two2 = (heatmap2[0][2] + heatmap2[1][3] + heatmap2[2][0] + heatmap2[3][1]) / len(ys) * 100
    thr2 = (heatmap2[0][3] + heatmap2[0][3]) / len(ys) * 100

    # third run
    data_a = torch.load(p3a, map_location=torch.device("cpu"))
    data_b = torch.load(p3b, map_location=torch.device("cpu"))
    data_c = torch.load(p3c, map_location=torch.device("cpu"))
    data_d = torch.load(p3d, map_location=torch.device("cpu"))
    (
        ys_a,
        ps_a,
    ) = (
        data_a[2],
        data_a[3],
    )
    (
        ys_b,
        ps_b,
    ) = (
        data_b[2],
        data_b[3],
    )
    (
        ys_c,
        ps_c,
    ) = (
        data_c[2],
        data_c[3],
    )
    (
        ys_d,
        ps_d,
    ) = (
        data_d[2],
        data_d[3],
    )
    ys = ys_a + ys_b + ys_c + ys_d
    ys = [float(x) for x in ys]
    ps = ps_a + ps_b + ps_c + ps_d
    ps = [float(x) for x in ps]
    ps = [0 if x <= 0.025 else x for x in ps]
    ys = get_bins(ys)
    ps = get_bins(ps)
    ps = [p / 3 * 100 for p in ps]
    ys = [y / 3 * 100 for y in ys]
    heatmap, xedges, yedges = np.histogram2d(ps, ys, bins=categories)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # heatmap3 = np.flip(heatmap.T, 0).astype(int)
    heatmap3 = heatmap.T.astype(int)
    pcc3 = pearsonr(ps, ys)
    mae3 = mean_absolute_error(ps, ys)
    zer3 = (heatmap3[0][0] + heatmap3[1][1] + heatmap3[2][2] + heatmap3[3][3]) / len(ys) * 100
    one3 = (
        (heatmap3[0][1] + heatmap3[1][0] + heatmap3[1][2] + heatmap3[2][1] + heatmap3[2][3] + heatmap3[3][2])
        / len(ys)
        * 100
    )
    two3 = (heatmap3[0][2] + heatmap3[1][3] + heatmap3[2][0] + heatmap3[3][1]) / len(ys) * 100
    thr3 = (heatmap3[0][3] + heatmap3[0][3]) / len(ys) * 100

    # Combined heatmaps
    heatmaps = np.array([heatmap1, heatmap2, heatmap3])
    pccs = np.array([float(pcc1[0]), float(pcc2[0]), float(pcc3[0])])
    maes = np.array([float(mae1), float(mae2), float(mae3)])
    zers = np.array([zer1, zer2, zer3])
    ones = np.array([one1, one2, one3])
    twos = np.array([two1, two2, two3])
    thrs = np.array([thr1, thr2, thr3])

    print(
        name,
        np.round(np.mean(zers), 2),
        f"({np.round(np.std(zers), 1)}) & ",
        np.round(np.mean(ones), 2),
        f"({np.round(np.std(ones), 1)}) & ",
        np.round(np.mean(twos), 2),
        f"({np.round(np.std(twos), 1)}) & ",
        np.round(np.mean(thrs), 2),
        f"({np.round(np.std(thrs), 1)})",
    )

    # print(heatmaps.shape)
    h_mean = np.mean(heatmaps, axis=0)
    h_std = np.std(heatmaps, axis=0)
    pcc_mean = np.mean(pccs)
    pcc_std = np.std(pccs)
    mae_mean = np.mean(maes)
    mae_std = np.std(maes)

    print("PCCs:", pcc_mean, pcc_std)
    print("MAEs:", mae_mean, mae_std)

    # h_mean = np.flip(h_mean.T, 0)
    # h_std = np.flip(h_std.T, 0)
    # print(h_mean, h_std)

    # Start plot
    fig, ax = plt.subplots()
    im = ax.imshow(h_mean, cmap="Blues", vmin=0, vmax=VMAX)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(exp)))
    ax.set_yticks(np.arange(len(pre)))
    ax.set_xticklabels(exp)
    ax.set_yticklabels(pre)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(pre)):
        for j in range(len(exp)):
            if h_mean[i, j] >= COLOR_CUT:
                text = ax.text(
                    j,
                    i,
                    f"{np.round(h_mean[i, j], 1)}\n(±{np.round(h_std[i, j], 1)})",
                    ha="center",
                    va="center",
                    color="w",
                    size=20,
                )
            else:
                text = ax.text(
                    j,
                    i,
                    f"{np.round(h_mean[i, j], 1)}\n(±{np.round(h_std[i, j], 1)})",
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
    plt.ylabel("Experimental reaction yield / %", fontsize=24)
    plt.xlabel("Predicted reaction yield / %", fontsize=24)
    plt.gcf().set_size_inches(9, 8)
    fig.tight_layout()
    plt.savefig(name[:-4] + "t.png", dpi=500, transparent=True)
    plt.savefig(name, dpi=500, transparent=False)
    plt.clf()

    return


if __name__ == "__main__":
    # runs = [320, 321, 322, 323]
    runs = [350, 351, 352, 353]
    runs = [
        500,
    ]
    os.makedirs("plots/std_heatmaps/", exist_ok=True)

    for i, run in enumerate(sorted(runs)):
        get_heat(p=run, name=f"plots/std_heatmaps/std_heat_{run}.png")
