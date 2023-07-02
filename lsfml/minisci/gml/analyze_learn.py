import matplotlib.pyplot as plt
import torch, os
import numpy as np
from glob import glob


def get_lr(p, name):
    plt.figure(figsize=(10, 8))

    data = torch.load(p, map_location=torch.device("cpu"))
    tr, ev, te = data[0], data[1], data[4]

    epochs = np.arange(0, len(tr))

    # print(f"MAE of saved model: {name}, Error: {ev[-1]}, Epoch: {len(tr)}")

    """plt.plot(
        epochs,
        te,
        "red",
        label="Test Loss \nAvg. last 10 epochs = " + str(np.round(np.mean(te[-10:]), 4)),
    )"""
    plt.plot(
        epochs,
        ev,
        "orange",
        label="Validation Loss \nAvg. last 10 epochs = " + str(np.round(np.mean(ev[-10:]), 4)),
    )

    plt.plot(
        epochs,
        tr,
        "royalblue",
        label="Training Loss \nAvg. last 10 epochs = " + str(np.round(np.mean(tr[-10:]), 4)),
    )

    print(
        name,
        np.round(np.mean(tr[-10:]), 4),
        np.round(np.mean(ev[-10:]), 4),
        np.round(np.mean(te[-10:]), 4),
    )

    plt.legend(loc="best", fontsize=18)
    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)
    plt.ylabel("Average loss per epoch", fontsize=16)
    plt.xlabel("Epochs", fontsize=16)
    os.makedirs(os.path.dirname(name), exist_ok=True)
    plt.savefig(name, dpi=400)
    plt.clf()


def get_scatter(p, name):
    plt.figure(figsize=(10, 8))

    data = torch.load(p, map_location=torch.device("cpu"))
    (
        ys,
        ps,
        maes,
    ) = (
        data[2],
        data[3],
        data[4],
    )

    ps = [p * 100 for p in ps]
    ps = [p if p < 100 else 100 for p in ps]
    ps = [p if p > 0 else 0 for p in ps]
    ys = [y * 100 for y in ys]

    mae = np.mean(maes)
    print(name.split("_")[-3], name.split("_")[-2], mae)

    plt.xlim([-2, 102])
    plt.ylim([-2, 102])
    plt.scatter(ps, ys, marker="o", s=25, c="royalblue")
    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)
    plt.ylabel("Experimental yield / %", fontsize=16)
    plt.xlabel("Predicted yield / %", fontsize=16)
    os.makedirs(os.path.dirname(name), exist_ok=True)
    plt.savefig(name, dpi=400)
    plt.clf()

    return


def get_heat(p, name, categories=2):
    # plt.figure(figsize=(20, 15))

    data = torch.load(p, map_location=torch.device("cpu"))
    (
        ys,
        ps,
    ) = (
        data[2],
        data[3],
    )
    num_reactions = len(ps)

    # print(ys.count(0), ys.count(1), ys.count(2), ys.count(3))

    ps = [p * 100 for p in ps]
    ps = [p if p < 100 else 100 for p in ps]
    ps = [p if p > 0 else 0 for p in ps]
    ys = [y * 100 for y in ys]

    pre = ["True", "False"]
    exp = ["False", "True"]

    heatmap, xedges, yedges = np.histogram2d(ps, ys, bins=categories)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    heatmap = np.array(heatmap)  # / num_reactions * 100
    heatmap = np.flip(heatmap.T, 0).astype(int)

    fn, tp, tn, fp = heatmap[0][0], heatmap[0][1], heatmap[1][0], heatmap[1][1]

    if tp == 0:
        ppv = 0
        tpr = 0
    else:
        ppv = tp / (tp + fp)
        tpr = tp / (tp + fn)

    if tn == 0:
        npv = 0
        tnr = 0
    else:
        npv = tn / (tn + fn)
        tnr = tn / (tn + fp)

    print(f"{name}, Combined: {ppv + npv + tpr + tnr}, PPV: {ppv}, NPV: {npv}, TPR: {tpr}, TNR: {tnr}")

    # zer = (heatmap[0][3] + heatmap[1][2] + heatmap[2][1] + heatmap[3][0]) / len(ys)  * 100
    # one = (heatmap[0][2] + heatmap[1][1] + heatmap[1][3] + heatmap[2][0] + heatmap[2][2] + heatmap[3][1]) / len(ys)  * 100
    # two = (heatmap[0][1] + heatmap[1][0] + heatmap[2][3] + heatmap[3][2]) / len(ys)  * 100
    # thr = (heatmap[0][0] + heatmap[3][3]) / len(ys) * 100
    # print(zer, one, two, thr)

    # print(heatmap)

    fig, ax = plt.subplots()
    im = ax.imshow(heatmap, cmap="copper")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(exp)))
    ax.set_yticks(np.arange(len(pre)))
    ax.set_xticklabels(exp)
    ax.set_yticklabels(pre)
    # ax.text(1.3, 1.9, f"Positive Predictive Value: {int(ppv * 1000)/10} %\nNegative Predictive Value : {int(npv * 1000)/10} %\nTrue Positive Rate : {int(tpr * 1000)/10} %\nTrue Negative Rate: {int(tnr * 1000)/10} %", bbox={"facecolor": "white", "pad": 8}, fontsize=12)
    ax.text(
        1.2,
        1.7,
        f"PPV: {int(ppv * 1000)/10} %  NPV: {int(npv * 1000)/10} %\nTPR: {int(tpr * 1000)/10} %  TNR: {int(tnr * 1000)/10} %",
        bbox={"facecolor": "white", "pad": 8},
        fontsize=14,
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(pre)):
        for j in range(len(exp)):
            text = ax.text(j, i, heatmap[i, j], ha="center", va="center", color="w", size=16)

    # ax.set_title("Reaction yield prediction of literature data (borylation)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.ax.set_ylabel("Number of reactions / $N$", fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    cbar.minorticks_on()

    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)
    plt.ylabel("Experimental / borylation observed", fontsize=16)
    plt.xlabel("Predicted  / borylation observed", fontsize=16)
    plt.gcf().set_size_inches(9, 8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(name), exist_ok=True)
    plt.savefig(name, dpi=400)
    plt.clf()

    return


if __name__ == "__main__":
    runs = sorted(glob("results/config_50*"))  # [:2]
    print("Number of files: ", len(runs))
    runs = [x.split("/")[-1] for x in runs]
    os.makedirs("plots/", exist_ok=True)

    for i, run in enumerate(runs):
        # get_heat(p=f"results/{run}", name=f"plots/heatmaps/heatmap_{run[:-3]}.png")
        get_lr(p=f"results/{run}", name=f"plots/learnings/learning_{run[:-3]}.png")
        # get_scatter(p=f"results/{run}", name=f"plots/scatters/scatter_{run[:-3]}.png")
