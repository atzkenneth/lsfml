import matplotlib.pyplot as plt
import torch, os
import numpy as np
from glob import glob
from sklearn.metrics import mean_absolute_error, roc_auc_score
import pandas as pd

def get_accuracy_roc(results):

    # print(results)
    p1, p2, p3 = results[0], results[1], results[2]

    # first run
    data = torch.load(p1, map_location=torch.device('cpu'))
    ys, ps, = data[2], data[3]
    ps = [1 if x.item() >= 0.5 else 0 for x in ps]
    ys = [int(x.item()) for x in ys]
    # print("True:", ys)
    # print("Pred:", ps)
    roc_1 = roc_auc_score(np.array(ys), np.array(ps))

    # second run
    data = torch.load(p2, map_location=torch.device('cpu'))
    ys, ps, = data[2], data[3]
    ps = [1 if x.item() >= 0.5 else 0 for x in ps]
    ys = [int(x.item()) for x in ys]
    roc_2 = roc_auc_score(np.array(ys), np.array(ps))

    # third run
    data = torch.load(p3, map_location=torch.device('cpu'))
    ys, ps, = data[2], data[3]
    ps = [1 if x.item() >= 0.5 else 0 for x in ps]
    ys = [int(x.item()) for x in ys]
    roc_3 = roc_auc_score(np.array(ys), np.array(ps))

    # Combined heatmaps
    rocs = np.array([roc_1 * 100, roc_2 * 100, roc_3 * 100])

    # print(heatmaps.shape)
    percentage = np.mean(rocs, axis=0)
    percentage_std = np.std(rocs, axis=0)
    print(percentage, percentage_std)

    return percentage, percentage_std

def get_accuracy_tripicate(results, categories=2):

    pre = ["True", "False"]
    exp = ["False", "True"]

    p1, p2, p3 = results[0], results[1], results[2]

    # first run
    data = torch.load(p1, map_location=torch.device('cpu'))
    ys, ps, = data[2], data[3]
    ps = [p if p < 1 else 1 for p in ps]
    ps = [p if p > 0 else 0 for p in ps]
    ys = [float(y) for y in ys]
    ps = [float(p) for p in ps]
    heatmap, xedges, yedges = np.histogram2d(ys, ps, bins=categories, range=[[0, 1], [0, 1]])
    heatmap = np.array(heatmap) 
    heatmap1 = np.flip(heatmap, 0).astype(int)

    # second run
    data = torch.load(p2, map_location=torch.device('cpu'))
    ys, ps, = data[2], data[3]
    ps = [p if p < 1 else 1 for p in ps]
    ps = [p if p > 0 else 0 for p in ps]
    ys = [float(y) for y in ys]
    ps = [float(p) for p in ps]
    heatmap, xedges, yedges = np.histogram2d(ys, ps, bins=categories, range=[[0, 1], [0, 1]])
    heatmap = np.array(heatmap) 
    heatmap2 = np.flip(heatmap, 0).astype(int)

    # third run
    data = torch.load(p3, map_location=torch.device('cpu'))
    ys, ps, = data[2], data[3]
    ps = [p if p < 1 else 1 for p in ps]
    ps = [p if p > 0 else 0 for p in ps]
    ys = [float(y) for y in ys]
    ps = [float(p) for p in ps]
    heatmap, xedges, yedges = np.histogram2d(ys, ps, bins=categories, range=[[0, 1], [0, 1]])
    heatmap = np.array(heatmap) 
    heatmap3 = np.flip(heatmap, 0).astype(int)

    # Get 3x percentage
    percentage1 = int((heatmap1[0][1] + heatmap1[1][0]) / len(ys) * 1000) / 10
    percentage2 = int((heatmap2[0][1] + heatmap2[1][0]) / len(ys) * 1000) / 10
    percentage3 = int((heatmap3[0][1] + heatmap3[1][0]) / len(ys) * 1000) / 10

    return percentage1, percentage2, percentage3


def get_model_list_rand_split(idx=42, testset=3):

    files = sorted(glob(f"results/config_{idx}*_{testset}.pt"))
    print(files, testset)

    gt_2f = files[:3]
    gt_3f = files[3:6]
    gtq2f = files[6:9]
    gtq3f = files[9:12]

    gt_2m, gt_2s = get_accuracy_roc(gt_2f)
    gtq2m, gtq2s = get_accuracy_roc(gtq2f)
    gt_3m, gt_3s = get_accuracy_roc(gt_3f)
    gtq3m, gtq3s = get_accuracy_roc(gtq3f)

    model_dict_mean = {
        "GTNN2D": gt_2m,
        "GTNN2DQM": gtq2m,
        "GTNN3D": gt_3m,
        "GTNN3DQM": gtq3m,
    }

    model_dict_std = {
        "GTNN2D": gt_2s,
        "GTNN2DQM": gtq2s,
        "GTNN3D": gt_3s,
        "GTNN3DQM": gtq3s,
    }

    gt_2a, gt_2b, gt_2c = get_accuracy_tripicate(gt_2f)
    gtq2a, gtq2b, gtq2c = get_accuracy_tripicate(gtq2f)
    gt_3a, gt_3b, gt_3c = get_accuracy_tripicate(gt_3f)
    gtq3a, gtq3b, gtq3c = get_accuracy_tripicate(gtq3f)

    model_dict_trip = {
        "GTNN2D": [gt_2a, gt_2b, gt_2c],
        "GTNN2DQM": [gtq2a, gtq2b, gtq2c],
        "GTNN3D": [gt_3a, gt_3b, gt_3c],
        "GTNN3DQM": [gtq3a, gtq3b, gtq3c],
    }

    return model_dict_mean, model_dict_std, model_dict_trip

def get_yield_mae(results):

    p1, p2, p3 = results[0], results[1], results[2]

    # first run
    data = torch.load(p1, map_location=torch.device('cpu'))
    ys, ps, = data[2], data[3]
    ps = [p if p < 1 else 1 for p in ps]
    ps = [p if p > 0 else 0 for p in ps]
    ys = [float(y) for y in ys]
    ps = [float(p) for p in ps]
    mae1 = mean_absolute_error(ps, ys)

    # second run
    data = torch.load(p2, map_location=torch.device('cpu'))
    ys, ps, = data[2], data[3]
    ps = [p if p < 1 else 1 for p in ps]
    ps = [p if p > 0 else 0 for p in ps]
    ys = [float(y) for y in ys]
    ps = [float(p) for p in ps]
    mae2 = mean_absolute_error(ps, ys)

    # third run
    data = torch.load(p3, map_location=torch.device('cpu'))
    ys, ps, = data[2], data[3]
    ps = [p if p < 1 else 1 for p in ps]
    ps = [p if p > 0 else 0 for p in ps]
    ys = [float(y) for y in ys]
    ps = [float(p) for p in ps]
    mae3 = mean_absolute_error(ps, ys)

    mae1 *= 100
    mae2 *= 100
    mae3 *= 100

    tripli = [mae1, mae2, mae3]

    return np.mean(tripli), np.std(tripli), tripli

def get_mad(data, axis=-1, keepdims=True):
    return np.abs(data - data.mean(axis, keepdims=keepdims)).sum(axis)/len(data)

def get_lr(p, name):

    plt.figure(figsize=(10, 8))

    data = torch.load(p, map_location=torch.device('cpu'))
    tr, ev, te = data[0], data[1], data[4]
    
    epochs = np.arange(0, len(tr))

    # print(f"MAE of saved model: {name}, Error: {ev[-1]}, Epoch: {len(tr)}")

    plt.plot(
        epochs,
        te,
        "red",
        label="Test Loss \nAvg. last 10 epochs = "
        + str(np.round(np.mean(te[-10:]), 4)),
    )
    plt.plot(
        epochs,
        ev,
        "orange",
        label="Validation Loss \nAvg. last 10 epochs = "
        + str(np.round(np.mean(ev[-10:]), 4)),
    )

    plt.plot(
        epochs,
        tr,
        "royalblue",
        label="Training Loss \nAvg. last 10 epochs = "
        + str(np.round(np.mean(tr[-10:]), 4)),
    )

    print(name, np.round(np.mean(tr[-10:]), 4), np.round(np.mean(ev[-10:]), 4), np.round(np.mean(te[-10:]), 4))
    
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

    data = torch.load(p, map_location=torch.device('cpu'))
    ys, ps, maes, = data[2], data[3], data[4]

    ps = [p * 100 for p in ps]
    ps = [p if p < 100 else 100 for p in ps]
    ps = [p if p > 0 else 0 for p in ps]
    ys = [y * 100 for y in ys]

    mae = np.mean(maes)
    print(name.split("_")[-3], name.split("_")[-2], mae)
    
    plt.xlim([-2, 102])
    plt.ylim([-2, 102])
    plt.scatter(ps, ys, marker='o', s=25, c='royalblue')
    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)
    plt.ylabel("Experimental yield / %", fontsize=16)
    plt.xlabel("Predicted yield / %", fontsize=16)
    os.makedirs(os.path.dirname(name), exist_ok=True)
    plt.savefig(name, dpi=400)
    plt.clf()
    
    return

def get_mad_of_dataset(p):

    plt.figure(figsize=(15, 9))

    data1 = torch.load(p+"1.pt", map_location=torch.device('cpu'))
    data2 = torch.load(p+"2.pt", map_location=torch.device('cpu'))
    data3 = torch.load(p+"3.pt", map_location=torch.device('cpu'))
    data4 = torch.load(p+"4.pt", map_location=torch.device('cpu'))
    ys1, ps1, = data1[2], data1[3],
    ys2, ps2, = data2[2], data2[3],
    ys3, ps3, = data3[2], data3[3],
    ys4, ps4, = data4[2], data4[3],

    ps = ps1 + ps2 + ps3 + ps4
    ys = ys1 + ys2 + ys3 + ys4

    ps = [p * 100 for p in ps]
    ps = [p if p < 100 else 100 for p in ps]
    ps = [p if p > 0 else 0 for p in ps]
    ys = [y * 100 for y in ys]

    mad = get_mad(np.array(ys))
    mad = int(mad * 1000) / 1000

    return mad

def get_scatter_full(p, name):

    plt.figure(figsize=(15, 9))

    data1 = torch.load(p+"1.pt", map_location=torch.device('cpu'))
    data2 = torch.load(p+"2.pt", map_location=torch.device('cpu'))
    data3 = torch.load(p+"3.pt", map_location=torch.device('cpu'))
    data4 = torch.load(p+"4.pt", map_location=torch.device('cpu'))
    ys1, ps1, = data1[2], data1[3],
    ys2, ps2, = data2[2], data2[3],
    ys3, ps3, = data3[2], data3[3],
    ys4, ps4, = data4[2], data4[3],

    ps = ps1 + ps2 + ps3 + ps4
    ys = ys1 + ys2 + ys3 + ys4

    ps = [p * 100 for p in ps]
    ps = [p if p < 100 else 100 for p in ps]
    ps = [p if p > 0 else 0 for p in ps]
    ys = [y * 100 for y in ys]

    mad = get_mad(np.array(ys))
    mad = int(mad * 1000) / 1000
    mae = mean_absolute_error(ps, ys)
    mae = int(mae * 1000) / 1000
    print(name, mae, mad, mad / mae)
    
    plt.xlim([-2, 102])
    plt.ylim([-2, 102])
    plt.plot([-2, 102], [-2, 102], 'black', linewidth=1, linestyle='dashed')
    plt.scatter(ps, ys, marker='o', s=60, c='lightskyblue', label=f"MAE: {mae}\nNull model: {mad}\nNull model / MAE: {int(1000 * mad/mae) / 1000}")
    legend =plt.legend(loc='upper left', fontsize=18, handlelength=0, handletextpad=0,)
    LH = legend.legendHandles
    LH[0].set_color('w') 
    plt.tick_params(axis="x", labelsize=30)
    plt.tick_params(axis="y", labelsize=30)
    plt.ylabel("Experimental yield / %", fontsize=30)
    plt.xlabel("Predicted yield / %", fontsize=30)
    os.makedirs(os.path.dirname(name), exist_ok=True)
    plt.tight_layout()
    plt.savefig(name, dpi=400)
    # plt.savefig(name[:-4]+".pdf", dpi=400)
    plt.clf()
    
    return

def get_model_list_yield(idx=32, testset=1, mad_of_dataset=12.0):

    files = sorted(glob(f"results/config_{idx}*_{testset}.pt"))

    gt_2f = files[:3]
    gt_3f = files[3:6]
    gtq2f = files[6:9]
    gtq3f = files[9:12]

    gt_2m, gt_2s, gt_2t = get_yield_mae(gt_2f)
    gtq2m, gtq2s, gtq2t = get_yield_mae(gtq2f)
    gt_3m, gt_3s, gt_3t = get_yield_mae(gt_3f)
    gtq3m, gtq3s, gtq3t = get_yield_mae(gtq3f)

    model_dict_mean = {
        "GTNN2D": gt_2m,
        "GTNN2DQM": gtq2m,
        "GTNN3D": gt_3m,
        "GTNN3DQM": gtq3m,
        # "Null model": mad_of_dataset,
    }

    model_dict_std = {
        "GTNN2D": gt_2s,
        "GTNN2DQM": gtq2s,
        "GTNN3D": gt_3s,
        "GTNN3DQM": gtq3s,
        # "Null model": 0,
    }

    model_dict_trip = {
        "GTNN2D": gt_2t, 
        "GTNN2DQM": gtq2t,
        "GTNN3D": gt_3t,
        "GTNN3DQM": gtq3t,
        # "Null model": [mad_of_dataset, mad_of_dataset, mad_of_dataset],
    }

    return model_dict_mean, model_dict_std, model_dict_trip

def get_model_list_yield_large(idx=3, testset=1, mad_of_dataset=12.0):

    files = sorted(glob(f"results/config_{idx}*_{testset}.pt"))

    gt_2f = files[:3]
    gn_2f = files[3:6]
    gt_3f = files[6:9]
    gn_3f = files[9:12]
    gtq2f = files[12:15]
    gnq2f = files[15:18]
    gtq3f = files[18:21]
    gnq3f = files[21:24]

    gn_2m, gn_2s, gn_2t = get_yield_mae(gn_2f)
    gnq2m, gnq2s, gnq2t = get_yield_mae(gnq2f)
    gn_3m, gn_3s, gn_3t = get_yield_mae(gn_3f)
    gnq3m, gnq3s, gnq3t = get_yield_mae(gnq3f)
    gt_2m, gt_2s, gt_2t = get_yield_mae(gt_2f)
    gtq2m, gtq2s, gtq2t = get_yield_mae(gtq2f)
    gt_3m, gt_3s, gt_3t = get_yield_mae(gt_3f)
    gtq3m, gtq3s, gtq3t = get_yield_mae(gtq3f)


    model_dict_mean = {
        "GNN2D": gn_2m,
        "GTNN2D": gt_2m,
        "GNN2DQM": gnq2m,
        "GTNN2DQM": gtq2m,
        "GNN3D": gn_3m,
        "GTNN3D": gt_3m,
        "GNN3DQM": gnq3m,
        "GTNN3DQM": gtq3m,
        # "Null": mad_of_dataset,
    }

    model_dict_std = {
        "GNN2D": gn_2s,
        "GTNN2D": gt_2s,
        "GNN2DQM": gnq2s,
        "GTNN2DQM": gtq2s,
        "GNN3D": gn_3s,
        "GTNN3D": gt_3s,
        "GNN3DQM": gnq3s,
        "GTNN3DQM": gtq3s,
        # "Null": 0,
    }

    model_dict_trip = {
        "ECFP4NN": bs_1t,
        "GNN2D": gn_2t,
        "GTNN2D": gt_2t, 
        "GNN2DQM": gnq2t,
        "GTNN2DQM": gtq2t,
        "GNN3D": gn_3t,
        "GTNN3D": gt_3t,
        "GNN3DQM": gnq3t,
        "GTNN3DQM": gtq3t,
        # "Null": [mad_of_dataset, mad_of_dataset, mad_of_dataset],
    }

    return model_dict_mean, model_dict_std, model_dict_trip

def plt_barplot(sorted_model_dict, model_dict_std, name, ylabel, xlabel, ylim, imgsize1, imgsize2, inverted):
    
    plt.figure(figsize=(imgsize1, imgsize2))

    keys = list(sorted_model_dict.keys())
    accs = []
    stds = []

    for k in sorted_model_dict:
        accs.append(np.mean(np.array(sorted_model_dict[k])))
        stds.append(np.mean(np.array(model_dict_std[k])))
    
    if inverted:
        plt.barh(keys, accs, xerr=stds, color="lightskyblue")
        plt.tick_params(axis="x", labelsize=24)
        plt.tick_params(axis="y", labelsize=24)
        plt.xlabel(f"\n{ylabel} / %", fontsize=24)
        plt.ylabel(f"\n{xlabel}", fontsize=24)
    else:
        plt.bar(keys, accs, color="lightskyblue") # grey
        plt.errorbar(keys, accs, yerr=stds, color="black", ls='none', elinewidth=2.5) 
        plt.tick_params(axis="x", labelsize=30, rotation=90)
        plt.tick_params(axis="y", labelsize=30)
        plt.xlabel(f"\n{xlabel}", fontsize=36)
        plt.ylabel(f"\n{ylabel} / %", fontsize=36)
    
    if ylim:
        bottom, top = plt.xlim()
        plt.ylim((min(accs) - 10, max(accs) + 3)) 
    plt.tight_layout()
    plt.savefig(f"plots/scatters/{name}.png", dpi=400)
    plt.clf()



if __name__ == "__main__":

    os.makedirs("plots/", exist_ok=True)
    test_sets = [1,2,3,4,]


    get_scatter_full(p=f"results/config_320_1_", name=f"plots/scatters/gtnn2d.png")
    get_scatter_full(p=f"results/config_321_1_", name=f"plots/scatters/gtnn3d.png")
    get_scatter_full(p=f"results/config_322_1_", name=f"plots/scatters/gtnn2dqm.png")
    get_scatter_full(p=f"results/config_323_1_", name=f"plots/scatters/gtnn3dqm.png")
    mad_of_dataset = get_mad_of_dataset(p=f"results/config_320_1_")


    for test_set in test_sets:

        model_dict_mean, model_dict_std, model_dict_trip = get_model_list_yield(idx=32, testset=test_set, mad_of_dataset=mad_of_dataset)

        sorted_models = sorted(model_dict_mean, key=lambda k: model_dict_mean[k], reverse=True)
        sorted_model_dict = {}
        sorted_model_dict_trip = {}
        for k in sorted_models:
            sorted_model_dict[k] = model_dict_mean[k]
            sorted_model_dict_trip[k] = model_dict_trip[k]
        
        print(f"\nModel performance (MAE, smaller is better) on yield prediction:")
        for k in sorted_model_dict_trip:
            print(k, np.mean(np.array(sorted_model_dict_trip[k])), np.std(np.array(sorted_model_dict_trip[k])))
        
        df4 = pd.DataFrame(sorted_model_dict_trip)
        sorted_model_dict1, model_dict_std1 = sorted_model_dict, model_dict_std
        plt_barplot(sorted_model_dict, model_dict_std, f"barplot_model_type_yield_{test_set}", f"Mean absolute error on \nyield prediction", f"Neural network type", None, 8, 11, None)

    for test_set in test_sets:
        model_dict_mean, model_dict_std, model_dict_trip = get_model_list_rand_split(idx=42, testset=test_set)

        sorted_models = sorted(model_dict_mean, key=lambda k: model_dict_mean[k], reverse=False)
        sorted_model_dict = {}
        for k in sorted_models:
            sorted_model_dict[k] = model_dict_mean[k]
        
        print(f"\nModel performance (accuracy, greater is better) on random split:")
        for k in sorted_model_dict:
            print(k, np.mean(np.array(sorted_model_dict[k])), model_dict_std[k])
        
        sorted_model_dict2, model_dict_std2 = sorted_model_dict, model_dict_std
        plt_barplot(sorted_model_dict2, model_dict_std2, f"barplot_model_type_bin_{test_set}", f"\nBalanced accuracy (AUC)", f"Neural network type", True, 8, 11, None)
