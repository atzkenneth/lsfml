from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import SimilarityMaps
from IPython.display import SVG
import io
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from rdkit.Chem import Draw
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl

import argparse, torch, configparser, os, time
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from net_latent import GraphTransformer
from net_utils import DataLSF, get_rxn_ids 
from train import eval

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval(
    model, eval_loader,
):

    model.eval()

    feats = []
    ys = []

    with torch.no_grad():

        for g, g2  in eval_loader:

            g = g.to(DEVICE)
            g2 = g2.to(DEVICE)
            pred = model(g, g2)
            ys.append(g.rxn_trg.detach().cpu().numpy())
            feats.append(pred)
            # print(pred.size(), g.rxn_trg)

    feats = torch.cat(feats, dim=0)
    return ys, feats

def get_predictions(config_id="321"):

    # Load model
    CONFIG_PATH = f"config/config_{config_id}.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    N_KERNELS = int(config["PARAMS"]["N_KERNELS"])
    D_MLP = int(config["PARAMS"]["D_MLP"])
    D_KERNEL = int(config["PARAMS"]["D_KERNEL"])
    D_EMBEDDING = int(config["PARAMS"]["D_EMBEDDING"])
    BATCH_SIZE = int(config["PARAMS"]["BATCH_SIZE"])
    POOLING_HEADS = int(config["PARAMS"]["POOLING_HEADS"])
    QML = int(config["PARAMS"]["QML"])
    GEOMETRY = int(config["PARAMS"]["GEOMETRY"])
    TARGET = str(config["PARAMS"]["TARGET"])
    SPLIT = str(config["PARAMS"]["SPLIT"])
    ELN = str(config["PARAMS"]["ELN"])
    QML = True if QML >= 1 else False
    GEOMETRY = True if GEOMETRY >= 1 else False
    GRAPH_DIM = "edge_3d" if GEOMETRY >= 1 else "edge_2d"


    model_path = "models/"
    model1 = GraphTransformer(N_KERNELS, POOLING_HEADS, D_MLP, D_KERNEL, D_EMBEDDING, QML, GEOMETRY)
    model1.load_state_dict(torch.load(f"{model_path}config_{config_id}_1_1.pt",map_location=torch.device("cpu")))
    model1 = model1.to(DEVICE)
    model2 = GraphTransformer(N_KERNELS, POOLING_HEADS, D_MLP, D_KERNEL, D_EMBEDDING, QML, GEOMETRY)
    model2.load_state_dict(torch.load(f"{model_path}config_{config_id}_1_2.pt",map_location=torch.device("cpu")))
    model2 = model2.to(DEVICE)
    model3 = GraphTransformer(N_KERNELS, POOLING_HEADS, D_MLP, D_KERNEL, D_EMBEDDING, QML, GEOMETRY)
    model3.load_state_dict(torch.load(f"{model_path}config_{config_id}_1_3.pt",map_location=torch.device("cpu")))
    model3 = model3.to(DEVICE)
    model4 = GraphTransformer(N_KERNELS, POOLING_HEADS, D_MLP, D_KERNEL, D_EMBEDDING, QML, GEOMETRY)
    model4.load_state_dict(torch.load(f"{model_path}config_{config_id}_1_4.pt",map_location=torch.device("cpu")))
    model4 = model4.to(DEVICE)

    tran_ids, eval_ids, test_ids = get_rxn_ids(split=SPLIT, eln=ELN, testset="2")
    test_data1 = DataLSF(rxn_ids=test_ids, graph_dim=GRAPH_DIM, rxn_trg=TARGET)
    test_loader1 = DataLoader(test_data1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    tran_ids, eval_ids, test_ids = get_rxn_ids(split=SPLIT, eln=ELN, testset="3")
    test_data2 = DataLSF(rxn_ids=test_ids, graph_dim=GRAPH_DIM, rxn_trg=TARGET)
    test_loader2 = DataLoader(test_data2, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    tran_ids, eval_ids, test_ids = get_rxn_ids(split=SPLIT, eln=ELN, testset="4")
    test_data3 = DataLSF(rxn_ids=test_ids, graph_dim=GRAPH_DIM, rxn_trg=TARGET)
    test_loader3 = DataLoader(test_data3, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    tran_ids, eval_ids, test_ids = get_rxn_ids(split=SPLIT, eln=ELN, testset="1")
    test_data4 = DataLSF(rxn_ids=test_ids, graph_dim=GRAPH_DIM, rxn_trg=TARGET)
    test_loader4 = DataLoader(test_data4, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    gtnns = [model1,model2,model3,model4]
    testsets = [test_loader1,test_loader2,test_loader3,test_loader4]

    gtnn_latents = []
    targets = []

    for idx, model in enumerate(gtnns):
        te_ys, features = eval(model, testsets[idx])
        ys = [item for sublist in te_ys for item in sublist]
        gtnn_latents.append(features)
        targets += ys
    
    gtnn_latents = torch.cat(gtnn_latents, dim=0)
    
    print(targets)
    print(gtnn_latents.size())

    return gtnn_latents.detach().cpu().numpy(), np.array(targets)


def get_coloured_pca(matrix, labels, name, pointsize=20):
    
    pca = PCA(n_components=2)
    pca.fit(matrix)
    expl_variance = pca.explained_variance_ratio_
    pca2 = pca.transform(matrix)
    pca2 = np.array(pca2)
    print(pca2.shape)
    print(labels.shape)

    cmap = plt.cm.get_cmap('coolwarm')

    fontsize = 15

    fig = plt.figure(figsize=(10, 10))
    plt.scatter(pca2[:, 0], pca2[:, 1], c=labels, s=pointsize, cmap=cmap, label="Expl. variance: " + str(expl_variance[0]* 100)[:4] + " % and " +  str(expl_variance[1]* 100)[:3] + " %")
    plt.legend(loc="best", fontsize=fontsize - 2, handletextpad=0, handlelength=0, markerscale=0)
    plt.colorbar()
    plt.tick_params(axis="x", labelsize=fontsize)
    plt.tick_params(axis="y", labelsize=fontsize)
    plt.xlabel(str("Principal component 1"), fontsize=fontsize)
    plt.ylabel(str("Principal component 2"), fontsize=fontsize)

    out_name = os.path.join(f"plots/latents/", name + ".png")
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    # plt.tight_layout()
    plt.savefig(out_name, dpi=200)
    plt.clf()

if __name__ == '__main__':

    runs = ["320", "321", "322", "323", "420", "421", "422", "423"]

    for run in runs:
        gtnn_latents, targets = get_predictions(config_id=run)
        get_coloured_pca(gtnn_latents, targets, name=f"pca_{run}", pointsize=20)

