
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse, configparser, os

from net import GraphTransformer
from net_utils import DataLSF, get_rxn_ids 
from torch_geometric.loader import DataLoader
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mae_loss = lambda x, y: F.l1_loss(x, y).item()
os.makedirs("results/", exist_ok=True)
os.makedirs("models/", exist_ok=True)

def train(
    model, optimizer, criterion, train_loader,
):

    model.train()
    training_loss = []

    for g, g2 in train_loader:

        g = g.to(DEVICE)
        g2 = g2.to(DEVICE)
        optimizer.zero_grad()

        pred = model(g, g2)

        loss = criterion(pred, g.rxn_trg)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            mae = mae_loss(pred, g.rxn_trg)
            training_loss.append(mae)
        
        # print(mae, pred[:2], g.rxn_trg[:2])

    return np.mean(training_loss)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, default="100")
    parser.add_argument("-cv", type=str, default="1")
    args = parser.parse_args()

    CONFIG_PATH = "config/"
    config = configparser.ConfigParser()
    CONFIG_NAME = "config_" + str(args.config) + ".ini"
    print(CONFIG_PATH + CONFIG_NAME)
    config.read(CONFIG_PATH + CONFIG_NAME)
    print({section: dict(config[section]) for section in config.sections()})

    LR_FACTOR = float(config["PARAMS"]["LR_FACTOR"])
    LR_STEP_SIZE = int(config["PARAMS"]["LR_STEP_SIZE"])
    N_KERNELS = int(config["PARAMS"]["N_KERNELS"])
    POOLING_HEADS = int(config["PARAMS"]["POOLING_HEADS"])
    D_MLP = int(config["PARAMS"]["D_MLP"])
    D_KERNEL = int(config["PARAMS"]["D_KERNEL"])
    D_EMBEDDING = int(config["PARAMS"]["D_EMBEDDING"])
    BATCH_SIZE = int(config["PARAMS"]["BATCH_SIZE"])
    SPLIT = str(config["PARAMS"]["SPLIT"])
    ELN = str(config["PARAMS"]["ELN"])
    QML = int(config["PARAMS"]["QML"])
    GEOMETRY = int(config["PARAMS"]["GEOMETRY"])
    TARGET = str(config["PARAMS"]["TARGET"])
    QML = True if QML >= 1 else False
    GEOMETRY = True if GEOMETRY >= 1 else False
    GRAPH_DIM = "edge_3d" if GEOMETRY >= 1 else "edge_2d"

    print(f"QML: {QML}, GEOMETRY: {GEOMETRY}")

    tran_ids, eval_ids, test_ids = get_rxn_ids(split=SPLIT, eln=ELN, testset="1")
    tran_ids += eval_ids
    tran_ids += test_ids

    train_data = DataLSF(rxn_ids=tran_ids, graph_dim=GRAPH_DIM, rxn_trg=TARGET)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = GraphTransformer(N_KERNELS, POOLING_HEADS, D_MLP, D_KERNEL, D_EMBEDDING, QML, GEOMETRY)
    model = model.to(DEVICE)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_parameters = sum([np.prod(e.size()) for e in model_parameters])
    print("\nmodel_parameters", model_parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FACTOR, weight_decay=1e-10)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=0.5, verbose=False)

    for epoch in range(1000):

        tr_l = train(model, optimizer, criterion, train_loader)
        scheduler.step()
        print(f"MAEs (Epoch = {epoch + 1}): {tr_l}")

        if epoch >= 999:
            torch.save(model.state_dict(), f"models/config_{args.config}_{args.cv}.pt")
            print(f"\nNew min. MAE -> Epoch: {epoch + 1}; MAE Loss (Train, Eval): {tr_l}\n")
            
