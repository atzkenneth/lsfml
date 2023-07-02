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

t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
n = torch.cuda.get_device_name(0)
print(f"GPU name: {n}, CUDA: total: {t / (10**9)} GB, reserved: {r / (10**9)} GB, allocated: {a / (10**9)} GB")


def train(
    model,
    optimizer,
    criterion,
    train_loader,
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


def eval(
    model,
    eval_loader,
):
    model.eval()
    eval_loss = []

    preds = []
    ys = []

    with torch.no_grad():
        for g, g2 in eval_loader:
            g = g.to(DEVICE)
            g2 = g2.to(DEVICE)
            pred = model(g, g2)
            mae = mae_loss(pred, g.rxn_trg)
            eval_loss.append(mae)
            ys.append(g.rxn_trg)
            preds.append(pred)
            # print(mae, pred, g.rxn_trg)

    return np.mean(eval_loss), ys, preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, default="100")
    parser.add_argument("-cv", type=str, default="1")
    parser.add_argument("-testset", type=str, default="1")
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
    GEOMETRY = int(config["PARAMS"]["GEOMETRY"])
    TARGET = str(config["PARAMS"]["TARGET"])
    GEOMETRY = True if GEOMETRY >= 1 else False
    GRAPH_DIM = "edge_3d" if GEOMETRY >= 1 else "edge_2d"

    print(f"GEOMETRY: {GEOMETRY}")

    tran_ids, eval_ids, test_ids = get_rxn_ids(split=SPLIT, eln=ELN, testset=args.testset)
    # tran_ids += eval_ids

    train_data = DataLSF(rxn_ids=tran_ids, graph_dim=GRAPH_DIM, rxn_trg=TARGET)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    eval_data = DataLSF(rxn_ids=eval_ids, graph_dim=GRAPH_DIM, rxn_trg=TARGET)
    eval_loader = DataLoader(eval_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test_data = DataLSF(rxn_ids=test_ids, graph_dim=GRAPH_DIM, rxn_trg=TARGET)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = GraphTransformer(
        n_kernels=N_KERNELS,
        pooling_heads=POOLING_HEADS,
        mlp_dim=D_MLP,
        kernel_dim=D_KERNEL,
        embeddings_dim=D_EMBEDDING,
        geometry=True,
    )
    model = model.to(DEVICE)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_parameters = sum([np.prod(e.size()) for e in model_parameters])
    print("\nmodel_parameters", model_parameters)

    # opt_params = list(model.parameters()) + list(fnn.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FACTOR, weight_decay=1e-10)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=0.5, verbose=False)

    min_mae = 100
    tr_losses = []
    ev_losses = []
    te_losses = []

    for epoch in range(1000):
        tr_l = train(model, optimizer, criterion, train_loader)
        # ev_l = train(model, optimizer, criterion, eval_loader) # early stopping might not be necessary
        ev_l, ev_ys, ev_pred = eval(model, eval_loader)
        te_l, te_ys, te_pred = eval(model, test_loader)

        tr_losses.append(tr_l)
        ev_losses.append(ev_l)
        te_losses.append(te_l)

        scheduler.step()
        print(f"MAEs (Epoch = {epoch + 1}): {tr_l}, {ev_l}, {te_l}")
        # print(f"MAEs (Epoch = {epoch + 1}): {tr_l}, {te_l}")

        ys = [item for sublist in te_ys for item in sublist]
        pred = [item for sublist in te_pred for item in sublist]

        if epoch >= 20:
            if ev_l <= min_mae:
                min_mae = ev_l
                torch.save(
                    model.state_dict(),
                    f"models/config_{args.config}_{args.cv}_{args.testset}.pt",
                )
                print(f"\nNew min. MAE -> Epoch: {epoch + 1}; MAE Loss (Train, Eval): {tr_l}, {ev_l}, {te_l}\n")
                # print(f"\nNew min. MAE -> Epoch: {epoch + 1}; MAE Loss (Train, Eval): {tr_l}, {te_l}\n")
                ys_saved = ys
                pred_saved = pred

            torch.save(
                [tr_losses, ev_losses, ys_saved, pred_saved, te_losses],
                f"results/config_{args.config}_{args.cv}_{args.testset}.pt",
            )
