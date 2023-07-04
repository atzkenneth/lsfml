#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (©) 2023 Kenneth Atz (ETH Zurich)


import argparse
import configparser
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from lsfml.experimental.net import EGNN, FNN, GraphTransformer
from lsfml.experimental.net_utils import DataLSF, get_rxn_ids
from lsfml.utils import mae_loss, UTILS_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUB_DATA = os.path.join(UTILS_PATH, "data/experimental_substrates.h5")
RXN_DATA = os.path.join(UTILS_PATH, "data/experimental_rxndata.h5")


def train(
    model,
    optimizer,
    criterion,
    train_loader,
):
    """Train loop.

    :param model: Model
    :type model: class
    :param optimizer: Optimizer
    :type optimizer: class
    :param criterion: Loss
    :type criterion: class
    :param train_loader: Data loader
    :type train_loader: torch_geometric.loader.dataloader.DataLoader
    :return: RMSE Loss
    :rtype: numpy.float64
    """
    model.train()
    training_loss = []

    for g in train_loader:
        g = g.to(DEVICE)
        optimizer.zero_grad()

        pred = model(g)

        loss = criterion(pred, g.rxn_trg)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mae = mae_loss(pred, g.rxn_trg)
            training_loss.append(mae)

    return np.mean(training_loss)


def eval(
    model,
    eval_loader,
):
    """Validation & test loop.

    :param model: Model
    :type model: class
    :param eval_loader: Data loader
    :type eval_loader: torch_geometric.loader.dataloader.DataLoader
    :return: tuple including essential information to quantify network perfromance such as MAE, predirctions, labels etc.
    :rtype: tuple
    """

    model.eval()
    eval_loss = []

    preds = []
    ys = []

    with torch.no_grad():
        for g in eval_loader:
            g = g.to(DEVICE)
            pred = model(g)
            mae = mae_loss(pred, g.rxn_trg)
            eval_loss.append(mae)
            ys.append(g.rxn_trg)
            preds.append(pred)

    return np.mean(eval_loss), ys, preds


if __name__ == "__main__":
    # python train.py -config 420 -mode a -cv 1 -testset 1 -early_stop 0

    # Make Folders for Results and Models
    os.makedirs("results/", exist_ok=True)
    os.makedirs("models/", exist_ok=True)

    # Read Passed Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, default="100")
    parser.add_argument("-mode", type=str, default="a")
    parser.add_argument("-cv", type=str, default="1")
    parser.add_argument("-testset", type=str, default="1")
    parser.add_argument("-early_stop", type=int, default=1)
    args = parser.parse_args()

    # Define Configuration form Model and Dataset
    config = configparser.ConfigParser()
    CONFIG_PATH = os.path.join(UTILS_PATH, f"config/config_{str(args.config)}.ini")
    config.read(CONFIG_PATH)
    print({section: dict(config[section]) for section in config.sections()})
    early_stop = True if args.early_stop >= 1 else False

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
    TARGET = str(config["PARAMS"]["TARGET"])
    QML = int(config["PARAMS"]["QML"])
    GEOMETRY = int(config["PARAMS"]["GEOMETRY"])
    FINGERPRINT = str(config["PARAMS"]["FINGERPRINT"])
    QML = True if QML >= 1 else False
    GEOMETRY = True if GEOMETRY >= 1 else False
    GRAPH_DIM = "edge_3d" if GEOMETRY >= 1 else "edge_2d"
    FINGERPRINT = FINGERPRINT if args.mode == "c" else None
    FP_DIM = 1024 if FINGERPRINT == "ecfp6_1" else 256
    CONFORMER_LIST = ["a", "b", "c", "d", "e"]

    # Initialize Model
    if args.mode == "a":
        model = GraphTransformer(
            n_kernels=N_KERNELS,
            pooling_heads=POOLING_HEADS,
            mlp_dim=D_MLP,
            kernel_dim=D_KERNEL,
            embeddings_dim=D_EMBEDDING,
            qml=QML,
            geometry=GEOMETRY,
        )
    elif args.mode == "b":
        model = EGNN(
            n_kernels=N_KERNELS,
            mlp_dim=D_MLP,
            kernel_dim=D_KERNEL,
            embeddings_dim=D_EMBEDDING,
            qml=QML,
            geometry=GEOMETRY,
        )
    elif args.mode == "c":
        model = FNN(
            fp_dim=FP_DIM,
            mlp_dim=D_MLP,
            kernel_dim=D_KERNEL,
            embeddings_dim=D_EMBEDDING,
        )

    model = model.to(DEVICE)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_parameters = sum([np.prod(e.size()) for e in model_parameters])
    print("\nmodel_parameters", model_parameters)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR_FACTOR,
        weight_decay=1e-10,
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LR_STEP_SIZE,
        gamma=0.5,
        verbose=False,
    )

    # Neural Netowork Training
    tr_losses = []
    ev_losses = []

    if early_stop:
        # Get Datasets
        tran_ids, eval_ids, test_ids = get_rxn_ids(
            data=RXN_DATA,
            split=SPLIT,
            eln=ELN,
            testset=args.testset,
        )
        train_data = DataLSF(
            rxn_ids=tran_ids,
            data=RXN_DATA,
            data_substrates=SUB_DATA,
            target=TARGET,
            graph_dim=GRAPH_DIM,
            fingerprint=FINGERPRINT,
            conformers=CONFORMER_LIST,
        )
        train_loader = DataLoader(
            train_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
        )
        eval_data = DataLSF(
            rxn_ids=eval_ids,
            data=RXN_DATA,
            data_substrates=SUB_DATA,
            target=TARGET,
            graph_dim=GRAPH_DIM,
            fingerprint=FINGERPRINT,
            conformers=CONFORMER_LIST,
        )
        eval_loader = DataLoader(
            eval_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
        )
        test_data = DataLSF(
            rxn_ids=test_ids,
            data=RXN_DATA,
            data_substrates=SUB_DATA,
            target=TARGET,
            graph_dim=GRAPH_DIM,
            fingerprint=FINGERPRINT,
            conformers=CONFORMER_LIST,
        )
        test_loader = DataLoader(
            test_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
        )

        # Training with Early Stopping
        min_mae = 1000

        for epoch in range(1000):
            # Training and Eval Loops
            tr_l = train(model, optimizer, criterion, train_loader)
            ev_l, ev_ys, ev_pred = eval(model, eval_loader)
            tr_losses.append(tr_l)
            ev_losses.append(ev_l)
            scheduler.step()

            if ev_l <= min_mae:
                # Define new min-loss
                min_mae = ev_l

                # Test model
                te_l, te_ys, te_pred = eval(model, test_loader)

                ys_saved = [item for sublist in te_ys for item in sublist]
                pred_saved = [item for sublist in te_pred for item in sublist]

                # Save Model and Save Loos + Predictions
                if SPLIT == "eln":
                    torch.save(model.state_dict(), f"models/config_{args.config}_{args.mode}_{TARGET}_{args.cv}.pt")
                    torch.save(
                        [tr_losses, ev_losses, ys_saved, pred_saved, ELN, TARGET],
                        f"results/config_{args.config}_{args.mode}_{args.cv}.pt",
                    )
                elif SPLIT == "random":
                    torch.save(
                        model.state_dict(),
                        f"models/config_{args.config}_{args.mode}_{TARGET}_{args.cv}_{args.testset}.pt",
                    )
                    torch.save(
                        [tr_losses, ev_losses, ys_saved, pred_saved, ELN, TARGET],
                        f"results/config_{args.config}_{args.mode}_{args.cv}_{args.testset}.pt",
                    )

    else:
        # Get Datasets
        tran_ids, eval_ids, test_ids = get_rxn_ids(
            data=RXN_DATA,
            split=SPLIT,
            eln=ELN,
            testset=args.testset,
        )
        tran_ids += eval_ids
        train_data = DataLSF(
            rxn_ids=tran_ids,
            data=RXN_DATA,
            data_substrates=SUB_DATA,
            target=TARGET,
            graph_dim=GRAPH_DIM,
            fingerprint=FINGERPRINT,
            conformers=CONFORMER_LIST,
        )
        train_loader = DataLoader(
            train_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
        )
        test_data = DataLSF(
            rxn_ids=test_ids,
            data=RXN_DATA,
            data_substrates=SUB_DATA,
            target=TARGET,
            graph_dim=GRAPH_DIM,
            fingerprint=FINGERPRINT,
            conformers=CONFORMER_LIST,
        )
        test_loader = DataLoader(
            test_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
        )

        # Training without Early Stopping
        for epoch in range(1000):
            # Training Loop
            tr_l = train(model, optimizer, criterion, train_loader)
            tr_losses.append(tr_l)
            scheduler.step()

            if epoch >= 999:
                # Test model
                te_l, te_ys, te_pred = eval(model, test_loader)

                ys_saved = [item for sublist in te_ys for item in sublist]
                pred_saved = [item for sublist in te_pred for item in sublist]

                # Save Model and Save Loos + Predictions
                if SPLIT == "eln":
                    torch.save(model.state_dict(), f"models/config_{args.config}_{args.mode}_{TARGET}_{args.cv}.pt")
                    torch.save(
                        [tr_losses, ev_losses, ys_saved, pred_saved, ELN, TARGET],
                        f"results/config_{args.config}_{args.mode}_{args.cv}.pt",
                    )
                elif SPLIT == "random":
                    torch.save(
                        model.state_dict(),
                        f"models/config_{args.config}_{args.mode}_{TARGET}_{args.cv}_{args.testset}.pt",
                    )
                    torch.save(
                        [tr_losses, ev_losses, ys_saved, pred_saved, ELN, TARGET],
                        f"results/config_{args.config}_{args.mode}_{args.cv}_{args.testset}.pt",
                    )
