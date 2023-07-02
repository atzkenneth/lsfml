#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (©) 2022 Kenneth Atz (ETH Zurich), David F. Nippa (F. Hoffmann-La Roche Ltd) & Alex T. Müller (F. Hoffmann-La Roche Ltd)

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import argparse
import configparser
import os

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from lsfqml.lsfqml.publication.literature.regioselectivity.net import (
    Atomistic_EGNN,
)
from lsfqml.lsfqml.publication.literature.regioselectivity.net_utils import (
    DataLSF,
    get_rxn_ids,
)
from lsfqml.lsfqml.publication.utils import mae_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    rxn_ids = []
    atm_ids = []
    pt_trgs = []

    with torch.no_grad():

        for g in eval_loader:

            g = g.to(DEVICE)
            pred = model(g)
            mae = mae_loss(pred, g.rxn_trg)
            eval_loss.append(mae)
            ys.append(g.rxn_trg)
            preds.append(pred)
            rxn_ids.append(g.rxn_id)
            atm_ids.append(g.atom_id)
            pt_trgs.append(g.pot_trg)

    return np.mean(eval_loss), ys, preds, rxn_ids, pt_trgs


if __name__ == "__main__":

    # python train.py -config 141 -mode a -cv 1 -early_stop 0

    # Make Folders for Results and Models
    os.makedirs("results/", exist_ok=True)
    os.makedirs("models/", exist_ok=True)

    # Read Passed Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, default="100")
    parser.add_argument("-mode", type=str, default="a")
    parser.add_argument("-cv", type=str, default="1")
    parser.add_argument("-early_stop", type=int, default=1)
    args = parser.parse_args()

    # Define Configuration form Model and Dataset
    CONFIG_PATH = "config/"
    config = configparser.ConfigParser()
    CONFIG_NAME = "config_" + str(args.config) + ".ini"
    config.read(CONFIG_PATH + CONFIG_NAME)
    print({section: dict(config[section]) for section in config.sections()})
    early_stop = True if args.early_stop >= 1 else False

    LR_FACTOR = float(config["PARAMS"]["LR_FACTOR"])
    LR_STEP_SIZE = int(config["PARAMS"]["LR_STEP_SIZE"])
    N_KERNELS = int(config["PARAMS"]["N_KERNELS"])
    D_MLP = int(config["PARAMS"]["D_MLP"])
    D_KERNEL = int(config["PARAMS"]["D_KERNEL"])
    D_EMBEDDING = int(config["PARAMS"]["D_EMBEDDING"])
    BATCH_SIZE = int(config["PARAMS"]["BATCH_SIZE"])
    QML = int(config["PARAMS"]["QML"])
    GEOMETRY = int(config["PARAMS"]["GEOMETRY"])
    QML = True if QML >= 1 else False
    GEOMETRY = True if GEOMETRY >= 1 else False
    GRAPH_DIM = "edge_3d" if GEOMETRY >= 1 else "edge_2d"

    # Initialize Model
    model = Atomistic_EGNN(N_KERNELS, D_MLP, D_KERNEL, D_EMBEDDING, QML, GEOMETRY)
    model = model.to(DEVICE)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_parameters = sum([np.prod(e.size()) for e in model_parameters])
    print("\nmodel_parameters", model_parameters)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FACTOR, weight_decay=1e-10)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=0.5, verbose=False)

    # Neural Netowork Training
    tr_losses = []
    ev_losses = []

    if early_stop:

        # Get Datasets
        tran_ids, eval_ids, test_ids = get_rxn_ids()
        train_data = DataLSF(rxn_ids=tran_ids, graph_dim=GRAPH_DIM)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        eval_data = DataLSF(rxn_ids=eval_ids, graph_dim=GRAPH_DIM)
        eval_loader = DataLoader(eval_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        test_data = DataLSF(rxn_ids=test_ids, graph_dim=GRAPH_DIM)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        # Training with Early Stopping
        min_mae = 100

        for epoch in range(1000):

            tr_l = train(model, optimizer, criterion, train_loader)
            ev_l, ev_ys, ev_pred, ev_rxns, ev_pt_trgs = eval(model, eval_loader)
            tr_losses.append(tr_l)
            ev_losses.append(ev_l)
            scheduler.step()

            print(epoch, tr_l, ev_l)

            if ev_l <= min_mae:

                # Define new min-loss
                min_mae = ev_l

                # Test model
                te_l, te_ys, te_pred, te_rxns, te_pt_trgs = eval(model, test_loader)

                ys_saved = [float(item) for sublist in te_ys for item in sublist]
                pred_saved = [float(item) for sublist in te_pred for item in sublist]
                rxns_saved = [str(item) for sublist in te_rxns for item in sublist]
                pt_trgs_saved = [int(item) for sublist in te_pt_trgs for item in sublist]

                print(len(ys_saved), len(pred_saved), len(rxns_saved), len(pt_trgs_saved))

                # Save Model and Save Loos + Predictions
                torch.save(model.state_dict(), f"models/config_{args.config}_{args.cv}.pt")
                torch.save(
                    [tr_losses, ev_losses, ys_saved, pred_saved, rxns_saved, pt_trgs_saved],
                    f"results/config_{args.config}_{args.cv}.pt",
                )
    else:

        # Get Datasets
        tran_ids, eval_ids, test_ids = get_rxn_ids()
        tran_ids += eval_ids
        train_data = DataLSF(rxn_ids=tran_ids, graph_dim=GRAPH_DIM)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        test_data = DataLSF(rxn_ids=test_ids, graph_dim=GRAPH_DIM)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        # Training without Early Stopping
        for epoch in range(1000):

            tr_l = train(model, optimizer, criterion, train_loader)
            tr_losses.append(tr_l)
            scheduler.step()

            if epoch >= 999:

                # Test model
                te_l, te_ys, te_pred, te_rxns, te_pt_trgs = eval(model, test_loader)

                ys_saved = [float(item) for sublist in te_ys for item in sublist]
                pred_saved = [float(item) for sublist in te_pred for item in sublist]
                rxns_saved = [str(item) for sublist in te_rxns for item in sublist]
                pt_trgs_saved = [int(item) for sublist in te_pt_trgs for item in sublist]

                # Save Model and Save Loos + Predictions
                torch.save(model.state_dict(), f"models/config_{args.config}_{args.cv}.pt")
                torch.save(
                    [tr_losses, ev_losses, ys_saved, pred_saved, rxns_saved, pt_trgs_saved],
                    f"results/config_{args.config}_{args.cv}.pt",
                )
