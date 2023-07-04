#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (©) 2023 Kenneth Atz, & Gisbert Schneider (ETH Zurich)

import configparser
import os

import torch

from lsfml.qml.qml_net import DeltaNetAtomic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HERE = os.path.abspath(os.path.dirname(__file__))


def get_model(gpu=True):
    """Returns loaded and initialized QML model. 

    :param gpu: Running model on GPU (True or False), defaults to True
    :type gpu: bool, optional
    :return: QML model
    :rtype: class
    """
    # Load config parameters
    config = configparser.ConfigParser()
    config.read(os.path.join(HERE, "config_14000.ini"))

    KERNEL_DIM = int(config["PARAMS"]["KERNEL_DIM"])
    KERNEL_NUM = int(config["PARAMS"]["KERNEL_NUM"])
    MLP_DIM = int(config["PARAMS"]["MLP_DIM"])
    MLP_NUM = int(config["PARAMS"]["MLP_NUM"])
    EDGE_DIM = int(config["PARAMS"]["EDGE_DIM"])
    OUTPUT_DIM = int(config["PARAMS"]["OUTPUT_DIM"])
    AGGR = str(config["PARAMS"]["AGGR"]).strip('"')
    FOURIER = int(config["PARAMS"]["FOURIER"])

    # Load model
    model = DeltaNetAtomic(
        embedding_dim=KERNEL_DIM,
        n_kernels=KERNEL_NUM,
        n_mlp=MLP_NUM,
        mlp_dim=MLP_DIM,
        n_outputs=OUTPUT_DIM,
        m_dim=EDGE_DIM,
        initialize_weights=True,
        fourier_features=FOURIER,
        aggr=AGGR,
    )

    model.load_state_dict(
        torch.load(
            os.path.join(HERE, "model1.pkl"),
            map_location=torch.device("cpu"),
        )
    )

    if gpu:
        model = model.to(DEVICE)
        print(f"QML model has been sent to {DEVICE}")
    else:
        model = model.to("cpu")
        print("QML model has been sent to cpu")

    return model


if __name__ == "__main__":

    get_model()
