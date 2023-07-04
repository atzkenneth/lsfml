#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (©) 2023 Kenneth Atz, & Gisbert Schneider (ETH Zurich)

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps

import configparser
import io
import os
import time

import numpy as np
import torch
from PIL import Image
from torch_geometric.data import Data

from lsfml.literature.regioselectivity.graph_mapping import (
    get_regioselectivity,
)
from lsfml.literature.regioselectivity.net import (
    Atomistic_EGNN,
)

ATOMTYPE_DICT = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4, "P": 5, "S": 6, "Cl": 7, "Br": 8, "I": 9}


def get_predictions(smiles, models):
    """Main function to apply regioselectivity predioction given a SMILES-string and a model id.

    :param smiles: SMILES-string
    :type smiles: str
    :param models: model id
    :type models: str
    """
    CONFIG_PATH = "config/"
    config = configparser.ConfigParser()

    for model_id in models:
        # Load model
        CONFIG_NAME = f"config_{model_id}.ini"
        config.read(CONFIG_PATH + CONFIG_NAME)

        N_KERNELS = int(config["PARAMS"]["N_KERNELS"])
        D_MLP = int(config["PARAMS"]["D_MLP"])
        D_KERNEL = int(config["PARAMS"]["D_KERNEL"])
        D_EMBEDDING = int(config["PARAMS"]["D_EMBEDDING"])
        QML = int(config["PARAMS"]["QML"])
        GEOMETRY = int(config["PARAMS"]["GEOMETRY"])
        QML = True if QML >= 1 else False
        GEOMETRY = True if GEOMETRY >= 1 else False

        model = Atomistic_EGNN(
            n_kernels=N_KERNELS,
            mlp_dim=D_MLP,
            kernel_dim=D_KERNEL,
            embeddings_dim=D_EMBEDDING,
            qml=QML,
            geometry=GEOMETRY,
        )

        model.load_state_dict(
            torch.load(
                f"models/config_{model_id}_1.pt",
                map_location=torch.device("cpu"),
            )
        )

        for j, smi in enumerate(smiles):
            name = f"regiosel_{j}"
            print(name, smi)
            preds_stat = []

            for k in range(3):
                seeds = [0xF00A, 0xF00B, 0xF00C, 0xF00E, 0xF00F, 0xF10D, 0xF20D, 0xF00D]
                pred_list = []

                for k in seeds:
                    (
                        atom_id,
                        ring_id,
                        hybr_id,
                        arom_id,
                        charges,
                        edge_2d,
                        edge_3d,
                        crds_3d,
                        pot_trg,
                        rxn_trg,
                    ) = get_regioselectivity(smi, k)

                    # Generate graph
                    num_nodes = torch.LongTensor(atom_id).size(0)

                    graph_data = Data(
                        atom_id=torch.LongTensor(atom_id),
                        ring_id=torch.LongTensor(ring_id),
                        hybr_id=torch.LongTensor(hybr_id),
                        arom_id=torch.LongTensor(arom_id),
                        charges=torch.FloatTensor(charges),
                        crds_3d=torch.FloatTensor(crds_3d),
                        rxn_trg=torch.FloatTensor(rxn_trg),
                        edge_index=torch.LongTensor(edge_3d),  # TODO: !!!
                        num_nodes=num_nodes,
                    )

                    pred = model(graph_data)
                    pred = [float(item) for item in pred]
                    pred_list.append(pred)

                preds_stat.append(np.mean(np.array(pred_list), axis=0))

            pred = np.mean(np.array(preds_stat), axis=0)
            pred_stds = np.std(np.array(preds_stat), axis=0)

            # Get image
            mol_no_Hs = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol_no_Hs)

            atomids = []
            pot_trg = []
            for idx, i in enumerate(mol.GetAtoms()):
                atomids.append(ATOMTYPE_DICT[i.GetSymbol()])
                nghbrs = [x.GetSymbol() for x in i.GetNeighbors()]
                if (i.GetSymbol() == "C") and ("H" in nghbrs):
                    pot_trg.append(1)
                else:
                    pot_trg.append(0)

            atomids = np.array(atomids)
            pot_trg = np.array(pot_trg)

            trth = rxn_trg
            pred = pred
            for i, x in enumerate(pred):
                if atom_id[i] == 1:
                    print(int(x * 10000) / 100, int(pred_stds[i] * 10000) / 100, hybr_id[i], atom_id[i])

            # Remove Hs for image
            trth2 = [int(-x) for i, x in enumerate(trth) if atomids[i] != 0]
            pred2 = [-x for i, x in enumerate(pred) if atomids[i] != 0]
            pot_trg = [int(x) for i, x in enumerate(pot_trg) if atomids[i] != 0]
            pred2 = [x * pot_trg[i] for i, x in enumerate(pred2)]
            trth2 = [float(x * pot_trg[i]) for i, x in enumerate(trth2)]
            RemoveHs = Chem.MolFromSmiles(smi)

            # Image
            d = Draw.MolDraw2DCairo(650, 650)
            d.SetFontSize(26)
            s = d.drawOptions()
            s.bondLineWidth = 6
            SimilarityMaps.GetSimilarityMapFromWeights(RemoveHs, list(pred2), draw2d=d)
            d.FinishDrawing()
            data = d.GetDrawingText()
            bio = io.BytesIO(data)
            img = Image.open(bio)
            img.save(f"regiosel_imgs/AVG_mol_{name}_{model_id}_pred.png")
            time.sleep(3)


if __name__ == "__main__":
    os.makedirs("regiosel_imgs", exist_ok=True)
    smiles = [
        "c1cc(OC)cc2cc(C(=O)OCC)[nH]c12",
        "c1c(Br)ccc2c1c(C(=O)C(C)(C)C)cn2S(=O)(=O)c1ccc(C)cc1",
        "c1ccc(C(=O)N(CCCCCC)CCCCCC)cc1Br",
        "O=C(N1CCOCC1)C2=C(Br)C=CC=C2",
        "O=C(OCC)c(ccc1)c2c1cc[nH]2",
        "CSC1=CC(C2OCCO2)=CC=C1",
    ]
    models = [
        "161",
    ]
    get_predictions(smiles, models)
