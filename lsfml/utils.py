#! /usr/bin/env python
# -*- coding: utf-8
#
# Copyright (©) 2023 Kenneth Atz, & Gisbert Schneider (ETH Zurich)

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np
from collections import Counter
import torch.nn.functional as F
import os

UTILS_PATH = os.path.dirname(__file__)


def mae_loss(x, y):
    """Calculates the MAE loss.

    :param x: Predicted values.
    :type x: Tensor
    :param y: True values.
    :type y: Tensor
    :return: Calculated MAE
    :rtype: numpy.float64
    """
    return F.l1_loss(x, y).item()


def get_dict_for_embedding(list):
    """Creates a dictionary from a list of strings as keys and values form 0 to N, where N = len(list).

    :param list: List of strings.
    :type list:  list[str]
    :return: dictionary, mapping each string to an integer.
    :rtype: dirct[str] = int
    """

    list_dict = {}
    list_counter = Counter(list)

    for idx, x in enumerate(list_counter):
        list_dict[x] = idx

    return list_dict


def get_fp_from_smi(smi):
    """Calculates ECFP from SMILES-sting

    :param smi: SMILES string
    :type smi: str
    :return: ECFP fingerprint vector
    :rtype: np.ndarray
    """
    mol_no_Hs = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol_no_Hs)

    return np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=256))


HYBRIDISATIONS = [
    "SP3",
    "SP2",
    "SP",
    "UNSPECIFIED",
    "S",
]

AROMATOCITY = [
    "True",
    "False",
]

IS_RING = [
    "True",
    "False",
]

ATOMTYPES = [
    "H",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
]

QML_ATOMTYPES = [
    "X",
    "H",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
]
