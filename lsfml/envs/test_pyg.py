import rdkit
from rdkit import Chem

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.aggr import GraphMultisetTransformer
from torch_geometric.typing import Adj, Size, Tensor
from torch_geometric.utils.scatter import scatter

if __name__ == "__main__":
    print(torch_geometric.__version__)
    print(torch.__version__)
    print(rdkit.__version__)