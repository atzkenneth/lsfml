from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import SimilarityMaps
from IPython.display import SVG
import io
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from rdkit.Chem import Draw

import argparse, torch, configparser, os, time
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from net import GraphTransformer
from lsfqml.minisci.preprocessh5 import get_3dG_from_smi
from lsfqml.clustering.cluster import SaveXlsxFromFrame, neutralize_atoms

ACIDS = [
    'O=C(O)C1CCCOC1',
    'O=C(OC(C)(C)C)N1CCCC1C(=O)O',
    'O=C(O)C1CCOCC1',
    'O=C(O)COCC=1C=CC=CC1',
    'O=C(O)C1OCCCC1',
    'O=C(OC(C)(C)C)N1CC(C(=O)O)C1',
    'O=C(OC(C)(C)C)N1CCC(C(=O)O)C1',
    'O=C(OC(C)(C)C)N1CCCC(C(=O)O)C1',
    'O=C(OC(C)(C)C)N1CCC(C(=O)O)CC1',
    'O=C(O)CCCC#C',
    'O=C(O)C1OCCC1',
    'O=C(O)CC1=CC=C(OC)C(OC)=C1',
    'O=C(O)CC',
    'O=C(O)C1CCC(F)(F)CC1',
    'O=C(O)CC=1C=CC=CC1',
    'O=C(O)C(NC(=O)C)C',
    'O=C(O)C(C)(C)C',
    'O=C(O)C1CCCC1',
    'O=C(O)C1CC=CCC1',
    'O=C(O)C1CCCCC1',
    'O=C1NCCC(C(=O)O)C1',
    'O=C(O)C1COC1',
    'O=C(O)C1CCC1',
    'O=C(O)C1COCC1',
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_filtered_smiles_and_fps(smi, ern):

    smi_passed = []
    ern_passed = []

    for i, sm in enumerate(smi):
        try: 
            sm = sm.split(".")[0]
            mol = Chem.MolFromSmiles(sm)
            mol_washed = neutralize_atoms(mol)
            smiles_washed = Chem.MolToSmiles(mol_washed)
            mwt = rdMolDescriptors.CalcExactMolWt(mol_washed)
            if ("n" in sm) and ("[n+]" not in sm) and (mwt >= 180):
                smi_passed.append(smiles_washed)
                ern_passed.append(ern[i])
            else:
                pass
        except:
            pass
        
    return smi_passed, ern_passed


def get_predictions(smi, ern, config_id):

    # Load config
    config = configparser.ConfigParser()
    config.read(f"config/config_{config_id}.ini")
    N_KERNELS = int(config["PARAMS"]["N_KERNELS"])
    D_MLP = int(config["PARAMS"]["D_MLP"])
    D_KERNEL = int(config["PARAMS"]["D_KERNEL"])
    D_EMBEDDING = int(config["PARAMS"]["D_EMBEDDING"])
    POOLING_HEADS = int(config["PARAMS"]["POOLING_HEADS"])
    QML = int(config["PARAMS"]["QML"])
    GEOMETRY = int(config["PARAMS"]["GEOMETRY"])
    QML = True if QML >= 1 else False
    GEOMETRY = True if GEOMETRY >= 1 else False
    GRAPH_DIM = "edge_3d" if GEOMETRY >= 1 else "edge_2d"

    # Load model
    model_path = "models/"
    model1 = GraphTransformer(N_KERNELS, POOLING_HEADS, D_MLP, D_KERNEL, D_EMBEDDING, QML, GEOMETRY)
    model1.load_state_dict(torch.load(f"{model_path}config_{config_id}_b_1_1.pt",map_location=torch.device("cpu")))
    model1 = model1.to(DEVICE)
    model2 = GraphTransformer(N_KERNELS, POOLING_HEADS, D_MLP, D_KERNEL, D_EMBEDDING, QML, GEOMETRY)
    model2.load_state_dict(torch.load(f"{model_path}config_{config_id}_b_1_2.pt",map_location=torch.device("cpu")))
    model2 = model2.to(DEVICE)
    model3 = GraphTransformer(N_KERNELS, POOLING_HEADS, D_MLP, D_KERNEL, D_EMBEDDING, QML, GEOMETRY)
    model3.load_state_dict(torch.load(f"{model_path}config_{config_id}_b_1_3.pt",map_location=torch.device("cpu")))
    model3 = model3.to(DEVICE)
    model4 = GraphTransformer(N_KERNELS, POOLING_HEADS, D_MLP, D_KERNEL, D_EMBEDDING, QML, GEOMETRY)
    model4.load_state_dict(torch.load(f"{model_path}config_{config_id}_b_1_4.pt",map_location=torch.device("cpu")))
    model4 = model4.to(DEVICE)

    # Loop over molecules
    smi_passed = []
    ern_passed = []
    totl_score = []

    mean_d6 = []
    mean_d5 = []
    mean_d4 = []
    mean_d3 = []
    mean_d2 = []
    mean_d1 = []
    mean_c6 = []
    mean_c5 = []
    mean_c4 = []
    mean_c3 = []
    mean_c2 = []
    mean_c1 = []
    mean_b6 = []
    mean_b5 = []
    mean_b4 = []
    mean_b3 = []
    mean_b2 = []
    mean_b1 = []
    mean_a6 = []
    mean_a5 = []
    mean_a4 = []
    mean_a3 = []
    mean_a2 = []
    mean_a1 = []

    stdv_d6 = []
    stdv_d5 = []
    stdv_d4 = []
    stdv_d3 = []
    stdv_d2 = []
    stdv_d1 = []
    stdv_c6 = []
    stdv_c5 = []
    stdv_c4 = []
    stdv_c3 = []
    stdv_c2 = []
    stdv_c1 = []
    stdv_b6 = []
    stdv_b5 = []
    stdv_b4 = []
    stdv_b3 = []
    stdv_b2 = []
    stdv_b1 = []
    stdv_a6 = []
    stdv_a5 = []
    stdv_a4 = []
    stdv_a3 = []
    stdv_a2 = []
    stdv_a1 = []

    cndtns = torch.FloatTensor([[6.0e+00, 2.0e+01, 1.6e-02]])

    for idx, sm in enumerate(tqdm(smi)):

        try: 
            preds_per_acid_mean = []
            preds_per_acid_stdv = []

            for acid in ACIDS:

                (
                    atom_id_1,
                    ring_id_1,
                    hybr_id_1,
                    arom_id_1,
                    charges_1,
                    edge_2d_1,
                    edge_3d_1,
                    crds_3d_1,
                    to_keep_1,
                ) = get_3dG_from_smi(sm, 0xf00d)

                (
                    atom_id_2,
                    ring_id_2,
                    hybr_id_2,
                    arom_id_2,
                    charges_2,
                    edge_2d_2,
                    edge_3d_2,
                    crds_3d_2,
                    to_keep_2,
                ) = get_3dG_from_smi(acid, 0xf00d)

                num_nodes_1 = torch.LongTensor(atom_id_1).size(0)
                num_nodes_2 = torch.LongTensor(atom_id_2).size(0)

                if GRAPH_DIM == "edge_3d":
                    edge_index_1 = edge_3d_1
                    edge_index_2 = edge_3d_2
                elif GRAPH_DIM == "edge_2d":
                    edge_index_1 = edge_2d_1
                    edge_index_2 = edge_2d_2

                graph_data = Data(
                    atom_id = torch.LongTensor(atom_id_1),
                    ring_id = torch.LongTensor(ring_id_1),
                    hybr_id = torch.LongTensor(hybr_id_1),
                    to_keep = torch.LongTensor(to_keep_1),
                    arom_id = torch.LongTensor(arom_id_1),
                    condtns = torch.FloatTensor(cndtns),
                    charges = torch.FloatTensor(charges_1),
                    crds_3d = torch.FloatTensor(crds_3d_1),
                    edge_index = torch.LongTensor(edge_index_1),
                    num_nodes=num_nodes_1,
                )

                graph_data2 = Data(
                    atom_id = torch.LongTensor(atom_id_2),
                    ring_id = torch.LongTensor(ring_id_2),
                    hybr_id = torch.LongTensor(hybr_id_2),
                    arom_id = torch.LongTensor(arom_id_2),
                    charges = torch.FloatTensor(charges_2),
                    crds_3d = torch.FloatTensor(crds_3d_2),
                    edge_index = torch.LongTensor(edge_index_2),
                    num_nodes=num_nodes_2,
                )

                graph_data = graph_data.to(DEVICE)
                graph_data2 = graph_data2.to(DEVICE)

                pred1 = model1(graph_data, graph_data2).detach().cpu().numpy()[0]
                pred2 = model2(graph_data, graph_data2).detach().cpu().numpy()[0]
                pred3 = model3(graph_data, graph_data2).detach().cpu().numpy()[0]
                pred4 = model4(graph_data, graph_data2).detach().cpu().numpy()[0]

                prd = int(np.mean(np.array([pred1, pred2, pred3, pred4])) * 100)
                std = int(np.std(np.array([pred1, pred2, pred3, pred4])) * 100)

                if prd >= 100:
                    prd = 100
                elif prd <= 0:
                    prd = 0

                # print(prd, std, sm, acid)

                preds_per_acid_mean.append(prd)
                preds_per_acid_stdv.append(std)
            
            # print(sum(v), preds_per_acid_mean)
            # print(preds_per_acid_stdv)

            smi_passed.append(sm)
            ern_passed.append(ern[idx])

            mean_d6.append(preds_per_acid_mean[0])
            mean_d5.append(preds_per_acid_mean[1])
            mean_d4.append(preds_per_acid_mean[2])
            mean_d3.append(preds_per_acid_mean[3])
            mean_d2.append(preds_per_acid_mean[4])
            mean_d1.append(preds_per_acid_mean[5])
            mean_c6.append(preds_per_acid_mean[6])
            mean_c5.append(preds_per_acid_mean[7])
            mean_c4.append(preds_per_acid_mean[8])
            mean_c3.append(preds_per_acid_mean[9])
            mean_c2.append(preds_per_acid_mean[10])
            mean_c1.append(preds_per_acid_mean[11])
            mean_b6.append(preds_per_acid_mean[12])
            mean_b5.append(preds_per_acid_mean[13])
            mean_b4.append(preds_per_acid_mean[14])
            mean_b3.append(preds_per_acid_mean[15])
            mean_b2.append(preds_per_acid_mean[16])
            mean_b1.append(preds_per_acid_mean[17])
            mean_a6.append(preds_per_acid_mean[18])
            mean_a5.append(preds_per_acid_mean[19])
            mean_a4.append(preds_per_acid_mean[20])
            mean_a3.append(preds_per_acid_mean[21])
            mean_a2.append(preds_per_acid_mean[22])
            mean_a1.append(preds_per_acid_mean[23])

            stdv_d6.append(preds_per_acid_stdv[0])
            stdv_d5.append(preds_per_acid_stdv[1])
            stdv_d4.append(preds_per_acid_stdv[2])
            stdv_d3.append(preds_per_acid_stdv[3])
            stdv_d2.append(preds_per_acid_stdv[4])
            stdv_d1.append(preds_per_acid_stdv[5])
            stdv_c6.append(preds_per_acid_stdv[6])
            stdv_c5.append(preds_per_acid_stdv[7])
            stdv_c4.append(preds_per_acid_stdv[8])
            stdv_c3.append(preds_per_acid_stdv[9])
            stdv_c2.append(preds_per_acid_stdv[10])
            stdv_c1.append(preds_per_acid_stdv[11])
            stdv_b6.append(preds_per_acid_stdv[12])
            stdv_b5.append(preds_per_acid_stdv[13])
            stdv_b4.append(preds_per_acid_stdv[14])
            stdv_b3.append(preds_per_acid_stdv[15])
            stdv_b2.append(preds_per_acid_stdv[16])
            stdv_b1.append(preds_per_acid_stdv[17])
            stdv_a6.append(preds_per_acid_stdv[18])
            stdv_a5.append(preds_per_acid_stdv[19])
            stdv_a4.append(preds_per_acid_stdv[20])
            stdv_a3.append(preds_per_acid_stdv[21])
            stdv_a2.append(preds_per_acid_stdv[22])
            stdv_a1.append(preds_per_acid_stdv[23])

            tot = int(sum(preds_per_acid_mean) / 24)
            totl_score.append(tot)
        
            # print(sm, ern[idx], tot)
        
        except:
            pass
    
    df = pd.DataFrame(
        {
            "smi_passed": smi_passed,
            "ern_passed": ern_passed,
            "totl_score": totl_score,
            "mean_d6": mean_d6, 
            "mean_d5": mean_d5, 
            "mean_d4": mean_d4, 
            "mean_d3": mean_d3, 
            "mean_d2": mean_d2, 
            "mean_d1": mean_d1, 
            "mean_c6": mean_c6, 
            "mean_c5": mean_c5, 
            "mean_c4": mean_c4, 
            "mean_c3": mean_c3, 
            "mean_c2": mean_c2, 
            "mean_c1": mean_c1, 
            "mean_b6": mean_b6, 
            "mean_b5": mean_b5, 
            "mean_b4": mean_b4, 
            "mean_b3": mean_b3, 
            "mean_b2": mean_b2, 
            "mean_b1": mean_b1, 
            "mean_a6": mean_a6, 
            "mean_a5": mean_a5, 
            "mean_a4": mean_a4, 
            "mean_a3": mean_a3, 
            "mean_a2": mean_a2, 
            "mean_a1": mean_a1, 
            "stdv_d6": stdv_d6, 
            "stdv_d5": stdv_d5, 
            "stdv_d4": stdv_d4, 
            "stdv_d3": stdv_d3, 
            "stdv_d2": stdv_d2, 
            "stdv_d1": stdv_d1, 
            "stdv_c6": stdv_c6, 
            "stdv_c5": stdv_c5, 
            "stdv_c4": stdv_c4, 
            "stdv_c3": stdv_c3, 
            "stdv_c2": stdv_c2, 
            "stdv_c1": stdv_c1, 
            "stdv_b6": stdv_b6, 
            "stdv_b5": stdv_b5, 
            "stdv_b4": stdv_b4, 
            "stdv_b3": stdv_b3, 
            "stdv_b2": stdv_b2, 
            "stdv_b1": stdv_b1, 
            "stdv_a6": stdv_a6, 
            "stdv_a5": stdv_a5, 
            "stdv_a4": stdv_a4, 
            "stdv_a3": stdv_a3, 
            "stdv_a2": stdv_a2, 
            "stdv_a1": stdv_a1, 

        }
    )

    df.sort_values(by="totl_score", ascending=False, inplace=True, ignore_index=True)

    out_name = f"labelled_data/lebels_{config_id}.xlsx"
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    df.to_csv(f"labelled_data/lebels_{config_id}.csv", index=False,)

    SaveXlsxFromFrame(
        df,
        out_name,
        molCols=["smi_passed",],
        size=(300, 300),
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, default="100")
    args = parser.parse_args()
    
    data = pd.read_excel(f"../data/Minisci_Substrates.xlsx", engine='openpyxl')
    smi = list(data[f"Canonical_Smiles"])
    ern = list(data[f"ERN"])
    print("Total number of moelcules: ", len(smi), len(ern))
    smi, ern = get_filtered_smiles_and_fps(smi, ern)
    print("Number of moelcules after filtering: ", len(smi), len(ern))
    # smi, ern = smi[12:24], ern[12:24]

    get_predictions(smi, ern, config_id=args.config)
    
    