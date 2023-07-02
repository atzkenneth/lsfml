import torch, h5py, random
import numpy as np


from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from lsfml.minisci.gml.pygdataset import Dataset

random.seed(2)
DATASET_NAME = "20230608_alkylation_rxndata_full_all_new"  # hte_only, plus_lit


def get_rxn_ids(
    data=f"../data/rxn_smi_dict_{DATASET_NAME}.pt",
    split="random",
    eln="ELN036496-147",
    testset="1",
):
    rxn_smi_dict = torch.load(data)
    print(f"Data set name: {data}")

    # Load all rxn keys
    rxn_ids = list(rxn_smi_dict.keys())
    random.shuffle(rxn_ids)

    # Define subset of rxn keys
    if split == "random":
        if testset == "1":
            tran_ids = rxn_ids[: int(len(rxn_ids) / 2)]
            eval_ids = rxn_ids[int(len(rxn_ids) / 4) * 3 :]
            test_ids = rxn_ids[int(len(rxn_ids) / 2) : int(len(rxn_ids) / 4) * 3]
        if testset == "2":
            tran_ids = rxn_ids[: int(len(rxn_ids) / 2)]
            eval_ids = rxn_ids[int(len(rxn_ids) / 2) : int(len(rxn_ids) / 4) * 3]
            test_ids = rxn_ids[int(len(rxn_ids) / 4) * 3 :]
        if testset == "3":
            tran_ids = rxn_ids[int(len(rxn_ids) / 2) :]
            eval_ids = rxn_ids[: int(len(rxn_ids) / 4)]
            test_ids = rxn_ids[int(len(rxn_ids) / 4) : int(len(rxn_ids) / 2)]
        if testset == "4":
            tran_ids = rxn_ids[int(len(rxn_ids) / 2) :]
            eval_ids = rxn_ids[int(len(rxn_ids) / 4) : int(len(rxn_ids) / 2)]
            test_ids = rxn_ids[: int(len(rxn_ids) / 4)]
    elif split == "eln":
        rxn_ids_train = [x for x in rxn_ids if eln not in x]
        tran_ids = rxn_ids_train[int(len(rxn_ids_train) / 3) :]
        eval_ids = rxn_ids_train[: int(len(rxn_ids_train) / 3)]
        test_ids = [x for x in rxn_ids if eln in x]

    # print(f"\nDataset split initialized:")
    # print(f"Chosen testset (1-4): {testset}")
    # print(f"Chosen split (random or eln): {split}")

    return tran_ids, eval_ids, test_ids


class DataLSF(Dataset):
    def __init__(
        self,
        rxn_ids,
        data=f"../data/lsf_rxn_conditions_{DATASET_NAME}.h5",
        data_substrates=f"../data/lsf_rxn_substrate_{DATASET_NAME}.h5",
        data_acids=f"../data/lsf_rxn_carbacids_{DATASET_NAME}.h5",
        rxn_smi_dict=f"../data/rxn_smi_dict_{DATASET_NAME}.pt",
        graph_dim="edge_3d",  # edge_2d of edge_3d
        conformers=["a", "b", "c", "d", "e"],
        rxn_trg="trg_yld",  # trg_yld trg_bin
    ):
        super(DataLSF, self).__init__()
        print("Number of rxn ids: ", len(rxn_ids))

        # Define Target
        self.graph_dim = graph_dim
        self.conformers = conformers
        self.rxn_trg = rxn_trg

        # Load data from h5 file
        self.h5f_cond = h5py.File(data)
        self.h5f_subs = h5py.File(data_substrates)
        self.h5f_acds = h5py.File(data_acids)

        # load rxn to smi dicht
        self.rxn_smi_dict = torch.load(rxn_smi_dict)

        # Load all rxn keys
        self.rxn_ids = rxn_ids

        # Generate dict (int to rxn keys)
        nums = list(range(0, len(self.rxn_ids)))
        self.idx2rxn = {}
        for x in range(len(self.rxn_ids)):
            self.idx2rxn[nums[x]] = self.rxn_ids[x]

        print(f"\nLoader initialized:")
        print(f"RXN Ids: {self.rxn_ids}")
        print(f"Number of reactions loaded: {len(self.rxn_ids)}")
        print(f"Chosen target (binary or mono): {self.rxn_trg }")
        print(f"Chosen graph_dim (edge_2d of edge_3d): {self.graph_dim}")

    def __getitem__(self, idx):
        # int to rxn_id
        rxn_id = self.idx2rxn[idx]

        # get smiles from rxn_id
        sbst_rxn, acid_rxn = [
            self.rxn_smi_dict[rxn_id][0],
            self.rxn_smi_dict[rxn_id][1],
        ]

        # pick random conformer
        conformer = random.choice(self.conformers)

        # Molecule
        atom_id_1 = np.array(self.h5f_subs[sbst_rxn][f"atom_id_1_{conformer}"])
        ring_id_1 = np.array(self.h5f_subs[sbst_rxn][f"ring_id_1_{conformer}"])
        hybr_id_1 = np.array(self.h5f_subs[sbst_rxn][f"hybr_id_1_{conformer}"])
        arom_id_1 = np.array(self.h5f_subs[sbst_rxn][f"arom_id_1_{conformer}"])
        crds_3d_1 = np.array(self.h5f_subs[sbst_rxn][f"crds_3d_1_{conformer}"])
        to_keep_1 = np.array(self.h5f_subs[sbst_rxn][f"to_keep_1_{conformer}"])

        atom_id_2 = np.array(self.h5f_acds[acid_rxn][f"atom_id_2_{conformer}"])
        ring_id_2 = np.array(self.h5f_acds[acid_rxn][f"ring_id_2_{conformer}"])
        hybr_id_2 = np.array(self.h5f_acds[acid_rxn][f"hybr_id_2_{conformer}"])
        arom_id_2 = np.array(self.h5f_acds[acid_rxn][f"arom_id_2_{conformer}"])
        crds_3d_2 = np.array(self.h5f_acds[acid_rxn][f"crds_3d_2_{conformer}"])

        # Edge IDs with desired dimension
        edge_index_1 = np.array(self.h5f_subs[sbst_rxn][f"{self.graph_dim}_1_{conformer}"])
        edge_index_2 = np.array(self.h5f_acds[acid_rxn][f"{self.graph_dim}_2_{conformer}"])

        # Conditions
        rgt_eq = np.array([self.h5f_cond[rxn_id]["rgt_eq"]])
        sm2_eq = np.array([self.h5f_cond[rxn_id]["sm2_eq"]])
        conc_m = np.array([self.h5f_cond[rxn_id]["conc_m"]])
        tmp_de = np.array([self.h5f_cond[rxn_id]["tmp_de"]])
        hours_ = np.array([self.h5f_cond[rxn_id]["hours_"]])
        scale_ = np.array([self.h5f_cond[rxn_id]["scale_"]])
        cat_eq = np.array([self.h5f_cond[rxn_id]["cat_eq"]])
        add_eq = np.array([self.h5f_cond[rxn_id]["add_eq"]])
        sol_f1 = np.array([self.h5f_cond[rxn_id]["sol_f1"]])
        sol_f2 = np.array([self.h5f_cond[rxn_id]["sol_f2"]])
        cndtns = np.concatenate(
            (rgt_eq, sm2_eq, conc_m, tmp_de, hours_, scale_, cat_eq, add_eq, sol_f1, sol_f2), axis=1
        )

        rea_id = np.array([self.h5f_cond[rxn_id]["rea_id"]])
        so1_id = np.array([self.h5f_cond[rxn_id]["so1_id"]])
        so2_id = np.array([self.h5f_cond[rxn_id]["so2_id"]])
        cat_id = np.array([self.h5f_cond[rxn_id]["cat_id"]])
        add_id = np.array([self.h5f_cond[rxn_id]["add_id"]])
        atm_id = np.array([self.h5f_cond[rxn_id]["atm_id"]])

        # Tragets
        rxn_trg = np.array(self.h5f_cond[rxn_id][self.rxn_trg])

        # print("here", sbst_rxn, acid_rxn, rxn_id, rxn_trg, conformer)

        num_nodes_1 = torch.LongTensor(atom_id_1).size(0)
        num_nodes_2 = torch.LongTensor(atom_id_2).size(0)

        graph_data = Data(
            atom_id=torch.LongTensor(atom_id_1),
            ring_id=torch.LongTensor(ring_id_1),
            hybr_id=torch.LongTensor(hybr_id_1),
            to_keep=torch.LongTensor(to_keep_1),
            arom_id=torch.LongTensor(arom_id_1),
            condtns=torch.FloatTensor(cndtns),
            crds_3d=torch.FloatTensor(crds_3d_1),
            rxn_trg=torch.FloatTensor(rxn_trg),
            edge_index=torch.LongTensor(edge_index_1),
            num_nodes=num_nodes_1,
            rea_id=torch.LongTensor(rea_id),
            so1_id=torch.LongTensor(so1_id),
            so2_id=torch.LongTensor(so2_id),
            cat_id=torch.LongTensor(cat_id),
            add_id=torch.LongTensor(add_id),
            atm_id=torch.LongTensor(atm_id),
        )

        graph_data2 = Data(
            atom_id=torch.LongTensor(atom_id_2),
            ring_id=torch.LongTensor(ring_id_2),
            hybr_id=torch.LongTensor(hybr_id_2),
            arom_id=torch.LongTensor(arom_id_2),
            crds_3d=torch.FloatTensor(crds_3d_2),
            edge_index=torch.LongTensor(edge_index_2),
            num_nodes=num_nodes_2,
        )

        return graph_data, graph_data2

    def __len__(self):
        return len(self.rxn_ids)


if __name__ == "__main__":
    tran_ids, eval_ids, test_ids = get_rxn_ids()
    print("Training set: ", len(tran_ids))
    train_data = DataLSF(rxn_ids=tran_ids)
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=2)
    print("Training set: ", len(train_data))

    for e in range(4):
        print("NEW")
        # for g in tqdm(train_loader):
        for g, g2 in train_loader:
            print(g.rea_id, g.so1_id, g.so2_id, g.cat_id, g.add_id, g.atm_id)
            print(g.condtns)
            # print("")
            # print("g.arom_id", g.arom_id)
            # print("g.atom_id", g.atom_id)
            # print("g.to_keep", g.to_keep)
            # print("g.ring_id)", g.ring_id.size())
            # print("g.hybr_id)", g.hybr_id.size())
            # print("g.arom_id)", g.arom_id.size())
            # print("g.condtns", g.condtns.size())
            # print("g.charges)", g.charges.size(), g.charges)
            # print("g.ecfp_fp)", g.ecfp_fp.size())
            # print("g.crds_3d)", g.crds_3d.size())
            # print("g.rxn_trg)", g.rxn_trg.size(), g.rxn_trg)
            pass
