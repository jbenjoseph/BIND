import json
import random
import math
from tqdm import tqdm
import logging
import torch
from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.data import Data
from torch.utils.data import Dataset

import loading
from data import BondType

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)

# Optimisation
length_cutoff = 2048

print("Loading main database")

main_database = json.loads(open("examples/bindingdb_sample.json").read())
all_smiles_pairs = dict()

for key in main_database.keys():
    sequence = main_database[key]["sequence"].upper()
    if sequence not in all_smiles_pairs:
        all_smiles_pairs[sequence] = []
    all_smiles_pairs[sequence].append(main_database[key]["smiles"])

all_smiles = list(set([x["smiles"] for x in main_database.values()]))


def get_graph(smiles):
    graph = loading.get_data(
        smiles, apply_paths=False, parse_cis_trans=False, unknown_atom_is_dummy=True
    )
    x, a, e = loading.convert(
        *graph,
        bonds=[
            BondType.SINGLE,
            BondType.DOUBLE,
            BondType.TRIPLE,
            BondType.AROMATIC,
            BondType.NOT_CONNECTED,
        ],
    )

    x = torch.Tensor(x)
    a = dense_to_sparse(torch.Tensor(a))[0]
    e = torch.Tensor(e)

    # Given an xae
    graph = Data(x=x, edge_index=a, edge_features=e)
    return graph


smiles_graphs = dict()

for allowed_smiles in tqdm(all_smiles):
    try:
        graph = get_graph(allowed_smiles)
        if graph.edge_index.shape[1] == 0:
            continue
        smiles_graphs[allowed_smiles] = graph
    except Exception as e:
        continue

allowed_smiles_set = set(smiles_graphs.keys())
allowed_smiles_list = list(allowed_smiles_set)

current_set = [x for x in main_database.items() if x[1]["smiles"] in allowed_smiles_set]

for key in main_database.keys():
    main_database[key]["length"] = len(main_database[key]["sequence"])

print("Reformatting main database")

random.Random(0).shuffle(current_set)

train_set = current_set[: math.floor(0.9 * len(current_set))]
validation_set = current_set[
    math.floor(0.9 * len(current_set)) : math.floor(0.92 * len(current_set))
]
test_set = current_set[math.floor(0.92 * len(current_set)) :]

train_dataset_dict = {x[0]: x[1] for x in train_set}
validation_dataset_dict = {x[0]: x[1] for x in validation_set}

train_range = [
    (x, y) for x, y in train_dataset_dict.items() if y["length"] < length_cutoff
]
validation_range = [
    (x, y) for x, y in validation_dataset_dict.items() if y["length"] < length_cutoff
]

print(len(train_range), "training datapoints loaded")
print(len(validation_range), "validation datapoints loaded")
print(len(allowed_smiles_list), "unique SMILES loaded")


class BindingDBDataset(Dataset):
    def __init__(
        self,
        data_range,
        smiles_graphs,
        allowed_smiles_list,
        all_smiles_pairs,
        is_train=True,
    ):
        self.data_range = data_range
        self.smiles_graphs = smiles_graphs
        self.allowed_smiles_list = allowed_smiles_list
        self.all_smiles_pairs = all_smiles_pairs
        self.is_train = is_train
        self.false_ratio = 0.5

    def __len__(self):
        return len(self.data_range)

    def __getitem__(self, idx):
        protein = self.data_range[idx]
        seq_info = protein[1]
        sequence = seq_info["sequence"].upper()
        smiles = seq_info["smiles"]

        binding_ligands_set = set(self.all_smiles_pairs[sequence])

        is_false_ligand = 0.0

        if random.random() < self.false_ratio:
            found = False
            max_tries = 100
            while not found and max_tries > 0:
                max_tries -= 1
                try:
                    random_smiles = random.choice(self.allowed_smiles_list)
                    if random_smiles in binding_ligands_set:
                        continue
                    smiles = random_smiles
                    found = True
                except Exception:
                    continue
            if found:
                is_false_ligand = 1.0

        graph = self.smiles_graphs[smiles]

        if is_false_ligand:
            ki, ic50, kd, ec50 = [-9999] * 4
        else:
            ki, ic50, kd, ec50 = [
                x if x is not None else -9999 for x in seq_info["log10_affinities"]
            ]

        sample = {
            "sequence": sequence,
            "graph": graph,
            "ki": ki,
            "ic50": ic50,
            "kd": kd,
            "ec50": ec50,
            "is_false_ligand": is_false_ligand,
        }

        return sample


train_dataset = BindingDBDataset(
    train_range, smiles_graphs, allowed_smiles_list, all_smiles_pairs, is_train=True
)
validation_dataset = BindingDBDataset(
    validation_range,
    smiles_graphs,
    allowed_smiles_list,
    all_smiles_pairs,
    is_train=False,
)
