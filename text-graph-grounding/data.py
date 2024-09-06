from __future__ import division

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import repeat

from rdkit import Chem


import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from graph_utils import mol_to_graph_data_obj_simple


def parse_csv(data_path):
    """
    may read csv data
    """
    cn_data = pd.read_csv(data_path, header=None)
    print(f"load data done.")
    return cn_data

class MolGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):

        self.root = root
        self.transform = transform
        self.graph_file = os.path.join(self.root, "woshi_test_mols_graph_data.npy")
        self.text_file = os.path.join(self.root, "mols_property_descriptions.csv")
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        # self.graph = np.load(graph_file, allow_pickle=True)
        super(MolGraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.load_Graph_and_text()
        # self.text_df = parse_csv(text_file)
        # self.text_df.columns =  ['mol_name', 'descriptions']
        # self.mol_descriptions = self.text_df.to_dict(orient = 'records')

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'
    def process(self):
        mol_and_descriptions_df = parse_csv(self.text_file)
        total_length = len(mol_and_descriptions_df)
        mol_and_descriptions_df.columns = ['mol_name', 'descriptions']
        graph_list = []
        for idx in tqdm(range(total_length)):
            mol_name = mol_and_descriptions_df.iloc[idx]['mol_name']
            mol = Chem.MolFromSmiles(mol_name)
            if mol is not None:
                graph = mol_to_graph_data_obj_simple(mol)
                graph_list.append(graph)

        if self.pre_filter is not None:
            graph_list = [graph for graph in graph_list if self.pre_filter(graph)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(graph) for graph in graph_list]

        graphs, slices = self.collate(graph_list)
        torch.save((graphs, slices), self.processed_paths[0])
        return
    def load_Graph_and_text(self):
        self.graphs, self.slices = torch.load(self.processed_paths[0])
        mol_and_descriptions_df = parse_csv(self.text_file)
        mol_and_descriptions_df.columns = ['mol_name', 'descriptions']
        self.text_list = mol_and_descriptions_df["descriptions"].tolist()
        #
        # self.text_df = parse_csv(text_file)
        # self.text_df.columns = ['mol_name', 'descriptions']
        #
        # self.text_list = self.text_df["descriptions"].tolist()
        return
    def __len__(self):
        return len(self.text_list)
    def get(self, idx):
        text = self.text_list[idx]

        data = Data()
        for key in self.graphs.keys():
            item, slices = self.graphs[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return text, data


class DataHelper(Dataset):
    def __init__(self, edge_index, args, directed=False, transform=None):
        # self.num_nodes = len(node_list)
        self.transform = transform

        self.degrees = dict()
        self.node_set = set()
        self.neighs = dict()
        self.args = args

        idx, degree = np.unique(edge_index, return_counts=True)
        for i in range(idx.shape[0]):
            self.degrees[idx[i]] = degree[i].item()

        self.node_dim = idx.shape[0]
        print("lenth of dataset", self.node_dim)

        train_edge_index = edge_index
        self.final_edge_index = train_edge_index.T

        for i in range(self.final_edge_index.shape[0]):
            s_node = self.final_edge_index[i][0].item()
            t_node = self.final_edge_index[i][1].item()

            if s_node not in self.neighs:
                self.neighs[s_node] = []
            if t_node not in self.neighs:
                self.neighs[t_node] = []

            self.neighs[s_node].append(t_node)
            if not directed:
                self.neighs[t_node].append(s_node)

        # self.neighs = sorted(self.neighs)
        self.idx = idx

        print("len of neighs", len(self.neighs))
        # print(self.neighs)

    def __len__(self):
        return self.node_dim

    def __getitem__(self, idx):
        s_n = self.idx[idx].item()
        t_n = [np.random.choice(self.neighs[s_n], replace=True).item() for _ in range(self.args.neigh_num)]
        t_n = np.array(t_n)

        sample = {
            "s_n": s_n,  # e.g., 5424
            "t_n": t_n,  # e.g., 5427
            # 'neg_n': neg_n
        }

        if self.transform:
            sample = self.transform(sample)
        # print(sample)

        return sample


if __name__ == '__main__':
    graph_datasets = MolGraphDataset(root="/data2/zhangyu/molecule-data/func_group_data/")

