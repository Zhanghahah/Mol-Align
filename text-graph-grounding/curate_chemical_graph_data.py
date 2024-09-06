import os
import sys
from rdkit import Chem
import numpy as np
import pandas as pd
from tqdm import tqdm
from graph_utils import mol_to_graph_data_obj_simple


def parse_csv(data_path):
    """
    may read csv data
    """
    cn_data = pd.read_csv(data_path, header=None)
    print(f"load data done.")
    return cn_data


def curate_graph_data_from_csv(csv_file_name):
    mol_graph_save_path = '/'.join(csv_file_name.split('/')[:-1])
    mol_and_descriptions_df = parse_csv(csv_file_name)
    total_length = len(mol_and_descriptions_df)
    mol_and_descriptions_df.columns = ['mol_name', 'descriptions']
    save_dict = dict()
    for idx in tqdm(range(total_length)):
        mol_name = mol_and_descriptions_df.iloc[idx]['mol_name']
        mol = Chem.MolFromSmiles(mol_name)
        if mol is not None:
            graph = mol_to_graph_data_obj_simple(mol)
            save_dict[mol_name] = graph

    np.save(f"{mol_graph_save_path}/woshi_test_mols_graph_data.npy", save_dict)





if __name__ == '__main__':
    mole_path = '/data2/zhangyu/molecule-data/func_group_data/mols_property_descriptions.csv'
    curate_graph_data_from_csv(mole_path)