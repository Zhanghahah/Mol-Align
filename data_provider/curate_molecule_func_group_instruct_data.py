"""
@Date: 2024/08/27/
@Author: cynthiazhang
@Email: cynthiazhang@sjtu.edu.cn

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os


def parse_csv(data_path):
    """
    may read csv data
    """
    cn_data = pd.read_csv(data_path)
    print(f"load data done.")
    return cn_data


def curate_molecule_data_from_csv(file_path):
    mol_property_save_path = '/'.join(file_path.split('/')[:-1])
    molecule_data = parse_csv(file_path)
    molecule_data.dropna(inplace=True)
    total_length = len(molecule_data)
    count = 0
    for idx in tqdm(range(total_length)):
        all_molecules = []
        all_molecules.extend(eval(molecule_data.iloc[idx]['canonical_reactants']))
        all_molecules.extend(eval(molecule_data.iloc[idx]['canonical_solvents']))
        all_molecules.extend(eval(molecule_data.iloc[idx]['canonical_reagents_and_catalysts']))
        all_molecules.extend(eval(molecule_data.iloc[idx]['canonical_products']))
        valid_mols, valid_descriptions = [], []
        tmp = dict()
        for mol_idx, mol in enumerate(all_molecules):
            func_groups = ""
            try:

                molecular_weight = eval(molecule_data.iloc[idx]['smi_to_descriptors'])[mol]['molecular weight']
                num_of_atoms = eval(molecule_data.iloc[idx]['smi_to_descriptors'])[mol]['number of atoms']
                num_of_H_bond_donors = eval(molecule_data.iloc[idx]['smi_to_descriptors'])[mol]['number of H-Bond donors']
                num_of_H_bond_acceptors = eval(molecule_data.iloc[idx]['smi_to_descriptors'])[mol]['number of H-Bond acceptors']
                tpsa = eval(molecule_data.iloc[idx]['smi_to_descriptors'])[mol]['topological polar surface area']
                logp = eval(molecule_data.iloc[idx]['smi_to_descriptors'])[mol]['logP']
                for fg, num in eval(molecule_data.iloc[idx]['smi_to_functional_groups'])[mol]:
                    func_groups += f"{num} {fg}, "

                instruct_data = (f"molecular_weight of this molecule is {molecular_weight}, its function groups as follows:{func_groups}"
                                 f"it has {num_of_atoms} number of atoms, {num_of_H_bond_donors} number of H-Bond donors,"
                                 f" {num_of_H_bond_acceptors} number of num_of_H_bond_acceptors, tpsa is {tpsa}, logP is {logp}.")
                valid_mols.append(mol)
                valid_descriptions.append(instruct_data)
            except:
                count += 1
                continue
        tmp['smiles'] = valid_mols
        tmp['descriptions'] = valid_descriptions
        df = pd.DataFrame(tmp)
        # save_csv_columns = ['smiles', 'descriptions']
        df.to_csv(os.path.join(mol_property_save_path, f"mols_property_descriptions.csv"), mode='a', index=False, sep=",",
                  header=False)
    print(count)


if __name__ == '__main__':
    mole_path = '/data2/zhangyu/molecule-data/func_group_data/test_woshi_year_ge_1990_if_ge_3.csv'
    curate_molecule_data_from_csv(mole_path)
