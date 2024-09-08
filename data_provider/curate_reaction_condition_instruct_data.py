"""
@Date: 2024/08/27/
@Author: cynthiazhang
@Email: cynthiazhang@sjtu.edu.cn

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json


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
    tmp = dict()
    total_save_json = open(
        f"/data2/zhangyu/GraphGPT/reaction_condition_instruction_data.json",
        'w',
        encoding='utf-8')

    for idx in tqdm(range(total_length)):
        all_molecules = []
        canonical_reactants = eval(molecule_data.iloc[idx]['canonical_reactants'])
        canonical_solvents = eval(molecule_data.iloc[idx]['canonical_solvents'])
        canonical_reagents_and_catalysts = eval(molecule_data.iloc[idx]['canonical_reagents_and_catalysts'])
        canonical_products = eval(molecule_data.iloc[idx]['canonical_products'])
        valid_mols, valid_descriptions = [], []
        tmp = dict()

        try:

            # molecular_weight = eval(molecule_data.iloc[idx]['smi_to_descriptors'])[mol]['molecular weight']
            # num_of_atoms = eval(molecule_data.iloc[idx]['smi_to_descriptors'])[mol]['number of atoms']
            # num_of_H_bond_donors = eval(molecule_data.iloc[idx]['smi_to_descriptors'])[mol]['number of H-Bond donors']
            # num_of_H_bond_acceptors = eval(molecule_data.iloc[idx]['smi_to_descriptors'])[mol]['number of H-Bond acceptors']
            # tpsa = eval(molecule_data.iloc[idx]['smi_to_descriptors'])[mol]['topological polar surface area']
            # logp = eval(molecule_data.iloc[idx]['smi_to_descriptors'])[mol]['logP']
            # for fg, num in eval(molecule_data.iloc[idx]['smi_to_functional_groups'])[mol]:
            #     func_groups += f"{num} {fg}, "
            tmp['canonical_reactants'] = canonical_reactants
            tmp['canonical_solvents'] = canonical_solvents
            tmp['canonical_products'] = canonical_products
            tmp['canonical_reagents_and_catalysts'] = canonical_reagents_and_catalysts

            instruct_data_q = (
                f"Given a reaction, reactant molecules are {canonical_reactants}, product molecule is {canonical_products}, "
                f"we also introduce a sequence of graph tokens <graph> as the feature representation of molecules, "
                f"could you please generate or recommend the condition for this reaction.")

            instruct_data_a = f"{'.'.join(canonical_solvents)}.{'.'.join(canonical_reagents_and_catalysts)}"

            conversation = [{'from': 'human', 'value': instruct_data_q},
                            {'from': 'gpt', 'value': instruct_data_a}]
            tmp['conversation'] = conversation
            total_save_json.write(
                json.dumps(
                    tmp,
                    ensure_ascii=False
                ) + "\n"
            )

        except:
            continue


if __name__ == '__main__':
    mole_path = '/data2/zhangyu/molecule-data/func_group_data/test_woshi_year_ge_1990_if_ge_3.csv'
    curate_molecule_data_from_csv(mole_path)
