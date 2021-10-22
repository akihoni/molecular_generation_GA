import pandas as pd

from rdkit import Chem
from rdkit.Chem import Recap

mol_path = './data/20210802_Dataset_C-1_and_C-4.csv'
# frag_csv_path = './data/frag_lib.csv'
frag_smi_path = './data/frag_lib_takasago.smi'

data = pd.read_csv(mol_path)

mol = data['SMILES'].map(lambda x: Chem.MolFromSmiles(x))
frag_r_mol = mol.map(lambda x: Recap.RecapDecompose(x) if x else 0)
frag_c = frag_r_mol.map(lambda x: list(x.children.keys()) if x != 0 else 0)
frag_c = pd.DataFrame(frag_c)
frag_c.columns = ['frag']

frag_counter = {}

for _, row in frag_c.iterrows():
    tmp_r = row['frag']
    if tmp_r != 0:
        for i in tmp_r:
            if i in frag_counter:
                frag_counter[i] += 1
            else:
                frag_counter[i] = 1

# print('frag_counter =', len(frag_counter))


def count_occurrence(frag):
    # print('total =', len(fgm))
    res = {}
    for key, value in frag.items():
        if value > 1:
            res[key] = value
    return res


frag_twice = count_occurrence(frag_counter)

frag_list = list(frag_twice.keys())
frag = pd.DataFrame(frag_list)
frag.columns = ['smiles']
# frag.to_csv(frag_csv_path, index=None)

# save fragments as smi file
frag.to_csv(frag_smi_path, index=None, header=None)

print('success!')
