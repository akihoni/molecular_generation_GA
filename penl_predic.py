import pandas as pd
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, DataStructs

from method.net import Net

# 预测各种数据结果

col = 'sweet'

smiles_file = 'data/random seed/mols_ga_sweet_n.csv'
model_path = 'model/%s_stat.pkl' % col
proba_res = 'data/random seed/mols_ga_sweet_n_%s.csv' % col


model = Net()
model.load_state_dict(torch.load(model_path))
model.to('cuda:0')

csv = pd.read_csv(smiles_file)
smiles = pd.DataFrame(csv['smiles'])
# smiles = (smiles.loc[:49999, ['mols']])
# smiles.columns = ['smiles']
# print(smiles.shape)
mol = smiles.applymap(lambda x: Chem.MolFromSmiles(x))
mol_clean = mol.applymap(lambda x: x if x else 0)
mol_clean = mol_clean[~mol_clean['smiles'].isin([0])]
mol_clean.index = [i for i in range(len(mol_clean))]
mol_clean_smiles = mol_clean.applymap(lambda x: Chem.MolToSmiles(x))
fp = mol_clean.applymap(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2))

# print(fp.shape)

fp_bit_arr = []
for i in fp.index:
    arr = np.zeros((0, ), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp.at[i, 'smiles'], arr)
    fp_bit_arr.append(arr)
fp_bit_array = np.array(fp_bit_arr)

# X_smiles.to_csv(filled_smiles_new, index=False)
X_fp_bit = pd.DataFrame(fp_bit_array)
# X_fp_bit.to_csv(filled_fp_new, index=False)

# 预测 filled fragment
mols_input = torch.FloatTensor(X_fp_bit.values)
mols_input = mols_input.cuda()
pred_mols = model(mols_input)
pred_mols.sigmoid_()
pred_mols = pred_mols.cpu()
# print(len(pred_mols))

pred_mols_arr = pred_mols.detach().numpy()
pred_mols_df = pd.DataFrame(pred_mols_arr)
pred_mols_df = pred_mols_df.applymap(lambda x: np.around(x, 4))

# id_df = pd.DataFrame([i for i in range(len(mol_clean_smiles))])

# print(mol_clean_smiles.shape)
# print(pred_mols_df.shape)

res_df = pd.concat([csv, pred_mols_df], axis=1)
# print(res_df.shape)
res_df.columns = ['id', 'smiles', 'fruity', col]
# res_df.columns = ['smiles', col]
ref_df = res_df.sort_values(by=col, ascending=False)

ref_df.index = [i for i in range(len(mol_clean_smiles))]
# print(ref_df)

# res_fin = pd.concat([id_df, ref_df], axis=1)
#res_fin.columns = ['id', 'mol', 'proba']

ref_df.to_csv(proba_res, index=None)
print('success!')
