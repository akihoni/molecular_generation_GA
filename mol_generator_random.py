# import datetime
# starttime = datetime.datetime.now()

from rdkit.Chem import AllChem
from rdkit import Chem
import torch
import pandas as pd
import numpy as np

from method.net import Net


frag_path = './data/frag_lib.smi'
res_path = './data/mols_rand_prob.csv'
model_path = './model/fruity_stat.pkl'

number_of_structures = 2000
PRED = 0.9

model = Net()
model.load_state_dict(torch.load(model_path))
model.to('cuda:0')
model.eval()

main_molecules = [molecule for molecule in Chem.SmilesMolSupplier(frag_path, delimiter='\t', titleLine=False)
                  if molecule is not None]
fragment_molecules = [molecule for molecule in Chem.SmilesMolSupplier(frag_path, delimiter='\t', titleLine=False)
                      if molecule is not None]
# fragment_molecules = [molecule for molecule in Chem.SDMolSupplier('fragments.sdf') if molecule is not None]

bond_list = [Chem.rdchem.BondType.UNSPECIFIED, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.QUADRUPLE, Chem.rdchem.BondType.QUINTUPLE,
             Chem.rdchem.BondType.HEXTUPLE, Chem.rdchem.BondType.ONEANDAHALF, Chem.rdchem.BondType.TWOANDAHALF,
             Chem.rdchem.BondType.THREEANDAHALF, Chem.rdchem.BondType.FOURANDAHALF, Chem.rdchem.BondType.FIVEANDAHALF,
             Chem.rdchem.BondType.AROMATIC, Chem.rdchem.BondType.IONIC, Chem.rdchem.BondType.HYDROGEN,
             Chem.rdchem.BondType.THREECENTER, Chem.rdchem.BondType.DATIVEONE, Chem.rdchem.BondType.DATIVE,
             Chem.rdchem.BondType.DATIVEL, Chem.rdchem.BondType.DATIVER, Chem.rdchem.BondType.OTHER,
             Chem.rdchem.BondType.ZERO]

prob = []
generated_structures = []
generated_structure_number = 0
while generated_structure_number < number_of_structures:
    selected_main_molecule_number = np.floor(
        np.random.rand(1) * len(main_molecules)).astype(int)[0]
    main_molecule = main_molecules[selected_main_molecule_number]
    # make adjacency matrix and get atoms for main molecule
    main_adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(main_molecule)
    for bond in main_molecule.GetBonds():
        main_adjacency_matrix[bond.GetBeginAtomIdx(
        ), bond.GetEndAtomIdx()] = bond_list.index(bond.GetBondType())
        main_adjacency_matrix[bond.GetEndAtomIdx(
        ), bond.GetBeginAtomIdx()] = bond_list.index(bond.GetBondType())
    main_atoms = []
    for atom in main_molecule.GetAtoms():
        main_atoms.append(atom.GetSymbol())

    r_index_in_main_molecule_old = [
        index for index, atom in enumerate(main_atoms) if atom == '*']
    for index, r_index in enumerate(r_index_in_main_molecule_old):
        modified_index = r_index - index
        atom = main_atoms.pop(modified_index)
        main_atoms.append(atom)
        tmp = main_adjacency_matrix[:,
                                    modified_index:modified_index + 1].copy()
        main_adjacency_matrix = np.delete(
            main_adjacency_matrix, modified_index, 1)
        main_adjacency_matrix = np.c_[main_adjacency_matrix, tmp]
        tmp = main_adjacency_matrix[modified_index:modified_index + 1, :].copy()
        main_adjacency_matrix = np.delete(
            main_adjacency_matrix, modified_index, 0)
        main_adjacency_matrix = np.r_[main_adjacency_matrix, tmp]
    r_index_in_main_molecule_new = [
        index for index, atom in enumerate(main_atoms) if atom == '*']

    r_bonded_atom_index_in_main_molecule = []
    for number in r_index_in_main_molecule_new:
        r_bonded_atom_index_in_main_molecule.append(
            np.where(main_adjacency_matrix[number, :] != 0)[0][0])
    r_bond_number_in_main_molecule = main_adjacency_matrix[
        r_index_in_main_molecule_new, r_bonded_atom_index_in_main_molecule]

    main_adjacency_matrix = np.delete(
        main_adjacency_matrix, r_index_in_main_molecule_new, 0)
    main_adjacency_matrix = np.delete(
        main_adjacency_matrix, r_index_in_main_molecule_new, 1)

    for i in range(len(r_index_in_main_molecule_new)):
        main_atoms.remove('*')
    main_size = main_adjacency_matrix.shape[0]

    selected_fragment_numbers = np.floor(np.random.rand(
        len(r_index_in_main_molecule_old)) * len(fragment_molecules)).astype(int)

    generated_molecule_atoms = main_atoms[:]
    generated_adjacency_matrix = main_adjacency_matrix.copy()
    for r_number_in_molecule in range(len(r_index_in_main_molecule_new)):
        fragment_molecule = fragment_molecules[selected_fragment_numbers[r_number_in_molecule]]
        fragment_adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(
            fragment_molecule)
        for bond in fragment_molecule.GetBonds():
            fragment_adjacency_matrix[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond_list.index(
                bond.GetBondType())
            fragment_adjacency_matrix[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond_list.index(
                bond.GetBondType())
        fragment_atoms = []
        for atom in fragment_molecule.GetAtoms():
            fragment_atoms.append(atom.GetSymbol())

        # integrate adjacency matrix
        r_index_in_fragment_molecule = fragment_atoms.index('*')

        r_bonded_atom_index_in_fragment_molecule = \
            np.where(fragment_adjacency_matrix[r_index_in_fragment_molecule, :] != 0)[
                0][0]
        if r_bonded_atom_index_in_fragment_molecule > r_index_in_fragment_molecule:
            r_bonded_atom_index_in_fragment_molecule -= 1

        fragment_atoms.remove('*')
        fragment_adjacency_matrix = np.delete(
            fragment_adjacency_matrix, r_index_in_fragment_molecule, 0)
        fragment_adjacency_matrix = np.delete(
            fragment_adjacency_matrix, r_index_in_fragment_molecule, 1)

        main_size = generated_adjacency_matrix.shape[0]
        generated_adjacency_matrix = np.c_[generated_adjacency_matrix, np.zeros(
            [generated_adjacency_matrix.shape[0], fragment_adjacency_matrix.shape[0]], dtype='int32')]
        generated_adjacency_matrix = np.r_[generated_adjacency_matrix, np.zeros(
            [fragment_adjacency_matrix.shape[0], generated_adjacency_matrix.shape[1]], dtype='int32')]

        generated_adjacency_matrix[r_bonded_atom_index_in_main_molecule[
            r_number_in_molecule], r_bonded_atom_index_in_fragment_molecule + main_size] = \
            r_bond_number_in_main_molecule[r_number_in_molecule]
        generated_adjacency_matrix[
            r_bonded_atom_index_in_fragment_molecule + main_size, r_bonded_atom_index_in_main_molecule[
                r_number_in_molecule]] = r_bond_number_in_main_molecule[r_number_in_molecule]
        generated_adjacency_matrix[main_size:,
                                   main_size:] = fragment_adjacency_matrix

        # integrate atoms
        generated_molecule_atoms += fragment_atoms

    # generate structures
    generated_molecule = Chem.RWMol()
    atom_index = []
    for atom_number in range(len(generated_molecule_atoms)):
        atom = Chem.Atom(generated_molecule_atoms[atom_number])
        molecular_index = generated_molecule.AddAtom(atom)
        atom_index.append(molecular_index)
    for index_x, row_vector in enumerate(generated_adjacency_matrix):
        for index_y, bond in enumerate(row_vector):
            if index_y <= index_x:
                continue
            if bond == 0:
                continue
            else:
                generated_molecule.AddBond(
                    atom_index[index_x], atom_index[index_y], bond_list[bond])

    generated_molecule = generated_molecule.GetMol()
    # generated_molecule = generated_molecule.updatePropertyCache()
    generated_molecule_smiles = Chem.MolToSmiles(generated_molecule)

    generated_mol = Chem.MolFromSmiles(generated_molecule_smiles)
    
    if not generated_mol:
        continue

    gene_mole_fp = AllChem.GetMorganFingerprintAsBitVect(generated_mol, 2)
    X = torch.FloatTensor(gene_mole_fp).cuda()
    X = X.unsqueeze(0)
    pred = model(X).sigmoid_().cpu().detach().numpy()[0][0]

    # filter: if prediction >= PRED, then output.
    if pred >= PRED and generated_molecule_smiles not in generated_structures:
        prob.append(np.around(pred, 4))
        generated_structures.append(generated_molecule_smiles)
        generated_structure_number += 1
        if (generated_structure_number + 1) % 100 == 0 or (generated_structure_number + 1) == number_of_structures:
            print(generated_structure_number + 1, '/', number_of_structures)
    else:
        continue


mols_df = pd.DataFrame(generated_structures)
prob_df = pd.DataFrame(prob)
res_df = pd.concat([mols_df, prob_df], axis=1)
res_df.columns = ['smiles', 'prob']
ref_df = res_df.sort_values(by='prob', ascending=False)
ref_df.index = [i for i in range(number_of_structures)]
id_df = pd.DataFrame([i for i in range(1, number_of_structures + 1)])
res_fin = pd.concat([id_df, ref_df], axis=1)
res_fin.columns = ['id', 'smiles', 'proba']
res_fin.to_csv(res_path, index=None)

# endtime = datetime.datetime.now()
# print((endtime - starttime).seconds)
