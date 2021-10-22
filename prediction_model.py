import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, DataStructs

from method.net import Net

# 训练预测模型

model_path = './model/sts_stat.pkl'
mol_label_path = './data/sts_train.csv'

# hyper parameter
BATCH_SIZE = 10
EPOCH = 75
LR = 0.001
N_SPLITS = 10

# load data
mol_label = pd.read_csv(mol_label_path)
smiles = mol_label['SMILES']

# change the odor label here!
label = mol_label['fruity'].to_frame()

# transform smiles into fingerprint (ECFP4, 2048 bits)
mol = smiles.map(lambda x: Chem.MolFromSmiles(x))
fingerprint = mol.map(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2))
fp_bit_arr = []
for i in fingerprint.index:
    arr = np.zeros((0, ), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint.at[i], arr)
    fp_bit_arr.append(arr)
fp_bit_array = np.array(fp_bit_arr)
fp_bit = pd.DataFrame(fp_bit_array)


# k-fold cross validation
def get_k_fold_data(X, y):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=1)
    for train_index, test_index in kf.split(X, y):
        yield torch.FloatTensor(X.iloc[train_index].values), \
            torch.FloatTensor(y.iloc[train_index].values), \
            torch.FloatTensor(X.iloc[test_index].values), \
            torch.FloatTensor(y.iloc[test_index].values)


# stratified k-fold cross validation
def get_s_k_fold_data(X, y):
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=1)
    for train_index, test_index in kf.split(X, y):
        yield torch.FloatTensor(X.iloc[train_index].values), \
            torch.FloatTensor(y.iloc[train_index].values), \
            torch.FloatTensor(X.iloc[test_index].values), \
            torch.FloatTensor(y.iloc[test_index].values)


# load data by batch
def data_loader(X, y):
    torch_ds = TensorDataset(X, y)
    loader = DataLoader(
        dataset=torch_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    return loader


# train model
def train(X_train, y_train, X_test, y_test):
    # instantiate network model
    model = Net()
    # print(multi_label_net)

    # move to GPU
    if torch.cuda.is_available():
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()
        model = model.cuda()

    # set optimizer and loss function
    optimizer = torch.optim.ASGD(model.parameters(), lr=LR, weight_decay=0.05)
    loss_func = nn.BCEWithLogitsLoss()

    # load training data
    train_loader = data_loader(X_train, y_train)

    # draw plot for loss function
    px, py = [], []

    # train model
    for epoch in range(EPOCH):
        for _, (batch_X, batch_y) in enumerate(train_loader):
            if len(batch_X) == 1:
                continue
            out = model(batch_X)
            loss = loss_func(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 15 == 0:
            print('epoch =', epoch + 1, 'loss =', loss.item())

        px.append(epoch)
        py.append(loss.item())

    # test model
    with torch.no_grad():
        pred_train = model(X_train)
        pred_test = model(X_test)

        assert pred_test.shape == y_test.shape, \
            'shapes output: {}, and target: {} are not aligned.'. \
            format(pred_test.shape, y_test.shape)

        pred_train.sigmoid_()
        pred_test.sigmoid_()

        # calculate AUC score & ROC curve of the prediction model
        train_auc = roc_auc_score(y_train.cpu().numpy(), pred_train.cpu().numpy(), average='micro')
        test_auc = roc_auc_score(y_test.cpu().numpy(), pred_test.cpu().numpy(), average='micro')
        # test_acc = torch.round(output).eq(y_test).sum().cpu().numpy() / y_test.numel()

        print('auc score on train set:', round(train_auc, 4))
        print('auc score on test set:', round(test_auc, 4))
        # print('accuracy on test set: %d %%' % (100 * acc))

        return train_auc, test_auc, model


auc_train, auc_test = [], []
k = 1
for X_train, y_train, X_test, y_test in get_s_k_fold_data(fp_bit, label):
    print('No.%d' % k)
    # print(X_train.dtype, y_train.dtype)
    auc_train_e, auc_test_e, model = train(X_train, y_train, X_test, y_test)
    auc_train.append(auc_train_e)
    auc_test.append(auc_test_e)
    # acc.append(acc_e)
    if k == 2:
        print('saving model...')
        torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)
        print('model saved!')
    k += 1
auc_train_ave = sum(auc_train) / len(auc_train)
auc_test_ave = sum(auc_test) / len(auc_test)
# acc_t = sum(acc) / len(acc)
print('total auc score on train set:', round(auc_train_ave, 4))
print('total auc score on test set:', round(auc_test_ave, 4))
# print('total accuracy: %d %%' % (100 * acc_t))
