import torch
import pandas as pd
import os
import numpy as np
from model_FM import FactorizationMachineModel
from model_GCN import FactorizationMachineModel_withGCN
from model_Popularity import Popularity_Recommender
from model_Random import RandomModel
from utilities import Popularity_Graphic
from train import testpartial, testfull, train_one_epoch
from build_dataset import CustomerArticleDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.sparse import identity
from torch_geometric.utils import from_scipy_sparse_matrix


if __name__ == '__main__':

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    logs_base_dir = "runHM_10000"
    dataset_path = "../data/"
    
    os.makedirs(logs_base_dir, exist_ok=True)

    tb_fm = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_FM/')
    
    full_dataset = CustomerArticleDataset(dataset_path, num_negatives_train=4, num_negatives_test=99)
    data_loader = DataLoader(full_dataset, batch_size=256, shuffle=True, num_workers=0)

    identity_matrix = identity(full_dataset.train_mat.shape[0])
    identity_matrix = identity_matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((identity_matrix.row, identity_matrix.col)).astype(np.int64))
    values = torch.from_numpy(identity_matrix.data)
    shape = torch.Size(identity_matrix.shape)

    identity_tensor = torch.sparse.FloatTensor(indices, values, shape)
    edge_idx, edge_attr = from_scipy_sparse_matrix(full_dataset.train_mat)

    model_FM = FactorizationMachineModel(full_dataset.field_dims[-1], 32).to(device)
    model_rand = RandomModel(full_dataset.field_dims)
    model_pop = Popularity_Recommender(full_dataset.train_mat)
    model_GCN = FactorizationMachineModel_withGCN(full_dataset.field_dims[-1], 64, identity_tensor.to(device),
                                              edge_idx.to(device), attention=False).to(device)
    model_GCN_att = FactorizationMachineModel_withGCN(full_dataset.field_dims[-1], 64, identity_tensor.to(device),
                                                      edge_idx.to(device), attention=True).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer_FM = torch.optim.Adam(params=model_FM.parameters(), lr=0.001)
    optimizer_GCN = torch.optim.Adam(params=model_GCN.parameters(), lr=0.001)
    optimizer_GCN_att = torch.optim.Adam(params=model_GCN_att.parameters(), lr=0.001)

    #Train the model
    tb = True
    topk = 10
    num_epochs=20
    metrics_model = pd.DataFrame(columns=['Dataset', 'Model', 'HR', 'NDCG', 'COVERAGE', 'GINI', 'NOVELTY', 'TESTSET'])

    for epoch_i in range(num_epochs):

        print(f'epoch {epoch_i}:')

        #FM
        train_loss_FM= train_one_epoch(model_FM, optimizer_FM, data_loader, criterion, device)
        hr, ndcg, cov, gini, dict_FM, nov = testpartial(model_FM, full_dataset, device, topk=topk)
        if epoch_i == num_epochs - 1:
            new_row = pd.Series(
                data={'Dataset': 'H&M', 'Model': 'FM', 'HR': hr, 'NDCG': ndcg, 'COVERAGE': cov, 'GINI': gini,
                      'NOVELTY': nov, 'TESTSET':'PARTIAL'})
            metrics_model = metrics_model.append(new_row, ignore_index=True)

        #GCN
        train_loss_GCN = train_one_epoch(model_GCN, optimizer_GCN, data_loader, criterion, device)
        hr, ndcg, cov, gini, dict_GCN, nov = testpartial(model_GCN, full_dataset, device, topk=topk)
        if epoch_i == num_epochs - 1:
            new_row = pd.Series(
                data={'Dataset': 'H&M', 'Model': 'GCN', 'HR': hr, 'NDCG': ndcg, 'COVERAGE': cov, 'GINI': gini,
                      'NOVELTY': nov, 'TESTSET': 'PARTIAL'})
            metrics_model = metrics_model.append(new_row, ignore_index=True)

        # GCN att
        train_loss_GCN_att = train_one_epoch(model_GCN_att, optimizer_GCN_att, data_loader, criterion, device)
        hr, ndcg, cov, gini, dict_GCN_att, nov = testpartial(model_GCN_att, full_dataset, device, topk=topk)
        if epoch_i == num_epochs - 1:
            new_row = pd.Series(
                data={'Dataset': 'H&M', 'Model': 'GCN_ATT', 'HR': hr, 'NDCG': ndcg, 'COVERAGE': cov, 'GINI': gini,
                      'NOVELTY': nov, 'TESTSET': 'PARTIAL'})
            metrics_model = metrics_model.append(new_row, ignore_index=True)

        print(f'training loss = {train_loss_GCN_att:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f}, COV@{topk} = '
              f'{cov:.4f}, GINI@{topk} = {gini:.4f}, NOV@{topk} = {nov:.4f} ')

        #Random
        hr, ndcg, cov, gini, dict_rand, nov = testpartial(model_rand, full_dataset, device, topk=topk)
        if epoch_i == num_epochs - 1:
            new_row = pd.Series(
                data={'Dataset': 'H&M', 'Model': 'Random', 'HR': hr, 'NDCG': ndcg, 'COVERAGE': cov, 'GINI': gini,
                      'NOVELTY': nov, 'TESTSET':'PARTIAL'})
            metrics_model = metrics_model.append(new_row, ignore_index=True)

        # Popularity
        hr, ndcg, cov, gini, dict_pop, nov = testpartial(model_pop, full_dataset, device, topk=topk)
        if epoch_i == num_epochs - 1:
            new_row = pd.Series(
                data={'Dataset': 'H&M', 'Model': 'Popularity', 'HR': hr, 'NDCG': ndcg, 'COVERAGE': cov, 'GINI': gini,
                      'NOVELTY': nov, 'TESTSET': 'PARTIAL'})
            metrics_model = metrics_model.append(new_row, ignore_index=True)

    print(metrics_model)

    Popularity_Graphic(dict_FM, 'FM MODEL')
    Popularity_Graphic(dict_GCN, 'GCN MODEL')
    Popularity_Graphic(dict_GCN_att, 'GCN MODEL')
    Popularity_Graphic(dict_rand, 'RANDOM MODEL')
    Popularity_Graphic(dict_pop, 'POPULARITY MODEL')


