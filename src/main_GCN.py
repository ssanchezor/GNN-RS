import torch
import numpy as np
import os
from model_GCN import FactorizationMachineModel_withGCN
from train import test,train_one_epoch
from build_dataset import CustomerArticleDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.sparse import identity
from torch_geometric.utils import from_scipy_sparse_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logs_base_dir = "runs2"
os.makedirs(logs_base_dir, exist_ok=True)

tb_fm = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_FM/')
tb_gcn = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_GCN/')
tb_gcn_attention = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_GCN_att/')

dataset_path = "../data/"
attention = False
full_dataset = CustomerArticleDataset(dataset_path, num_negatives_train=4, num_negatives_test=99)
data_loader = DataLoader(full_dataset, batch_size=256, shuffle=True, num_workers=0)


identity_matrix = identity(full_dataset.train_mat.shape[0])
identity_matrix = identity_matrix.tocoo().astype(np.float32)
indices = torch.from_numpy(np.vstack((identity_matrix.row, identity_matrix.col)).astype(np.int64))
values = torch.from_numpy(identity_matrix.data)
shape = torch.Size(identity_matrix.shape)

identity_tensor = torch.sparse.FloatTensor(indices, values, shape)
edge_idx, edge_attr = from_scipy_sparse_matrix(full_dataset.train_mat)


model = FactorizationMachineModel_withGCN(full_dataset.field_dims[-1], 64, identity_tensor.to(device),
                                          edge_idx.to(device), attention).to(device)

criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

#Train the model
tb = True
topk = 10
num_epochs=20

for epoch_i in range(num_epochs):
    train_loss = train_one_epoch(model, optimizer, data_loader, criterion, device)
    hr, ndcg = test(model, full_dataset, device, topk=topk)

    print(f'epoch {epoch_i}:')
    print(f'training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f} ')

    if tb:
        tb_fm.add_scalar('train/loss', train_loss, epoch_i)
        tb_fm.add_scalar('eval/HR@{topk}', hr, epoch_i)
        tb_fm.add_scalar('eval/NDCG@{topk}', ndcg, epoch_i)