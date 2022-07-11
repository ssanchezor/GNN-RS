import torch
import numpy as np
import os
from model_GCN import FactorizationMachineModel_withGCN
from train import testpartial,train_one_epoch
from build_dataset import CustomerArticleDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.sparse import identity
from torch_geometric.utils import from_scipy_sparse_matrix
from utilities import info_model_report # mcanals for report


if __name__ == '__main__':

    # checking GPU...
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # setting up TensorBoard and data paths...
    logs_base_dir = "runHM"
    os.makedirs(logs_base_dir, exist_ok=True)
    tb_gcn_attention = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_GCN_att/')
    dataset_path = "../data/"

    # generating full dataset...
    full_dataset = CustomerArticleDataset(dataset_path, num_negatives_train=4, num_negatives_test=99)
    # splitting dataset into different batches...
    data_loader = DataLoader(full_dataset, batch_size=256, shuffle=True, num_workers=0)

    # GCN embedding preparation...
    identity_matrix = identity(full_dataset.train_mat.shape[0])
    identity_matrix = identity_matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((identity_matrix.row, identity_matrix.col)).astype(np.int64))
    values = torch.from_numpy(identity_matrix.data)
    shape = torch.Size(identity_matrix.shape)
    identity_tensor = torch.sparse.FloatTensor(indices, values, shape)
    edge_idx, edge_attr = from_scipy_sparse_matrix(full_dataset.train_mat)

    # generating the model...
    attention = True
    model = FactorizationMachineModel_withGCN(full_dataset.field_dims[-1], 64, identity_tensor.to(device),
                                            edge_idx.to(device), attention).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    # training the model...
    tb = True
    topk = 10 # 10 articles to be recommended
    num_epochs=20

    for epoch_i in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, data_loader, criterion, device)    
        hr, ndcg, cov, gini, dict_recomend, nov, l_info = testpartial(model, full_dataset, device, topk=topk)

        print(f'epoch {epoch_i}:')
        print(f'training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f}, COV@{topk} = {cov:.4f}, GINI@{topk} = {gini:.4f}, NOV@{topk} = {nov:.4f} ')

        # saving results in TensorBoard
        if tb:
            tb_gcn_attention.add_scalar(f'train/loss', train_loss, epoch_i)
            tb_gcn_attention.add_scalar(f'eval/HR@{topk}', hr, epoch_i)
            tb_gcn_attention.add_scalar(f'eval/NDCG@{topk}', ndcg, epoch_i)
            tb_gcn_attention.add_scalar(f'eval/COV@{topk}', cov, epoch_i)
            tb_gcn_attention.add_scalar(f'eval/GINI@{topk}', gini, epoch_i)
            tb_gcn_attention.add_scalar(f'eval/NOV@{topk}', nov, epoch_i)

    # saving training results...
    PATH = "FactorizationMachineModel_withGCN_Attention.pt"
    torch.save(model.state_dict(), PATH)

    # generating customized report...
    res_header=[f"HR@{topk}", f"NDCG@{topk}", f"COV@{topk}",f"GINI@{topk}",f"NOV@{topk}" ]
    res_values=[f"{hr:.4f}", f"{ndcg:.4f}", f"{cov:.4f}", f"{gini:.4f}", f"{nov:.4f}"  ]
    res_info=[res_header,res_values]
    info_model_report (model, dataset_path, res_info, l_info, \
            full_dataset, dict_recomend, title="GCN with Attention - Partial", topk=10 )
