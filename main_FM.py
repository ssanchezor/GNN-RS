import torch
import os
from model_FM import FactorizationMachineModel
from train import test, train_one_epoch
from build_dataset import CustomerArticleDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logs_base_dir = "runs2"
os.makedirs(logs_base_dir, exist_ok=True)

tb_fm = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_FM/')
tb_gcn = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_GCN/')
tb_gcn_attention = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_GCN_att/')

dataset_path = "../data/"
full_dataset = CustomerArticleDataset(dataset_path, num_negatives_train=4, num_negatives_test=99)
data_loader = DataLoader(full_dataset, batch_size=256, shuffle=True, num_workers=0)

model = FactorizationMachineModel(full_dataset.field_dims[-1], 32).to(device)

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