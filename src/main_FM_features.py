import torch
import os
from model_FM_features import FieldAwareFactorizationMachineModel
from train_features import testpartial, train_one_epoch_features
from build_dataset_features import CustomerArticleDataset
from torch.utils.data import DataLoader
from utilities import Popularity_Graphic
from torch.utils.tensorboard import SummaryWriter



if __name__ == '__main__':

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    logs_base_dir = "runHM_10000"
    dataset_path = "../data/"
    
    os.makedirs(logs_base_dir, exist_ok=True)

    tb_fm = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_FM/')
    
    full_dataset = CustomerArticleDataset(dataset_path, num_negatives_train=4, num_negatives_test=99)
    data_loader = DataLoader(full_dataset, batch_size=256, shuffle=True, num_workers=0)

    model = FieldAwareFactorizationMachineModel(full_dataset.field_dims, 32).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    #Train the model
    tb = True
    topk = 10
    num_epochs=10

    for epoch_i in range(num_epochs):
        train_loss = train_one_epoch_features(model, optimizer, data_loader, criterion, device)
        hr, ndcg, cov, gini, dict_FM_features, nov = testpartial(model, full_dataset, device, topk=topk)

        print(f'epoch {epoch_i}:')
        print(f'training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f}, COV@{topk} = '
              f'{cov:.4f}, NOV@{topk} = {nov:.4f} ')

        if tb:
            tb_fm.add_scalar('train/loss', train_loss, epoch_i)
            tb_fm.add_scalar(f'eval/HR@{topk}', hr, epoch_i)
            tb_fm.add_scalar(f'eval/NDCG@{topk}', ndcg, epoch_i)
            tb_fm.add_scalar(f'eval/COV@{topk}', cov, epoch_i)

    Popularity_Graphic(dict_FM_features, 'FM MODEL FEATURES')
    dict_DataSet_HM = dict(full_dataset.Popular_Items)
    Popularity_Graphic(dict_DataSet_HM, 'DATASET H&M')