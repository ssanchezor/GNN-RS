import torch
import os
from model_FM import FactorizationMachineModel
from train import train_one_epoch,testpartial
from build_dataset import CustomerArticleDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils_report import info_model_report # mcanals for report



if __name__ == '__main__':

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logs_base_dir = "runHM_10000"
    dataset_path = "../data/"
    
    os.makedirs(logs_base_dir, exist_ok=True)

    tb_fm = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_FM/')
    
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
        #hr, ndcg, coverage = test(model, full_dataset, device, topk=topk)
        hr, ndcg, cov, gini, dict_recommend, nov, l_info = testpartial(model, full_dataset, device, topk=topk)

        print(f'epoch {epoch_i}:')
        print(f'training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f}, COV@{topk} = {cov:.4f}, GINI@{topk} = {gini:.4f}, NOV@{topk} = {nov:.4f} ')

        if tb:
            tb_fm.add_scalar('train/loss', train_loss, epoch_i)
            tb_fm.add_scalar(f'eval/HR@{topk}', hr, epoch_i)
            tb_fm.add_scalar(f'eval/NDCG@{topk}', ndcg, epoch_i)
            tb_fm.add_scalar(f'eval/COV@{topk}', cov, epoch_i)
            tb_fm.add_scalar(f'eval/GINI@{topk}', gini, epoch_i)
            tb_fm.add_scalar(f'eval/NOV@{topk}', nov, epoch_i)

        # Specify a path to save to
    PATH = "FactorizationMachineModel.pt"
    torch.save(model.state_dict(), PATH)

    
    res_header=[f"HR@{topk}", f"NDCG@{topk}", f"COV@{topk}",f"GINI@{topk}",f"NOV@{topk}" ]
    res_values=[f"{hr:.4f}", f"{ndcg:.4f}", f"{cov:.4f}", f"{gini:.4f}", f"{nov:.4f}"  ]
    res_info=[res_header,res_values]
    dataset_path = "../data/"

    info_model_report (model, dataset_path, res_info, l_info, \
            full_dataset,dict_recommend, title="Factorization Machine - Partial", topk=10 )