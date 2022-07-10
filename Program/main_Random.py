
import torch
import os
from train import testpartial
from build_dataset import CustomerArticleDataset
from torch.utils.data import DataLoader
from model_Random import RandomModel
from torch.utils.tensorboard import SummaryWriter
from utils_report import info_model_report

if __name__ == '__main__':

    # checking GPU...
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # setting up TensorBoard and data paths...
    logs_base_dir = "runMOVIE"
    os.makedirs(logs_base_dir, exist_ok=True)
    tb_Rand = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_RND/')
    dataset_path = "../data/"

    # generating full dataset...
    full_dataset = CustomerArticleDataset(dataset_path, num_negatives_train=4, num_negatives_test=99)
    # splitting dataset into different batches...
    data_loader = DataLoader(full_dataset, batch_size=256, shuffle=True, num_workers=0)

    # generating the model...
    model = RandomModel(data_loader.dataset.field_dims)

    # "training" the model...
    tb = True
    topk = 10 # 10 articles to be recommended
    num_epochs= 1 # random does not need training

    for epoch_i in range(num_epochs):
        train_loss=0 # no parameters to learn
        hr, ndcg, cov, gini, dict_recomend, nov, l_info = testpartial(model, full_dataset, device, topk=topk)

        print(f'epoch {epoch_i}:')
        print(f'training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f}, COV@{topk} = {cov:.4f}, GINI@{topk} = {gini:.4f}, NOV@{topk} = {nov:.4f} ')

        # saving results in TensorBoard
        if tb:
            tb_Rand.add_scalar('train/loss', train_loss, epoch_i)
            tb_Rand.add_scalar(f'eval/HR@{topk}', hr, epoch_i)
            tb_Rand.add_scalar(f'eval/NDCG@{topk}', ndcg, epoch_i)
            tb_Rand.add_scalar(f'eval/COV@{topk}', cov, epoch_i)
            tb_Rand.add_scalar(f'eval/GINI@{topk}', gini, epoch_i)
            tb_Rand.add_scalar(f'eval/NOV@{topk}', nov, epoch_i)

    # saving training results...
    PATH = "Random_Recommender_80000.pt"
    torch.save(model.state_dict(), PATH)

    # generating customized report...
    res_header=[f"HR@{topk}", f"NDCG@{topk}", f"COV@{topk}",f"GINI@{topk}",f"NOV@{topk}" ]
    res_values=[f"{hr:.4f}", f"{ndcg:.4f}", f"{cov:.4f}", f"{gini:.4f}", f"{nov:.4f}"  ]
    res_info=[res_header,res_values]
    info_model_report (model, dataset_path, res_info, l_info, \
            full_dataset, dict_recomend, title="Random Recommender - Partial", topk=10 )

    



