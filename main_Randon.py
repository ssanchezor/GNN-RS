
import torch
import os
from model_FM import FactorizationMachineModel
from train import test, train_one_epoch
from build_dataset import CustomerArticleDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random

class RandomModel(torch.nn.Module):
    def __init__(self, dims):
        super(RandomModel, self).__init__()
        """
        Simple random based recommender system
        """
        self.all_items = list(range(dims[0], dims[1]))

    def forward(self):
        pass

    def predict(self, interactions, device=None):
        return torch.FloatTensor(random.sample(self.all_items, len(interactions)))



if __name__ == '__main__':

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logs_base_dir = "runMOVIE"
    dataset_path = "../data/"
    
    os.makedirs(logs_base_dir, exist_ok=True)

    tb_Rand = SummaryWriter(log_dir=f'{logs_base_dir}/{logs_base_dir}_RND/')
    
    full_dataset = CustomerArticleDataset(dataset_path, num_negatives_train=4, num_negatives_test=99)
    data_loader = DataLoader(full_dataset, batch_size=256, shuffle=True, num_workers=0)

    #generar modelo
    model = RandomModel(data_loader.dataset.field_dims)
    

    
    #Train the model
    tb = True
    topk = 10
    num_epochs=20

    for epoch_i in range(num_epochs):
        #train_loss = train_one_epoch(model, optimizer, data_loader, criterion, device)
        train_loss=0

        hr, ndcg, coverage = test(model, full_dataset, device, topk=topk)

        print(f'epoch {epoch_i}:')
        print(f'training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f}, COV@{topk} = {coverage:.4f} ')


        if tb:
            tb_Rand.add_scalar('train/loss', train_loss, epoch_i)
            tb_Rand.add_scalar(f'eval/HR@{topk}', hr, epoch_i)
            tb_Rand.add_scalar(f'eval/NDCG@{topk}', ndcg, epoch_i)
            tb_Rand.add_scalar(f'eval/COV@{topk}', coverage, epoch_i)

    



