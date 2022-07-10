import random
import torch


class RandomModel(torch.nn.Module):
    # generates a random recommender model that predicts random articles that the costumer has not previously purchased
    def __init__(self, dims):
        super(RandomModel, self).__init__()
        self.all_items = list(range(dims[0], dims[1]))

    def forward(self):
        pass

    def predict(self, interactions, device=None):
        return torch.FloatTensor(random.sample(self.all_items, len(interactions)-1))
