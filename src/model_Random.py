import random
import torch


class RandomModel(torch.nn.Module):
    def __init__(self, dims):
        super(RandomModel, self).__init__()
        """
        Simple random based recommender system
        """
        self.all_items = list(range(dims[0], dims[1]))
        print("dimensions",dims[0], dims[1])
    def forward(self):
        pass

    def predict(self, interactions, device=None):
        #print("Random",self.all_items)
        #print("Random",len(self.all_items))
        #print("Random",len(interactions))
        return torch.FloatTensor(random.sample(self.all_items, len(interactions)-1)) #quitar el uno para hacer normal
