import torch


class Popularity_Recommender(torch.nn.Module):
    def __init__(self, adj_mx):
        super(Popularity_Recommender, self).__init__()

        """
        Simple popularity based recommender system
        """
        # Sum the occurences of each item to get is popularity, convert to array and lose the extra dimension
        self.all_items = torch.Tensor(adj_mx.sum(axis=0, dtype=int)).flatten()

    def forward(self):
        pass

    def predict(self, interactions, pop):
        items = torch.LongTensor(interactions[:, 1])
        return torch.index_select(self.all_items, 0, items)

