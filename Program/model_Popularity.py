import torch


class Popularity_Recommender(torch.nn.Module):
    # generates a popularity recommender model that predicts most popular items that the costumer has not previously purchased
    def __init__(self, adj_mx):
        super(Popularity_Recommender, self).__init__()
        self.all_items = torch.Tensor(adj_mx.sum(axis=0, dtype=int)).flatten() # sums the occurrences of each article to get its popularity score

    def forward(self):
        pass

    def predict(self, interactions, device=None):
        items = torch.LongTensor(interactions[:, 1])
        return torch.index_select(self.all_items, 0, items)

