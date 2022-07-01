import torch
import numpy as np
from model_FM import FeaturesLinear, FM_operation
from torch_geometric.nn import GCNConv, GATConv


class GraphModel(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, features, train_mat, attention):
        super().__init__()
        self.A = train_mat
        self.features = features
        if attention:
            print(f'Attention on')
            self.GCN_module = GATConv(int(field_dims), embed_dim, heads=8, dropout=0.6)
        else:
            print(f'Attention off')
            self.GCN_module = GCNConv(field_dims, embed_dim)

    def forward(self, x):
        return self.GCN_module(self.features, self.A)[x]


class FactorizationMachineModel_withGCN(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, X, A, attention=False):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.embedding = GraphModel(field_dims, embed_dim, X, A, attention=attention)
        self.fm = FM_operation(reduce_sum=True)

    def forward(self, interaction_pairs):
        out = self.linear(interaction_pairs) + self.fm(self.embedding(interaction_pairs))
        return out.squeeze(1)

    def predict(self, interactions, device):
        test_interactions = torch.from_numpy(interactions).to(dtype=torch.long, device=device)
        output_scores = self.forward(test_interactions)
        return output_scores