import torch
import numpy as np


class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        return torch.sum(self.fc(x), dim=1) + self.bias


class FieldAwareFactorizationMachine(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
                ix = torch.stack(ix, dim=1)
        return ix


class FieldAwareFactorizationMachineModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.ffm = FieldAwareFactorizationMachine(field_dims, embed_dim)

    def forward(self, x):
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + ffm_term
        return torch.sigmoid(x.squeeze(1))

    def predict(self, interactions, device):
        test_interactions = torch.from_numpy(interactions).to(dtype=torch.long, device=device)
        output_scores = self.forward(test_interactions)
        return output_scores