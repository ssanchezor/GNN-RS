import torch


class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(field_dims, output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        return torch.sum(self.fc(x), dim=1) + self.bias


class FM_operation(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class FactorizationMachineModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.embedding = torch.nn.Embedding(field_dims, embed_dim, sparse=False)
        self.fm = FM_operation(reduce_sum=True)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, interaction_pairs):
        out = self.linear(interaction_pairs) + self.fm(self.embedding(interaction_pairs))
        return out.squeeze(1)

    def predict(self, interactions, device):
        # return the score, inputs are numpy arrays, outputs are tensors
        test_interactions = torch.from_numpy(interactions).to(dtype=torch.long, device=device)
        output_scores = self.forward(test_interactions)
        return output_scores