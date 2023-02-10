import torch
from torch import nn
from torch.nn.functional import relu


class SnMLP(nn.Module):

    def __init__(self, config):
        self.vocab_size = config['vocab_size']
        self.embed_dim = config['embed_dim']
        self.model_dim = config['model_dim']
        self.lembed = nn.Embedding(num_embeddings=self.vocab_size, embed_dim=self.embed_dim)
        self.rembed = nn.Embedding(num_embeddings=self.vocab_size, embed_dim=self.embed_dim)
        self.linear = nn.Linear(in_features=(2 * self.embed_dim), out_features=self.model_dim, bias=False)
        self.unembed = nn.Linear(in_features=self.model_dim, out_features=self.vocab_size)
    
    def forward(self, x, y):
        permrep = torch.concatenate((self.lembed(x), self.rembed(y)), dim=1)
        linear1 = self.linear(permrep)
        logits = self.unembed(relu(linear1))
        return logits
