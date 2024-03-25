from itertools import pairwise
import torch
from torch import nn
from torch.nn.functional import relu, gelu, silu
from transformer_lens.hook_points import HookedRootModule, HookPoint


activation_functions = {
    'relu': relu,
    'gelu': gelu,
    'silu': silu
}


class MLP(HookedRootModule):

    def __init__(self, model_dims):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_features=d1, out_features=d2)
            for d1, d2 in pairwise(model_dims)
        ])
        self.hooks = [HookPoint() for _ in self.layers]
    
    def forward(self, x):
        for layer, hook in zip(self.layers, self.hooks):
            x = relu(hook(layer(x)))
        return x


class SnMLP(HookedRootModule):

    def __init__(self, vocab_size, embed_dim, model_dim, act_fn='relu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.lembed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
        self.rembed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)

        self.linear = nn.Linear(in_features=(2 * self.embed_dim), out_features =self.model_dim, bias=False)
        self.unembed = nn.Linear(in_features=self.model_dim, out_features=self.vocab_size)
        self.hook_lembed = HookPoint()
        self.hook_rembed = HookPoint()
        self.hook_linear = HookPoint()
        self.hook_unembed = HookPoint()
        self.act_fn = activation_functions[act_fn.lower()]
        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        self.setup()

    @classmethod
    def from_config(cls, config):
        vocab = config['vocab_size']
        embed_dim = config['embed_dim']
        model_dims = config['model_dim']
        if 'act_fn' in config:
            act_fn = config['act_fn']
        else:
            act_fn = 'relu'
        return cls(vocab, embed_dim, model_dims, act_fn)
    
    def forward(self, x, y):
        lembed = self.hook_lembed(self.lembed(x))
        rembed = self.hook_rembed(self.rembed(y))
        permrep = torch.concatenate([lembed, rembed], dim=-1)
        hidden = self.act_fn(self.hook_linear(self.linear(permrep)))
        logits = self.hook_unembed(self.unembed(hidden))
        return logits
