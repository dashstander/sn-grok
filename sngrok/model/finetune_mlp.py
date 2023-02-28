import torch
from torch import nn
from torch.nn.functional import relu
from transformer_lens.hook_points import HookedRootModule, HookPoint

from .mlp import SnMLP



class SnFinetuneMLP(HookedRootModule):
    def __init__(self, vocab_size: int, embed_dim: int, model_dim: int, subgroup_mlp: SnMLP):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.model_dim = model_dim

        self.subgroup_mlp = subgroup_mlp
        self.subgroup_vocab_size = subgroup_mlp.vocab_size

        self.ft_lembed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
        self.ft_rembed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
        self.linear2 = nn.Linear(in_features=self.model_dim, out_features=self.model_dim, bias=False)
        self.new_group_unembed = nn.Linear(in_features=self.model_dim, out_features=self.vocab_size)
        self.hook_lembed = HookPoint()
        self.hook_rembed = HookPoint()
        self.hook_linear1 = HookPoint()
        self.hook_linear2 = HookPoint()
        self.hook_unembed = HookPoint()
        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        self.setup()

    @classmethod
    def from_config(cls, config):
        model_config = config['model']
        vocab = model_config['vocab_size']
        embed_dim = model_config['embed_dim']
        model_dim = model_config['model_dim']
        pretrained_model = SnMLP.from_config(config['pretrained_model'])
        full_run = torch.load(config['pretrained_model']['checkpoint_path'])
        weights = full_run['model']
        pretrained_model.load_state_dict(weights)
        return cls(vocab, embed_dim, model_dim, pretrained_model)

    def _embedding_index(self, x_idx, y_idx):
        lembeds = torch.zeros((*x_idx.shape, self.embed_dim))
        rembeds = torch.zeros((*x_idx.shape, self.embed_dim))

        lsubgroup_idx = torch.argwhere(x_idx < self.subgroup_vocab_size).squeeze()
        lnewgroup_idx = torch.argwhere(x_idx >= self.subgroup_vocab_size).squeeze()
        rsubgroup_idx = torch.argwhere(y_idx < self.subgroup_vocab_size).squeeze()
        rnewgroup_idx = torch.argwhere(y_idx >= self.subgroup_vocab_size).squeeze()

        reduced_x_idx = torch.remainder(x_idx, self.subgroup_vocab_size)
        reduced_y_idx = torch.remainder(y_idx, self.subgroup_vocab_size)

        lsubgroup_embeds = self.subgroup_mlp.lembed(reduced_x_idx[lsubgroup_idx])
        lnewgroup_embeds = self.ft_lembed(reduced_x_idx[lnewgroup_idx])
        rsubgroup_embeds = self.subgroup_mlp.rembed(reduced_y_idx[rsubgroup_idx])
        rnewgroup_embeds = self.ft_rembed(reduced_y_idx[rnewgroup_idx])

        lembeds.index_add(0, lsubgroup_idx, lsubgroup_embeds)
        lembeds.index_add(0, lnewgroup_idx, lnewgroup_embeds)
        rembeds.index_add(0, rsubgroup_idx, rsubgroup_embeds)
        rembeds.index_add(0, rnewgroup_idx, rnewgroup_embeds)
        self.hook_lembed(lembeds)
        self.hook_rembed(rembeds)

        return torch.concatenate([lembeds, rembeds], dim=-1)

    def forward(self, x, y): 
        permrep = self._embedding_index(x, y)
        linear1 = self.hook_linear1(relu(self.subgroup_mlp.linear(permrep)))
        linear2 = self.hook_linear2(relu(self.linear2(linear1)))
        logits = self.hook_unembed(self.new_group_unembed(linear2))
        return logits

