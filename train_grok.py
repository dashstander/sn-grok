import copy
import einops
import math
import torch
from torch.utils.data import DataLoader, TensorDataset
#from torch.nn.functional import cosine_similarity
import tqdm.auto as tqdm
# from transformer_lens import HookedTransformer, HookedTransformerConfig
import wandb

from sngrok.permutations import Permutation, make_permutation_dataset
from sngrok.models import SnMLP


def make_dataset(n: int, device):
    _, mul_table = make_permutation_dataset(n)
    all_data = torch.tensor(mul_table, device=device)
    return torch.hsplit(all_data, 3)


def get_dataloaders(n: int, frac_train: float, batch_size: int, device):
    group_order = math.factorial(n)
    lperms, rperms, labels = make_dataset(n, device)
    indices = torch.randperm(group_order**2)
    assert len(indices) == len(lperms)
    cutoff = int((p**2) * frac_train)
    train_indices = indices[:cutoff]
    test_indices = indices[cutoff:]
    train_data = TensorDataset(lperms[train_indices], rperms[train_indices], labels[train_indices])
    test_data = TensorDataset(dataset[test_indices],  rperms[test_indices], labels[test_indices])
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return train_dataloader, test_dataloader


def loss_fn(logits, labels):
    if len(logits.shape) == 3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    return -correct_log_probs.mean()


def train_forward(model, dataloader):
    total_loss = torch.tensor(0., device='cuda', requires_grad=False)
    for batch, labels in dataloader:
        logits = model(batch)
        loss = loss_fn(logits, labels)
        loss.backward()
        total_loss += loss
    return total_loss


def test_forward(model, dataloader):
    total_loss = torch.tensor(0., device='cuda', requires_grad=False)
    for batch, labels in dataloader:
        logits = model(batch)
        loss = loss_fn(logits, labels)
        total_loss += loss
    return total_loss


def train(model, optimizer, train_dataloader, test_dataloader, checkpoint_every, num_epochs, grok_threshold):
    train_losses = []
    test_losses = []
    model_checkpoints = []
    opt_checkpoints = []
    checkpoint_epochs = []

    for epoch in tqdm.tqdm(range(num_epochs)):
        train_loss = train_forward(model, train_dataloader)
        np_train = train_loss.item()
        train_losses.append(np_train)

        train_grads = get_grads(model)

        optimizer.step()
        optimizer.zero_grad()

        
        test_loss = test_forward(model, test_dataloader)
        np_test = test_loss.item()
        test_losses.append(np_test)

        test_grads = get_grads(model)
        optimizer.zero_grad()
        for param in model.parameters():
            if param.requires_grad:
                param.grad = None

        similarity, full_grads = grad_similarity(train_grads, test_grads)

        similarity = {f'gradient_alignment/{n}': sim for n, sim in similarity.items()}
        
        msg = {'loss/train': np_train, 'loss/test': np_test}
        msg.update(similarity)

        wandb.log(msg)

        if (epoch % checkpoint_every) == 0:
            checkpoint_epochs.append(epoch)
            model_state = copy.deepcopy(model.state_dict())
            opt_state = copy.deepcopy(optimizer.state_dict())
            torch.save(
                {
                    "model": model_state,
                    "optimizer": opt_state,
                    "config": model.cfg,
                    "rng": torch.get_rng_state()
                },
                f'checkpoints/xy33/{epoch}.pth'
            )
            model_checkpoints.append(model_state)
            opt_checkpoints.append(opt_state)
            #print(f"Epoch {epoch} Train Loss {np_train} Test Loss {np_test}")

        if test_loss.item() <= grok_threshold:
            break
    torch.save(
     {
         "model":model.state_dict(),
         "config": model.cfg,
         "checkpoints": model_checkpoints,
         "checkpoint_epochs": checkpoint_epochs,
         "test_losses": test_losses,
         "train_losses": train_losses
     },
     "grokking_xy_33_full_run.pth")

def main():
    
    model = HookedTransformer(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
        betas=betas
    )

    wandb.init(
        entity="dstander",
        project="grokking_sn",
        group="S5_basic",
        config=cfg.to_dict()
    )

    #wandb.watch(model, log_freq=100)

    train_data, test_data = get_dataloaders(p, frac_train, batch_size, device)

    train(
        model,
        optimizer,
        train_data,
        test_data,
        checkpoint_every,
        num_epochs,
        grok_threshold
    )


if __name__ == '__main__':
    main()

