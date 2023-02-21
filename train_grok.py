import argparse
from confection import Config
import copy
import math
import torch
from torch.utils.data import DataLoader, TensorDataset
import tqdm.auto as tqdm
import wandb

from sngrok.permutations import Permutation, make_permutation_dataset
from sngrok.model import SnMLP


def set_seeds(seed):
    torch.manual_seed(seed)


def calculate_checkpoint_epochs(config):
    extra_checkpoints = config.get('extra_checkpoints', [])
    num_epochs = config['num_epochs']
    checkpoint_every = config['checkpoint_every']
    main_checkpoints = list(range(0, num_epochs, checkpoint_every))
    return sorted(extra_checkpoints + main_checkpoints)



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to TOML config file')
    args, _ = parser.parse_known_args()
    return args


def make_dataset(n: int, device):
    _, mul_table = make_permutation_dataset(n)
    all_data = torch.tensor(mul_table, device=device)
    return torch.hsplit(all_data, 3)


def get_dataloaders(n: int, frac_train: float, batch_size: int, device):
    group_order = math.factorial(n)
    lperms, rperms, labels = make_dataset(n, device)
    indices = torch.randperm(group_order**2)
    assert len(indices) == len(lperms)
    cutoff = int(group_order**2 * frac_train)
    train_indices = indices[:cutoff]
    test_indices = indices[cutoff:]
    train_data = TensorDataset(lperms[train_indices].squeeze(), rperms[train_indices].squeeze(), labels[train_indices].squeeze())
    test_data = TensorDataset(lperms[test_indices].squeeze(),  rperms[test_indices].squeeze(), labels[test_indices].squeeze())
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
    for x, y, labels in dataloader:
        logits = model(x, y)
        loss = loss_fn(logits, labels)
        loss.backward()
        total_loss += loss
    return total_loss


def test_forward(model, dataloader):
    total_loss = torch.tensor(0., device='cuda', requires_grad=False)
    for x, y, labels in dataloader:
        logits = model(x, y)
        loss = loss_fn(logits, labels)
        total_loss += loss
    return total_loss


def train(model, optimizer, train_dataloader, test_dataloader, config):
    num_epochs = config['num_epochs']
    grok_threshold = config['grok_threshold']
    checkpoint_epochs = calculate_checkpoint_epochs(config)
    train_losses = []
    test_losses = []
    model_checkpoints = []
    opt_checkpoints = []

    for epoch in tqdm.tqdm(range(num_epochs)):
        train_loss = train_forward(model, train_dataloader)
        np_train = train_loss.item()
        train_losses.append(np_train)

        optimizer.step()
        optimizer.zero_grad()

        test_loss = test_forward(model, test_dataloader)
        np_test = test_loss.item()
        test_losses.append(np_test)

        optimizer.zero_grad()
        for param in model.parameters():
            if param.requires_grad:
                param.grad = None

        msg = {'loss/train': np_train, 'loss/test': np_test}

        wandb.log(msg)

        if epoch in checkpoint_epochs:
            model_state = copy.deepcopy(model.state_dict())
            opt_state = copy.deepcopy(optimizer.state_dict())
            torch.save(
                {
                    "model": model_state,
                    "optimizer": opt_state,
                    "config": model.cfg,
                    "rng": torch.get_rng_state()
                },
                f'checkpoints/s5_40/{epoch}.pth'
            )
            model_checkpoints.append(model_state)
            opt_checkpoints.append(opt_state)
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
     "grokking_s5_40_full_run.pth")

def main():
    args = parse_arguments()
    config = Config().from_disk(args.config)

    device = torch.device('cuda')
    torch.manual_seed(config['train']['seed'])
    
    model = SnMLP(config['model']).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['wd'],
        betas=config['optimizer']['betas']
    )

    wandb.init(
        entity="dstander",
        project="grokking_sn",
        group="S5_basic",
        config=config
    )

    wandb.watch(model, log_freq=100)

    train_config = config['train']

    train_data, test_data = get_dataloaders(
        train_config['n'],
        train_config['frac_train'],
        train_config['batch_size'],
        device
    )

    train(
        model,
        optimizer,
        train_data,
        test_data,
        train_config
    )


if __name__ == '__main__':
    main()

