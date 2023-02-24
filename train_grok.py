import argparse
from confection import Config
import copy
import numpy as np
from pathlib import Path
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
import tqdm.auto as tqdm
import wandb

from sngrok.permutations import make_permutation_dataset
from sngrok.model import SnMLP


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to TOML config file')
    args, _ = parser.parse_known_args()
    return args


def set_seeds(seed):
    np_rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    return np_rng

def setup_checkpointing(config):
    base_dir = Path(config['checkpoint_dir'])
    checkpoint_dir = base_dir / config['run_dir']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir

def calculate_checkpoint_epochs(config):
    extra_checkpoints = config.get('extra_checkpoints', [])
    num_epochs = config['num_epochs']
    checkpoint_every = config['checkpoint_every']
    main_checkpoints = list(range(0, num_epochs, checkpoint_every))
    return sorted(extra_checkpoints + main_checkpoints)


def train_test_split(df, frac_train, rng):
    group_order = df.shape[0]
    num_train_samples = int(group_order * frac_train)
    zeroes = pl.zeros(group_order, dtype=pl.UInt8)
    train_split = rng.choice(group_order, num_train_samples, replace=False)
    zeroes[train_split] = 1
    return df.with_column(zeroes.alias('in_train'))


def get_subgroup(df, parity):
    if parity == 'all':
        return df
    elif parity == 0:
        return df.filter(
            (pl.col('parity') == 0) & (pl.col('parity_right') == 0)
        )
    elif parity == 1:
        return df.filter(
            (pl.col('parity') == 1) | (pl.col('parity_right') == 1)
        )
    else:
        raise ValueError('Parity can only be 0, 1, or "all". Not {parity}')


def get_dataloaders(config, rng, device):
    frac_train = config['frac_train']
    _, sn_mult_table = make_permutation_dataset(config['n'])
    sn_mult_table = train_test_split(sn_mult_table, frac_train, rng)
    sn_mult_table = get_subgroup(sn_mult_table, config['parity'])

    sn_split = sn_mult_table.partition_by('in_train', as_dict=True)
    train_lperms, train_rperms, train_targets = torch.as_tensor(
        sn_split[1].select(['index', 'index_right', 'result_index']).to_numpy(),
        device=device
    ).hsplit(3)
    test_lperms, test_rperms, test_targets = torch.as_tensor(
        sn_split[0].select(['index', 'index_right', 'result_index']).to_numpy(),
        device=device
    ).hsplit(3)
    train_data = TensorDataset(train_lperms, train_rperms, train_targets)
    test_data = TensorDataset(test_lperms, test_rperms,test_targets)    
    train_dataloader = DataLoader(train_data, batch_size= config['batch_size'])
    test_dataloader = DataLoader(test_data, batch_size= config['batch_size'])
    return train_dataloader, test_dataloader, sn_mult_table


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
    train_config = config['train']
    checkpoint_dir = setup_checkpointing(train_config)
    checkpoint_epochs = calculate_checkpoint_epochs(train_config)
    train_losses = []
    test_losses = []
    model_checkpoints = []
    opt_checkpoints = []

    for epoch in tqdm.tqdm(range(train_config['num_epochs'])):
        train_loss = train_forward(model, train_dataloader)
        np_train = train_loss.item()
        train_losses.append(np_train)

        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            test_loss = test_forward(model, test_dataloader)
        np_test = test_loss.item()
        test_losses.append(np_test)

        optimizer.zero_grad()
        msg = {'loss/train': np_train, 'loss/test': np_test}

        wandb.log(msg)

        if epoch in checkpoint_epochs:
            model_state = copy.deepcopy(model.state_dict())
            opt_state = copy.deepcopy(optimizer.state_dict())
            torch.save(
                {
                    "model": model_state,
                    "optimizer": opt_state,
                    "config": config['model'],
                    "rng": torch.get_rng_state()
                },
                checkpoint_dir / f'{epoch}.pth'
            )
            model_checkpoints.append(model_state)
            opt_checkpoints.append(opt_state)

        if np_test <= train_config['grok_threshold']:
            break

    torch.save(
        {
            "model": model.state_dict(),
            "config": config['model'],
            "checkpoints": model_checkpoints,
            "checkpoint_epochs": checkpoint_epochs,
            "test_losses": test_losses,
            "train_losses": train_losses},
        checkpoint_dir / "full_run.pth"
    )

def main():
    args = parse_arguments()
    config = Config().from_disk(args.config)

    device = torch.device('cuda')

    np_rng = set_seeds(config['train']['seed'])

    train_data, test_data, mult_table = get_dataloaders(
        config['train'],
        np_rng,
        device
    )

    model = SnMLP.from_config(config['model']).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['wd'],
        betas=config['optimizer']['betas']
    )

    wandb.init(
        **config['wandb'],
        config=config
    )
    wandb.watch(model, log='all', log_freq=100)

    train(
        model,
        optimizer,
        train_data,
        test_data,
        config
    )

    wandb.finish()


if __name__ == '__main__':
    main()

