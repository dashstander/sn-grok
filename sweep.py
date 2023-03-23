import argparse
from confection import Config
import copy
from pathlib import Path
from itertools import product
import polars as pl
import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import wandb

from sngrok.model import SnMLP
from sngrok.fourier import slow_ft_1d, calc_power
from sngrok.permutations import make_permutation_dataset
from sngrok.utils import (
    calculate_checkpoint_epochs,
    set_seeds
)



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to TOML config file')
    parser.add_argument('--model_seed', type=int)
    parser.add_argument('--data_seed', type=int)
    args, _ = parser.parse_known_args()
    return args


def train_test_split(n, frac_train, rng):
    _, df = make_permutation_dataset(n)
    group_order = df.shape[0]
    num_train_samples = int(group_order * frac_train)
    zeroes = pl.zeros(group_order, dtype=pl.UInt8)
    train_split = rng.choice(group_order, num_train_samples, replace=False)
    zeroes[train_split] = 1
    return df.with_columns(zeroes.alias('in_train'))


def get_dataloaders(config, data_fp, device):
    data = pl.read_parquet(data_fp)
    sn_split = data.partition_by('in_train', as_dict=True)
    train_lperms = torch.as_tensor(sn_split[1].select('index_left').to_numpy(), dtype=torch.int64, device=device)
    train_rperms = torch.as_tensor(sn_split[1].select('index_right').to_numpy(), dtype=torch.int64, device=device)
    train_targets = torch.as_tensor(sn_split[1].select('index_target').to_numpy(), dtype=torch.int64, device=device)
    test_lperms = torch.as_tensor(sn_split[0].select('index_left').to_numpy(), dtype=torch.int64, device=device)
    test_rperms = torch.as_tensor(sn_split[0].select('index_right').to_numpy(), dtype=torch.int64, device=device)
    test_targets = torch.as_tensor(sn_split[0].select('index_target').to_numpy(), dtype=torch.int64, device=device)
    train_data = TensorDataset(train_lperms, train_rperms, train_targets)
    test_data = TensorDataset(test_lperms, test_rperms,test_targets)
    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'])
    test_dataloader = DataLoader(test_data, batch_size=config['batch_size'])

    return train_dataloader, test_dataloader


def loss_fn(logits, labels):
    if len(logits.shape) == 3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels)[:, 0]
    return -1. * correct_log_probs


def train_forward(model, dataloader):
    total_loss = torch.tensor(0., device='cuda')
    for lperm, rperm, target in dataloader:
        logits = model(lperm, rperm)
        losses = loss_fn(logits, target)
        mean_loss = losses.mean()
        mean_loss.backward()
        total_loss += mean_loss
    return total_loss.item()


def test_forward(model, dataloader):
    total_loss = torch.tensor(0., device='cuda')
    for lperm, rperm, target in dataloader:
        logits = model(lperm, rperm)
        losses = loss_fn(logits, target)
        total_loss += losses.mean()
    return total_loss.item()


def train(model, optimizer, train_dataloader, test_dataloader, config, checkpoint_dir):
    train_config = config['train']
    checkpoint_epochs = calculate_checkpoint_epochs(train_config)
    model_checkpoints = []
    opt_checkpoints = []

    train_loss_data = []
    test_loss_data = []

    for epoch in tqdm.tqdm(range(train_config['num_epochs'])):
        train_loss = train_forward(model, train_dataloader)

        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            test_loss = test_forward(model, test_dataloader)

        optimizer.zero_grad()

        msg = {'loss/train': train_loss, 'loss/test': test_loss}        

        if epoch in checkpoint_epochs:

            lembed_ft = slow_ft_1d(model.lembed.weight.to('cpu'), 5)
            rembed_ft = slow_ft_1d(model.rembed.weight.to('cpu'), 5)
            lembed_power = calc_power(lembed_ft, 120)
            rembed_power = calc_power(rembed_ft, 120)
            msg.update({
                'left_embedding_irreps': lembed_power,
                'right_embedding_irreps': rembed_power
            })
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
            train_loss_data.append(train_loss)
            test_loss_data.append(test_loss)

        if test_loss <= train_config['grok_threshold']:
            break
        
        wandb.log(msg)

    checkpoint_epochs = checkpoint_epochs[:len(model_checkpoints)]
    checkpoint_epochs.append(epoch)
    model_state = copy.deepcopy(model.state_dict())
    model_checkpoints.append(model_state)
    train_loss_data.append(train_loss)
    test_loss_data.append(test_loss)
    
    torch.save(
        {
            "model": model_state,
            "config": config['model'],
            "checkpoints": model_checkpoints,
            "checkpoint_epochs": checkpoint_epochs,
            "train_loss": train_loss_data,
            "test_loss": test_loss_data
        },
        checkpoint_dir / "full_run.pth"
    )
    

def initialize_and_save_model(config, experiment_dir, seed):
    set_seeds(seed)
    model = SnMLP.from_config(config['model'])
    torch.save(
        {
            "model": model.state_dict(),
            "config": config['model']
        },
        experiment_dir / f'init_{seed}.pth'
    )


def initialize_and_save_data(config, experiment_dir, seed):
    rng = set_seeds(seed)
    n = config['train']['n']
    frac_train = config['train']['frac_train']
    sn_data = train_test_split(n, frac_train, rng)
    sn_data.select(
        [pl.col('^perm.*$'), pl.col('^index.*$'), 'in_train']
    ).write_parquet(experiment_dir / f'data_{seed}.parquet')


def main():
    
    args = parse_arguments()
    config = Config().from_disk(args.config)

    exp_dir = Path('checkpoints/experiments')
    run_name = f'model_{args.model_seed}_data_{args.data_seed}'

    run_dir = exp_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda')

    data_fp = exp_dir / f'data_{args.data_seed}.parquet'
    model_fp = exp_dir / f'init_{args.model_seed}.pth'

    train_data, test_data = get_dataloaders(config['train'], data_fp, device)

    model_init = torch.load(model_fp, map_location='cpu')
    model = SnMLP.from_config(config['model'])
    model.load_state_dict(model_init['model'])
    model.to(device)


    torch.manual_seed(314159)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['wd'],
        betas=config['optimizer']['betas']
    )

    wandb.init(
        **config['wandb'],
        config=config,
        name=run_name
    )
    wandb.watch(model, log='all', log_freq=1000)

    try:
        train(
            model,
            optimizer,
            train_data,
            test_data,
            config,
            run_dir
        )
    except KeyboardInterrupt:
        pass

    wandb.finish()


if __name__ == '__main__':
    main()

