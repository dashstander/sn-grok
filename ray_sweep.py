from confection import Config
import copy
from pathlib import Path
from itertools import product
import polars as pl
import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
import wandb

from sngrok.model import SnMLP
from sngrok.permutations import make_permutation_dataset
from sngrok.utils import (
    calculate_checkpoint_epochs,
    parse_arguments,
    set_seeds
    
)


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


@ray.remote(num_gpus=1)
def train(config):
    #config = copy.deepcopy(config)
    #config.update({'experiment': experiment_config})
    experiment_config = config['experiment']
    exp_dir = Path(experiment_config['experiment_dir'])
    run_dir = exp_dir / run_name
    run_dir.mkdir(exist_ok=True)

    model_version = config['model_seed']
    data_version = config['data_seed']
    run_name = f'model-{model_version}_data-{data_version}'

    device = torch.device('cuda')

    model = initialize_model(config)
    model = model.to(device)
    sn_data = initialize_and_save_data(config, run_dir)

    print(f'Beginning training {run_name} on device {ray.get_gpu_ids()}')
    

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['wd'],
        betas=config['optimizer']['betas']
    )

    train_dataloader, test_dataloader = get_dataloaders(config, sn_data, device)

    wandb.init(
        **config['wandb'],
        name=run_name,
        config=config
    )

    train_config = config['train']
    checkpoint_epochs = calculate_checkpoint_epochs(train_config)
    model_checkpoints = []
    opt_checkpoints = []
    train_loss_data = []
    test_loss_data = []

    for epoch in range(train_config['num_epochs']):
        train_loss = train_forward(model, train_dataloader)

        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            test_loss = test_forward(model, test_dataloader)

        optimizer.zero_grad()

        msg = {'loss/train': train_loss, 'loss/test': test_loss}

        if epoch in checkpoint_epochs:
            print(f'{run_name}: saving at epoch {epoch} with loss {test_loss}')
            model_state = copy.deepcopy(model.state_dict())
            opt_state = copy.deepcopy(optimizer.state_dict())
            torch.save(
                {
                    "model": model_state,
                    "optimizer": opt_state,
                    "config": config['model'],
                    "rng": torch.get_rng_state()
                },
                run_dir / f'{epoch}.pth'
            )
            model_checkpoints.append(model_state)
            opt_checkpoints.append(opt_state)
            train_loss_data.append(train_loss)
            test_loss_data.append(test_loss)

        wandb.log(msg)
        if test_loss <= train_config['grok_threshold']:
            break
        
    torch.save(
        {
            "model": model.state_dict(),
            "final_epoch": epoch,
            "config": config['model'],
            "full_config": config,
            "checkpoints": model_checkpoints,
            "checkpoint_epochs": checkpoint_epochs,
        },
        run_dir / "full_run.pth"
    )
    wandb.finish()
    return run_name, epoch, test_loss



def initialize_model(config):
    seed = config['seed']
    set_seeds(seed)
    model = SnMLP.from_config(config['model'])
    return model


def initialize_and_save_data(config, experiment_dir):
    seed = config['seed']
    rng = set_seeds(seed)
    n = config['train']['n']
    frac_train = config['train']['frac_train']
    sn_data = train_test_split(n, frac_train, rng)
    sn_data.select(
        [pl.col('^perm.*$'), pl.col('^index.*$'), 'in_train']
    ).write_parquet(experiment_dir / f'data_{seed}.parquet')
    return sn_data


def main():
    num_gpus = 8
    ray.init(num_gpus=num_gpus)

    gpus_per_trial = 0.49

    args = parse_arguments()
    config = Config().from_disk(args.config)

    model_seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    data_seeds = [8, 9, 10, 11, 12, 13, 14, 15]


    jobs = [train.remote(config, exp) for exp in exp_configs]
    time.sleep(300)
    while jobs:
        finished, jobs = ray.wait(jobs, num_returns=1)
        if len(finished) > 0:
            results = ray.get(finished)
            for name, epoch, loss in results:
                print(f'Job {name} finished after {epoch} epochs with loss {loss}')
        time.sleep(300)


if __name__ == '__main__':
    main()

