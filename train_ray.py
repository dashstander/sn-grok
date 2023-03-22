from confection import Config
import copy
from pathlib import Path
from itertools import product
import polars as pl
import ray
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
import wandb

from sngrok.model import SnMLP
from sngrok.permutations import make_permutation_dataset
from sngrok.utils import (
    calculate_checkpoint_epochs,
    parse_arguments,
    set_seeds,
    setup_checkpointing
)


@ray.remote
class ProgressActor:
    def __init__(self, total_tasks: int):
        self.total_tasks = 1.0 * total_tasks
        self.tasks = {}
        self.completed_tasks = 0
        

    def report_progress(self, task_id: str, num_epochs: int, test_loss: float, is_finished: bool) -> None:
        self.tasks[task_id] = {'epochs': num_epochs, 'loss': test_loss, 'is_finished': is_finished}

    def get_progress(self):
        update = {'jobs': {}}
        for task_id, progress in self.tasks.items():
            update['jobs'][task_id] = progress
            if progress['is_finished']:
                self.tasks.pop(task_id)
                self.completed_tasks += 1
        update['progress': self.completed_tasks / self.total_tasks]
        return update


def train_test_split(df, frac_train, rng):
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
def train(config, experiment_config, sentinel):
    config = copy.deepcopy(config)
    config.update({'experiment': experiment_config})
    exp_dir = Path(experiment_config['experiment_dir'])
    model_version = experiment_config['model_version']
    data_version = experiment_config['data_version']
    run_name = f'model-{model_version}_data-{data_version}'

    run_dir = exp_dir / run_name
    run_dir.mkdir(exist_ok=True)

    device = torch.device('cuda')
    model_fp = exp_dir / f'model_{model_version}.pth'
    data_fp = exp_dir / f'data_{data_version}.parquet'
    model = SnMLP.from_config(config['model'])
    model.load_state_dict(model_fp).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['wd'],
        betas=config['optimizer']['betas']
    )

    train_dataloader, test_dataloader = get_dataloaders(config, data_fp, device)

    wandb.init(
        **config['wandb'],
        name=run_name,
        config=config
    )
    #wandb.watch(model, log='all', log_freq=1000)

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

        if epoch % 100 == 0:
            sentinel.report_progress.remote(run_name, epoch, test_loss, False)

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
                run_dir / f'{epoch}.pth'
            )
            model_checkpoints.append(model_state)
            opt_checkpoints.append(opt_state)
            train_loss_data.append(train_loss)
            test_loss_data.append(test_loss)

        wandb.log(msg)
        if test_loss <= train_config['grok_threshold']:
            break
    
    sentinel.report_progress.remote(run_name, epoch, test_loss, True)    
    
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
    ray.init()

    args = parse_arguments()
    config = Config().from_disk(args.config)

    exp_dir = Path('checkpoints/experiments')
    model_seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    data_seeds = [8, 9, 10, 11, 12, 13, 14, 15]

    for s in model_seeds:
        initialize_and_save_model(config, exp_dir, s)
    
    for t in data_seeds:
        initialize_and_save_data(config, exp_dir, t)

    exp_configs = [
        {
        'experiment_dir': exp_dir,
        'model_version': mseed,
        'data_version': dseed
        } for mseed, dseed in product(model_seeds, data_seeds)
    ]

    sentinel = ProgressActor.remote(len(exp_configs))

    unfinished = [train.remote(config, exp, sentinel) for exp in exp_configs]
    time.sleep(60)
    while unfinished:
        update = sentinel.get_progress.remote()
        _, unfinished = ray.wait(unfinished, num_returns=1)
        print('###############')
        print(f'{update["progress"] * 100}% complete')
        for job in update['jobs']:
            print(job)
        print('###############')
        time.sleep(300)


if __name__ == '__main__':
    main()

