from confection import Config
from pathlib import Path
import polars as pl
import torch


from sngrok.permutations import make_permutation_dataset
from sngrok.model import SnMLP
from sngrok.utils import set_seeds


def train_test_split(n, frac_train, rng):
    _, df = make_permutation_dataset(n)
    group_order = df.shape[0]
    num_train_samples = int(group_order * frac_train)
    zeroes = pl.zeros(group_order, dtype=pl.UInt8)
    train_split = rng.choice(group_order, num_train_samples, replace=False)
    zeroes[train_split] = 1
    return df.with_columns(zeroes.alias('in_train'))


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


def initialize_and_save_data(n, frac_train, experiment_dir, seed):
    rng = set_seeds(seed)
    sn_data = train_test_split(n, frac_train, rng)
    sn_data.select(
        [pl.col('^perm.*$'), pl.col('^index.*$'), 'in_train']
    ).write_parquet(experiment_dir / f'data_{seed}.parquet')


def main():
    n = 5
    frac_train = 0.5
    model_seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    data_seeds = [10, 11, 12, 13, 14, 15, 16, 17]
    config = Config().from_disk('configs/s5_mlp.toml')
    exp_dir = Path('checkpoints/experiments')

    for ms in model_seeds:
        initialize_and_save_model(config, exp_dir, ms)
    
    for ds in data_seeds:
        initialize_and_save_data(n, frac_train, exp_dir, ds)

