import argparse
import numpy as np
from pathlib import Path
import torch


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
    data_dir = checkpoint_dir / 'run_data'
    data_dir.mkdir(exist_ok=True)
    return checkpoint_dir, data_dir


def calculate_checkpoint_epochs(config):
    extra_checkpoints = config.get('extra_checkpoints', [])
    num_epochs = config['num_epochs']
    checkpoint_every = config['checkpoint_every']
    main_checkpoints = list(range(0, num_epochs, checkpoint_every))
    return sorted(extra_checkpoints + main_checkpoints)


def to_numpy(x):
    return x.detach().cpu().numpy()