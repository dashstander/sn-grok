from accelerate import Accelerator
from confection import Config
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
import tqdm.auto as tqdm
import wandb

from sngrok.permutations import make_permutation_dataset
from sngrok.model import SnMLP
from sngrok.utils import (
    calculate_checkpoint_epochs,
    parse_arguments,
    setup_checkpointing
)


accelerator = Accelerator(log_with="wandb")


def train_test_split(df, frac_train, rng):
    group_order = df.shape[0]
    num_train_samples = int(group_order * frac_train)
    zeroes = pl.zeros(group_order, dtype=pl.UInt8)
    train_split = rng.choice(group_order, num_train_samples, replace=False)
    zeroes[train_split] = 1
    return df.with_columns(zeroes.alias('in_train'))


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


def get_dataloaders(config, rng):
    frac_train = config['frac_train']
    _, sn_mult_table = make_permutation_dataset(config['n'])
    sn_mult_table = train_test_split(sn_mult_table, frac_train, rng)
    sn_mult_table = get_subgroup(sn_mult_table, config['parity'])

    sn_split = sn_mult_table.partition_by('in_train', as_dict=True)
    train_lperms, train_rperms, train_targets = torch.as_tensor(
        sn_split[1].select(['index', 'index_right', 'result_index']).to_numpy()
    ).hsplit(3)
    test_lperms, test_rperms, test_targets = torch.as_tensor(
        sn_split[0].select(['index', 'index_right', 'result_index']).to_numpy()
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
    correct_log_probs = log_probs.gather(dim=-1, index=labels)[:, 0]
    return -1. * correct_log_probs


def train_forward(model, dataloader):
    total_loss = torch.tensor(0., device='cuda', requires_grad=False)
    per_sample_losses = []
    for x, y, labels in dataloader:
        logits = model(x, y)
        losses = loss_fn(logits, labels)
        mean_loss = losses.mean()
        accelerator.backward(mean_loss)
        #loss.backward()
        total_loss += mean_loss
        per_sample_losses.append(losses)
    return mean_loss


def test_forward(model, dataloader):
    per_sample_losses = []
    total_loss = torch.tensor(0., device='cuda', requires_grad=False)
    for x, y, labels in dataloader:
        logits = model(x, y)
        losses = loss_fn(logits, labels)
        total_loss += losses.mean()
        per_sample_losses.append(losses)
    return total_loss


def train(model, optimizer, train_dataloader, test_dataloader, config):
    train_config = config['train']
    checkpoint_dir = setup_checkpointing(train_config)
    checkpoint_epochs = calculate_checkpoint_epochs(train_config)
    train_losses = []
    test_losses = []
    progress_bar = tqdm.tqdm(
        range(train_config['num_epochs']),
        disable=not accelerator.is_local_main_process
    )
    for epoch in progress_bar:
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

        accelerator.log(msg, step=epoch)

        if epoch in checkpoint_epochs:
            accelerator.wait_for_everyone()
            accelerator.save_state(checkpoint_dir)

        if np_test <= train_config['grok_threshold']:
            break
    
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        torch.save(
            {
                "model": model.state_dict(),
                "config": config['model'],
                "test_losses": test_losses,
                "train_losses": train_losses
            },
            checkpoint_dir / "full_run.pth"
        )

def main():
    args = parse_arguments()
    config = Config().from_disk(args.config)

    accelerator = Accelerator()

    np_rng = np.random.default_rng(config['train']['seed'])

    train_data, test_data, _ = get_dataloaders(
        config['train'],
        np_rng 
    )
    accelerator.wait_for_everyone()

    model = SnMLP.from_config(config['model'])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['wd'],
        betas=config['optimizer']['betas']
    )

    accelerator.init_trackers(
        "grokking_sn",
        config=config,
        init_kwargs={'wandb': config['wandb']}
    )

    model, optimizer, training_dataloader = accelerator.prepare(model, optimizer, train_data)
    testing_dataloader = accelerator.prepare(test_data)

    try:
        train(
            model,
            optimizer,
            training_dataloader,
            testing_dataloader,
            config
        )
    except KeyboardInterrupt:
        pass

    accelerator.end_training()


if __name__ == '__main__':
    main()

