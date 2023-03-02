from confection import Config
import copy
import polars as pl
import torch
from torch.utils.data import DataLoader
import tqdm.auto as tqdm
import wandb

from sngrok.data import SnDataset
from sngrok.permutations import make_permutation_dataset
from sngrok.model import SnMLP
from sngrok.utils import (
    calculate_checkpoint_epochs,
    parse_arguments,
    set_seeds,
    setup_checkpointing,
    to_numpy
)


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
    train_data = SnDataset(config['n'], sn_split[1])
    test_data = SnDataset(config['n'], sn_split[0])  
    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'])
    test_dataloader = DataLoader(test_data, batch_size=config['batch_size'])
    return train_dataloader, test_dataloader, sn_mult_table


def loss_fn(logits, labels):
    if len(logits.shape) == 3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    return -1. * correct_log_probs


def log_conj_class_losses(data):
    left_grouped = data.groupby(['left_conj_class']).agg([pl.col('loss').mean()])
    right_grouped = data.groupby(['right_conj_class']).agg([pl.col('loss').mean()])
    target_grouped = data.groupby(['target_conj_class']).agg([pl.col('loss').mean()])
    full_grouped = data.groupby(
        ['left_conj_class', 'right_conj_class', 'target_conj_class']
    ).agg([pl.col('loss').mean()])

    msg = {}
    for record in left_grouped.to_dicts():
        name = f'left_class/{record["left_conj_class"]}'
        msg[name] = record['loss']
    
    for record in right_grouped.to_dicts():
        name = f'right_class/{record["right_conj_class"]}'
        msg[name] = record['loss']

    for record in target_grouped.to_dicts():
        name = f'target_class/{record["target_conj_class"]}'
        msg[name] = record['loss']
    
    for record in full_grouped.to_dicts():
        lconj = record['left_conj_class']
        rconj = record['right_conj_class']
        tconj = record['target_conj_class']
        name = f'conj_class/{lconj}x{rconj}->{tconj}'
        msg[name] = record['loss']
    return msg


def loss_data_df(lconj, rconj, target_conj, lperm, rperm, target, loss):
    data = {
        'left_perm': to_numpy(lperm),
        'right_perm': to_numpy(rperm),
        'target_perm': to_numpy(target),
        'left_conj_class': lconj,
        'right_conj_class': rconj,
        'target_conj_class': target_conj,
        'loss': to_numpy(loss)
    }
    return pl.DataFrame(data)


def train_forward(model, dataloader):
    total_loss = torch.tensor(0., device='cuda')
    for lconj, rconj, target_conj, lperm, rperm, target in dataloader:
        logits = model(lperm.to('cuda'), rperm.to('cuda'))
        losses = loss_fn(logits, target.to('cuda'))
        mean_loss = losses.mean()
        mean_loss.backward()
        total_loss += mean_loss
    return total_loss.item()


def test_forward(model, dataloader):
    #loss_data = []
    total_loss = torch.tensor(0., device='cuda')
    for lconj, rconj, target_conj, lperm, rperm, target in dataloader:
        logits = model(lperm.to('cuda'), rperm.to('cuda'))
        losses = loss_fn(logits, target.to('cuda'))
        total_loss += losses.mean()
        #loss_data.append(loss_data_df(lconj, rconj, target_conj, lperm, rperm, target, losses))
    return total_loss.item()


def conj_forward(model, dataloader):
    loss_data = []
    total_loss = torch.tensor(0., device='cuda')
    for lconj, rconj, target_conj, lperm, rperm, target in dataloader:
        logits = model(lperm.to('cuda'), rperm.to('cuda'))
        losses = loss_fn(logits, target.to('cuda'))
        total_loss += losses.mean()
        loss_data.append(loss_data_df(lconj, rconj, target_conj, lperm, rperm, target, losses))
    return pl.concat(loss_data)


def train(model, optimizer, train_dataloader, test_dataloader, config):
    train_config = config['train']
    checkpoint_dir, run_data_dir = setup_checkpointing(train_config)
    checkpoint_epochs = calculate_checkpoint_epochs(train_config)
    model_checkpoints = []
    opt_checkpoints = []

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

        if test_loss <= train_config['grok_threshold']:
            break

        if epoch > 0 and (epoch % 1000 == 0):
            test_loss_df = conj_forward(model, test_dataloader)
            num_vals = test_loss_df.shape[0]
            test_loss_data.append(
                test_loss_df.with_columns(
                    pl.Series(name='epoch', values=([epoch]*num_vals))
                )
            )
            msg.update(log_conj_class_losses(test_loss_df))
            
            test_loss_data = []
        
        wandb.log(msg)
    
    if len(test_loss_data) > 0:
        run_df = pl.concat(test_loss_data, how='vertical')
        run_df.write_parquet(run_data_dir / 'losses.parquet')

    torch.save(
        {
            "model": model.state_dict(),
            "config": config['model'],
            "checkpoints": model_checkpoints,
            "checkpoint_epochs": checkpoint_epochs,
        },
        checkpoint_dir / "full_run.pth"
    )

def main():
    args = parse_arguments()
    config = Config().from_disk(args.config)

    device = torch.device('cuda')

    np_rng = set_seeds(config['train']['seed'])

    train_data, test_data, _ = get_dataloaders(
        config['train'],
        np_rng
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
    wandb.watch(model, log='all', log_freq=1000)

    try:
        train(
            model,
            optimizer,
            train_data,
            test_data,
            config
        )
    except KeyboardInterrupt:
        pass

    wandb.finish()


if __name__ == '__main__':
    main()

