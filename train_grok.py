from confection import Config
import copy
import math
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
import tqdm.auto as tqdm
import wandb

from sngrok.fourier import slow_ft_1d, calc_power
from sngrok.permutations import make_permutation_dataset
from sngrok.model import SnMLP
from sngrok.utils import (
    calculate_checkpoint_epochs,
    parse_arguments,
    set_seeds,
    setup_checkpointing,
)


def calc_power_contributions(tensor, n):
    total_power = (tensor ** 2).mean(dim=0)
    fourier_transform = slow_ft_1d(tensor, n)
    irrep_power = calc_power(fourier_transform, math.factorial(n))
    power_contribs = {irrep: power / total_power for irrep, power in irrep_power.items()}
    irreps = list(power_contribs.keys())
    power_vals = torch.cat([power_contribs[irrep].unsqueeze(0) for irrep in irreps], dim=0)
    val_data = pl.DataFrame(power_vals.detach().numpy(), schema=[f'dim{i}' for i in range(256)])
    val_data.insert_at_idx(
        0, 
        pl.Series('irrep', [str(i) for i in irreps])
    )
    return val_data


def fourier_analysis(model, n, epoch):
    W = model.linear.weight
    lembeds = model.lembed.weight.T
    rembeds = model.rembed.weight.T
    lembed_power_df = slow_ft_1d(W[:, :256] @ lembeds, n)
    rembed_power_df = slow_ft_1d(W[:, 256:] @ rembeds, n)
    lembed_power_df.insert_at_idx(0, pl.Series('layer', ['left_linear'] * lembed_power_df.shape[0]))
    rembed_power_df.insert_at_idx(0, pl.Series('layer', ['right_linear'] * rembed_power_df.shape[0]))
    df = pl.concat([lembed_power_df, rembed_power_df], how='vertical')
    df.insert_at_idx(0, pl.Series('epoch', [epoch] * df.shape[0]))
    return df



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


def get_dataloaders(config, rng, device):
    frac_train = config['frac_train']
    _, sn_mult_table = make_permutation_dataset(config['n'])
    sn_mult_table = train_test_split(sn_mult_table, frac_train, rng)
    sn_mult_table = get_subgroup(sn_mult_table, config['parity'])
    sn_split = sn_mult_table.partition_by('in_train', as_dict=True)
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

    return train_dataloader, test_dataloader, sn_mult_table


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
        logits  = model(lperm, rperm)
        losses = loss_fn(logits, target)
        total_loss += losses.mean()
    return total_loss.item()


def train(model, optimizer, train_dataloader, test_dataloader, config):
    train_config = config['train']
    n = config['train']['n']
    checkpoint_dir, _ = setup_checkpointing(train_config)
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

        freq_data = fourier_analysis(model, n, epoch)

        msg = {
            'loss/train': train_loss,
            'loss/test': test_loss,
            'fourier_data': freq_data.to_pandas()
        }        

        if epoch in checkpoint_epochs:
            train_loss_data.append(train_loss)
            test_loss_data.append(test_loss)
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
        
        wandb.log(msg)


    torch.save(
        {
            "model": model.state_dict(),
            "config": config['model'],
            "checkpoints": model_checkpoints,
            "checkpoint_epochs": checkpoint_epochs[:len(model_checkpoints)],

        },
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

    checkpoint_dir, _ = setup_checkpointing(config['train'])
    mult_table.select(
        [pl.col('^perm.*$'), pl.col('^index.*$'), 'in_train']
    ).write_parquet(checkpoint_dir / 'data.parquet')

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

    wandb.watch(model, log='parameters', log_freq=1000)

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

