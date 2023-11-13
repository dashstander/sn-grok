from confection import Config, registry
import copy
import math
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens import HookedTransformerConfig, HookedTransformer
import tqdm.auto as tqdm
import wandb

from sngrok.fourier import calc_power
from sngrok.groups import group_registry
from sngrok.model import SnMLP
from sngrok.utils import (
    calculate_checkpoint_epochs,
    parse_arguments,
    set_seeds,
    setup_checkpointing,
)

registry.groups = group_registry


def get_optimizer(params, config):
    name = config.pop('algorithm')
    if name == 'adam':
        optimizer = torch.optim.AdamW(
            params,
            **config
        )
    elif name == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            **config
        )
    else:
        raise NotImplementedError(f'No optimizer named {name}')
    return optimizer


def calc_power_contributions(tensor, group):
    total_power = (tensor ** 2).mean(dim=0)
    fourier_transform = group.fourier_transform(tensor)
    irrep_power = calc_power(fourier_transform, group.order)
    power_contribs = {irrep: power / total_power for irrep, power in irrep_power.items()}
    irreps = list(power_contribs.keys())
    power_vals = torch.cat([power_contribs[irrep].unsqueeze(0) for irrep in irreps], dim=0)
    val_data = pl.DataFrame(power_vals.detach().cpu().numpy(), schema=[f'dim{i}' for i in range(power_vals.shape[1])])
    val_data.insert_at_idx(
        0,
        pl.Series('irrep', [str(i) for i in irreps])
    )
    return val_data, power_contribs


def fourier_analysis(model, group, epoch):
    W = model.linear.weight
    lembeds = model.lembed.weight.T
    rembeds = model.rembed.weight.T
    embed_dim = lembeds.shape[0]
    left_linear = (W[:, :embed_dim] @ lembeds).T
    right_linear = (W[:, embed_dim:] @ rembeds).T
    lembed_power_df, lpowers = calc_power_contributions(left_linear, group)
    rembed_power_df, rpowers = calc_power_contributions(right_linear, group)
    unembed_power_df, unembed = calc_power_contributions(model.unembed.weight, group)
    lembed_power_df.insert_at_idx(0, pl.Series('layer', ['left_linear'] * lembed_power_df.shape[0]))
    rembed_power_df.insert_at_idx(0, pl.Series('layer', ['right_linear'] * rembed_power_df.shape[0]))
    unembed_power_df.insert_at_idx(0, pl.Series('layer', ['unembed'] * unembed_power_df.shape[0]))
    df = pl.concat([lembed_power_df, rembed_power_df, unembed_power_df], how='vertical')
    df.insert_at_idx(0, pl.Series('epoch', [epoch] * df.shape[0]))
    return df, lpowers, rpowers, unembed


def train_test_split(df, frac_train, seed):
    group_order = df.shape[0]
    #zeroes = pl.zeros(group_order, dtype=pl.UInt8, eager=True)
    train_split = (
        pl.int_range(0, group_order, eager=True)
        .sample(
            fraction=frac_train,
            with_replacement=False,
            seed=seed)
    )
    return (
        df
        .with_row_count()
        .with_columns(
            pl.when(pl.col('row_nr').is_in(train_split))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias('in_train')
        )
        .select(pl.exclude('row_nr'))
    )
    


def get_dataloaders(group_mult_table, config, device):
    frac_train = config['frac_train']
    order = math.factorial(config['n'])
    equals = order + 1
    group_mult_table = train_test_split(group_mult_table, frac_train, config['seed'])
    sn_split = group_mult_table.partition_by('in_train', as_dict=True)
    train_perms = torch.as_tensor(sn_split[1].select(['index_left', 'index_right']).to_numpy(), dtype=torch.int64, device=device)
    train_perms = torch.concat([train_perms, torch.full((train_perms.shape[0], 1), equals, device=device)], dim=1)
    train_targets = torch.as_tensor(sn_split[1].select('index_target').to_numpy(), dtype=torch.int64, device=device)
    test_perms = torch.as_tensor(sn_split[0].select(['index_left', 'index_right']).to_numpy(), dtype=torch.int64, device=device)
    test_perms = torch.concat([test_perms, torch.full((test_perms.shape[0], 1), equals, device=device)], dim=1)
    test_targets = torch.as_tensor(sn_split[0].select('index_target').to_numpy(), dtype=torch.int64, device=device)
    train_data = TensorDataset(train_perms, train_targets)
    test_data = TensorDataset(test_perms, test_targets)
    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'])
    test_dataloader = DataLoader(test_data, batch_size=config['batch_size'])

    return train_dataloader, test_dataloader, group_mult_table


def loss_fn(logits, labels):
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels)[:, 0]
    return -1. * correct_log_probs


def train_forward(model, dataloader):
    total_loss = torch.tensor(0., device='cuda')
    for perms, target in dataloader:
        logits = model(perms)[:, -1]
        losses = loss_fn(logits, target)
        mean_loss = losses.mean()
        mean_loss.backward()
        total_loss += mean_loss
    return total_loss.item()


def test_forward(model, dataloader):
    total_loss = torch.tensor(0., device='cuda')

    for perms, target in dataloader:
        logits  = model(perms)[:, -1]
        losses = loss_fn(logits, target)
        total_loss += losses.mean()
    return total_loss.item()


def train(model, optimizer, train_dataloader, test_dataloader, config, seed, group):
    train_config = config['train']
    checkpoint_dir = setup_checkpointing(train_config, seed)
    checkpoint_epochs = calculate_checkpoint_epochs(train_config)
    model_checkpoints = []
    opt_checkpoints = []
    train_loss_data = []
    test_loss_data = []

    for epoch in tqdm.tqdm(range(train_config['num_epochs'])):
        train_loss = train_forward(model, train_dataloader)

        optimizer.step()
        optimizer.zero_grad()

        msg = {'loss/train': train_loss}

        #if epoch % 100 == 0:
        with torch.no_grad():
            test_loss = test_forward(model, test_dataloader)
            msg['loss/test'] = test_loss

        optimizer.zero_grad()

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
    registry_objects = registry.resolve(config)

    group = registry_objects['group']
    group_mult_table = group.make_multiplication_table()

    device = torch.device('cuda')
    seed = config['train']['seed']
    _ = set_seeds(seed)

    train_data, test_data, mult_table = get_dataloaders(
        group_mult_table,
        config['train'],
        device
    )

    checkpoint_dir  = setup_checkpointing(config['train'], seed)
    mult_table.select(
        [pl.col('^perm.*$'), pl.col('^index.*$'), 'in_train']
    ).write_parquet(checkpoint_dir / 'data.parquet')
    HookedTransformerConfig()
    transformer_config = HookedTransformerConfig(**config['model'])
    model = HookedTransformer(transformer_config)

    optimizer = get_optimizer(
        model.parameters(),
        config['optimizer']
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
            config,
            seed,
            group
        )
    except KeyboardInterrupt:
        pass

    wandb.finish()


if __name__ == '__main__':
    main()

