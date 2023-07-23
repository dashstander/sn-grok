from confection import Config, registry
import copy
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
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


def train_test_split(df, frac_train, rng):
    group_order = df.shape[0]
    num_train_samples = int(group_order * frac_train)
    zeroes = pl.zeros(group_order, dtype=pl.UInt8)
    train_split = rng.choice(group_order, num_train_samples, replace=False)
    zeroes[train_split] = 1
    return df.with_columns(zeroes.alias('in_train'))


def get_dataloaders(group_mult_table, config, rng, device):
    frac_train = config['frac_train']
    group_mult_table = train_test_split(group_mult_table, frac_train, rng)
    sn_split = group_mult_table.partition_by('in_train', as_dict=True)
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

    return train_dataloader, test_dataloader, group_mult_table


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


def train(model, optimizer, train_dataloader, test_dataloader, config, seed, group):
    train_config = config['train']
    checkpoint_dir  = setup_checkpointing(train_config, seed)
    checkpoint_epochs = calculate_checkpoint_epochs(train_config)
    train_loss_data = []
    test_loss_data = []
    loss_epochs = []

    for epoch in tqdm.tqdm(range(train_config['num_epochs'])):
        train_loss = train_forward(model, train_dataloader)

        optimizer.step()
        optimizer.zero_grad()

        msg = {'loss/train': train_loss}

        if (epoch % 100 == 0) or (epoch in checkpoint_epochs):
            with torch.no_grad():
                test_loss = test_forward(model, test_dataloader)
                msg['loss/test'] = test_loss
            train_loss_data.append(train_loss)
            test_loss_data.append(test_loss)
            loss_epochs.append(epoch)

        optimizer.zero_grad()

        if (epoch % 1000 == 0) or (epoch in checkpoint_epochs):
            _, left_powers, right_powers, unembed_powers = fourier_analysis(model, group, epoch)
            left_powers = {f'left_linear/{k}': v for k, v in left_powers.items()}
            right_powers = {f'right_linear/{k}': v for k, v in right_powers.items()}
            unembed_powers = {f'unembed/{k}': v for k, v in unembed_powers.items()}
            msg.update(left_powers)
            msg.update(right_powers)
            msg.update(unembed_powers)

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
        
        wandb.log(msg)


    torch.save(
        {
            "model": model.state_dict(),
            "config": config['model'],
            "checkpoint_epochs": checkpoint_epochs,
            "train_loss": train_loss_data,
            "test_loss": test_loss_data,
            "loss_epochs": loss_epochs
        },
        checkpoint_dir / "full_run.pth"
    )

def main():
    args = parse_arguments()
    config = Config().from_disk(args.config)
    registry_objects = registry.resolve(config)

    wandb.init(
        config=config
    )
    seed = wandb.config.seed

    group = registry_objects['group']
    group_mult_table = group.make_multiplication_table()

    device = torch.device('cuda')

    np_rng = set_seeds(seed)

    train_data, test_data, mult_table = get_dataloaders(
        group_mult_table,
        config['train'],
        np_rng,
        device
    )

    checkpoint_dir = setup_checkpointing(config['train'], seed)
    mult_table.select(
        [pl.col('^perm.*$'), pl.col('^index.*$'), 'in_train']
    ).write_parquet(checkpoint_dir / 'data.parquet')

    model = SnMLP.from_config(config['model']).to(device)
    optimizer = get_optimizer(
        model.parameters(),
        config['optimizer']
    )

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

