from argparse import ArgumentParser
import json
import math
from pathlib import Path
import polars as pl
import torch
from tqdm import tqdm

from sngrok.proper_subgroups import  all_s5_subgroups

from sngrok.cosets import make_left_full_coset_df, make_right_full_coset_df

from sngrok.fourier import slow_sn_ft_1d, sn_fourier_basis, calc_power
from sngrok.groups import generate_subgroup, Symmetric
from sngrok.permutations import Permutation
from sngrok.model import SnMLP
from sngrok.tableau import generate_partitions


parser = ArgumentParser()
parser.add_argument('-n', type=int, help='The number of elements being permuted.')
parser.add_argument('--input_dir', type=str, help='Path to checkpoints')
parser.add_argument('--output_dir', type=str, help='Path data will be saved')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--mod', type=int)


def loss_fn(logits, labels):
    if len(logits.shape) == 3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])
    return -1. * correct_log_probs


def calc_power_contributions(tensor, n):
    total_power = (tensor ** 2).mean(dim=0)
    group_order = math.factorial(n)
    fourier_transform = slow_sn_ft_1d(tensor, n)
    irrep_power = calc_power(fourier_transform, group_order)
    power_contribs = {irrep: power / total_power for irrep, power in irrep_power.items()}
    irreps = list(power_contribs.keys())
    power_vals = torch.cat([power_contribs[irrep].unsqueeze(0) for irrep in irreps], dim=0)
    val_data = pl.DataFrame(power_vals.detach().cpu().numpy(), schema=[f'dim{i}' for i in range(tensor.shape[1])])
    val_data.insert_at_idx(0, pl.Series('irrep', [str(i) for i in irreps]))
    return val_data, fourier_transform


def fourier_basis_to_df(tensor, n, layer):
    group_order, num_irreps, fn_dim = tensor.shape
    all_partitions = generate_partitions(n)
    permutations = Permutation.full_group(n)
    assert len(permutations) == group_order
    assert len(all_partitions) == num_irreps
    
    long_values = tensor.reshape((-1, fn_dim))
    group_col= []
    for s in permutations:
        group_col += [str(s.sigma)] * num_irreps
    part_col = [str(p) for p in all_partitions] * group_order
    assert len(group_col) == len(part_col) and len(group_col) == long_values.shape[0]
    val_data = pl.DataFrame(long_values.detach().numpy(), schema=[f'dim{i}' for i in range(fn_dim)])
    sn_metadata = pl.DataFrame({'layer': [layer] * len(group_col), 'permutation': group_col, 'irrep': part_col})
    return pl.concat([sn_metadata, val_data], how='horizontal')


def _all_data_coset_analysis(data, coset_df):
    base_df = (
        data
        .melt(id_vars=['layer', 'permutation', 'irrep'])
        .groupby(['layer', 'permutation', 'variable'])
        .agg(pl.col('value').sum())
    )
    summary_df = (
        base_df
        .groupby(['layer', 'variable'])
        .agg([
            pl.col('value').mean().alias('mean'),
            pl.col('value').min().alias('min'),
            pl.col('value').max().alias('max'),
            pl.col('value').var().alias('full_var'),
            (pl.col('value') ** 2).sum().alias('two_norm')
        ])
    )
    
    df = (
        base_df
        ##### Join against _all_ cosets, this is a many-to-1 join
        .join(coset_df, on='permutation', how='inner')
        .groupby(['layer', 'variable', 'subgroup', 'coset_rep'])
        .agg([
            # Get the average value and variance of the (per dim) activations over a single coset
            # Small variance --> activations highly concentrated on the coset
            pl.col('value').var().alias('coset_var'),
            pl.col('value').mean().alias('coset_mean')
        ])
        .sort(['variable', 'subgroup', 'coset_rep'])
        .groupby(['layer', 'variable', 'subgroup'], maintain_order=True)
        # Sum all the variances of the cosets for one subgroup
        .agg(pl.col('coset_var').sum().alias('coset_cond_var'))
        # Sort ascending by coset variance
        .sort(['variable', 'coset_cond_var'])
        .groupby(['layer', 'variable'], maintain_order=True)
        .agg([
            # 
            pl.col('subgroup').first(),
            pl.col('coset_cond_var').first().alias('min_coset_var'),
        ])
        .join(summary_df, on=['layer', 'variable'])
        .sort('subgroup')
        .with_columns(
            coset_var_ratio = (pl.col('min_coset_var') / pl.col('full_var')),
            subgroup_class = pl.col('subgroup').str.split(by='_').list.get(0)
        )
    )
    return df
         

def _make_one_coset_df(sg_def, n, name):
        all_subgroups = [generate_subgroup(gen) for gen in sg_def['generators']]
        right_coset_df = make_right_full_coset_df(all_subgroups, n, name)
        left_coset_df = make_left_full_coset_df(all_subgroups, n, name)
        return left_coset_df, right_coset_df


def make_full_coset_df(all_subgroups, n):
        left_cosets = []
        right_cosets = []
        
        for subgroup_name, subgroup_info in all_subgroups.items():
            ldf, rdf =  _make_one_coset_df(subgroup_info, n, subgroup_name)
            left_cosets.append(ldf)
            right_cosets.append(rdf)

        left_df = pl.concat(left_cosets)
        right_df = pl.concat(right_cosets)
        
        return left_df, right_df
                


def transpose_power_df(data):
    irreps = data['irrep'].to_list()
    return (
        data
        .select(pl.exclude('irrep'))
        .transpose(
            include_header=True,
            header_name='variable', 
            column_names=irreps)
    )


def fp_sort_key(fp):
    last_part = fp.parts[-1].strip('.pth')
    if last_part == 'full_run':
        return 2 ** 10000
    else:
        return int(last_part)
    
    
def _analysis(
        model,
        full_left_coset_df,
        full_right_coset_df,
        n,
        seed,
        epoch
    ):
    
    Sn = Symmetric(n)
    
    embed_dim = model.embed_dim
    W = model.linear.weight

    llinear_ft = slow_sn_ft_1d(model.lembed.weight @ W[:, :embed_dim].T, n)
    rlinear_ft = slow_sn_ft_1d(model.lembed.weight @ W[:, embed_dim:].T, n)

    llinear_decomp = sn_fourier_basis(llinear_ft, Sn)
    rlinear_decomp = sn_fourier_basis(rlinear_ft, Sn)
    
    llinear_df = fourier_basis_to_df(llinear_decomp, n, 'left_linear')
    rlinear_df = fourier_basis_to_df(rlinear_decomp, n, 'right_linear')
    
    ldf = _all_data_coset_analysis(llinear_df, full_right_coset_df)
    rdf = _all_data_coset_analysis(rlinear_df, full_left_coset_df)


    ldf.insert_at_idx(0, pl.Series('epoch', [epoch] * ldf.shape[0]))
    rdf.insert_at_idx(0, pl.Series('epoch', [epoch] * rdf.shape[0]))
    ldf.insert_at_idx(0, pl.Series('seed', [seed] * ldf.shape[0]))
    rdf.insert_at_idx(0, pl.Series('seed', [seed] * rdf.shape[0]))
    return ldf, rdf


def run_and_write(
        ckpt,
        config,
        full_left_coset_df,
        full_right_coset_df,
        epoch,
        model_seed,
        n,
        output_dir
):
    model = SnMLP.from_config(config)
    model.load_state_dict(ckpt)
    ldf, rdf = _analysis(
        model,
        full_left_coset_df,
        full_right_coset_df,
        n,
        model_seed,
        epoch
    )
    ldf.write_parquet(output_dir / f'left_cosets/{model_seed}.parquet')
    rdf.write_parquet(output_dir / f'right_cosets/{model_seed}.parquet')
    

def cosets_over_time(run_dir, full_left_coset_df, full_right_coset_df, n, output_dir, device):
    model_seed = int(run_dir.name.split('_')[-1])
    
    left_dir = output_dir / 'left_cosets'
    right_dir = output_dir / 'right_cosets'
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    left_exists = (output_dir / f'left_cosets/{model_seed}.parquet').exists()
    right_exists =  (output_dir / f'right_cosets/{model_seed}.parquet').exists()
    if left_exists and right_exists:
        return
    
    full_run = torch.load(run_dir / 'full_run.pth', map_location=device)
    
    checkpoint_epochs = full_run['checkpoint_epochs'] + [49999]
    
    for epoch, ckpt in zip(checkpoint_epochs, full_run['checkpoints']):
        run_and_write(
            ckpt,
            full_run['config'],
            full_left_coset_df,
            full_right_coset_df,
            epoch,
            model_seed,
            n,
            output_dir
        )


def tuplefy(generators):
    return [tuple(g) for g in generators]


def get_s6_subgroups():
    with open('s6_subgroups.json', mode='r') as jfile:
            all_s6_subgroups = json.load(jfile)
    for _, v in all_s6_subgroups.items():
        v['generators'] = [tuplefy(gens) for gens in v['generators']]
    return all_s6_subgroups


def main():
    args, _ = parser.parse_known_args()
    n = args.n
    device = torch.device(args.device)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if n == 5:
        all_subgroups = all_s5_subgroups
    elif n == 6:
        all_subgroups = get_s6_subgroups()
    
    print('Making coset dataframes....')
    full_left_coset_df, full_right_coset_df = make_full_coset_df(all_subgroups, n)
    
    for run_dir in tqdm(input_dir.iterdir()):
         seed = int(run_dir.name.split('_')[-1])
         if (seed % 4 == args.mod) and (run_dir / 'full_run.pth').exists():
            cosets_over_time(run_dir, full_left_coset_df, full_right_coset_df, n, output_dir, device)


if __name__ == '__main__':
     main()
