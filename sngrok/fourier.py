import torch
from itertools import product
import functorch

from .irreps import SnIrrep
from .permutations import Permutation
from .tableau import generate_partitions


def _dot(fx, rho):
    return fx * rho

fft_dot = functorch.vmap(_dot, in_dims=(0, 0))


def _fft_sum(fx, rho):
    if rho.dim() == 1:
        return torch.dot(fx, rho)
    else:
        return fft_dot(fx, rho).sum(dim=0)


def _frob_norm(ft_vals):
    dim = ft_vals.dim()
    if dim < 2:
        return ft_vals**2
    else:
        dim = ft_vals.shape[0]
        return dim * torch.trace(ft_vals.T @ ft_vals)


def _ift_trace(ft_vals, inv_rep):
    dim = inv_rep.dim()
    if dim < 2:
        return inv_rep * ft_vals
    else:
        dim = inv_rep.shape[0]
        return dim * torch.trace(inv_rep @ ft_vals)


ift_trace = functorch.vmap(_ift_trace, in_dims=(0, None))
fft_sum = functorch.vmap(_fft_sum, in_dims=(1, None))
batch_kron = functorch.vmap(torch.kron, in_dims=(0, 0))
frob = functorch.vmap(_frob_norm, in_dims=0)


def calc_power(ft, group_order):
    return {k: (frob(v) / group_order**2)  for k, v in ft.items()}


def slow_ft_1d(fn_vals, n):
    all_partitions = generate_partitions(n)
    all_irreps = [SnIrrep(n, p) for p in all_partitions]
    results = {}
    for irrep in all_irreps:
        matrices = irrep.matrix_tensor()
        results[irrep.shape] = fft_sum(fn_vals, matrices).squeeze()
    return results


def slow_ft_2d(fn_vals, n):
    all_partitions = generate_partitions(n)
    all_irreps = [SnIrrep(n, p) for p in all_partitions]
    results = {}
    for lirrep, rirrep in product(all_irreps, all_irreps):
        mats1, mats2 = zip(
            *product(
                lirrep.matrix_tensor().split(1),
                rirrep.matrix_tensor().split(1))
        )
        mats = batch_kron(torch.cat(mats1).squeeze(), torch.cat(mats2).squeeze())
        results[(lirrep.shape, rirrep.shape)] = fft_sum(fn_vals.to(torch.float64), mats).squeeze()
    return results


def sn_fourier_basis(ft, n):
    all_partitions = generate_partitions(n)
    permutations = Permutation.full_group(n)
    group_order = len(permutations)
    all_irreps = {p: SnIrrep(n, p).matrix_representations() for p in all_partitions}
    ift_decomps = []
    for perm in permutations:
        fourier_decomp = []
        for part in all_partitions:
            inv_rep = torch.asarray(all_irreps[part][perm.sigma].T).squeeze()            
            fourier_decomp.append(ift_trace(ft[part], inv_rep).unsqueeze(0))
        ift_decomps.append(torch.cat(fourier_decomp).unsqueeze(0))
    return torch.cat(ift_decomps) / group_order


def sn_fourier_basis_2d(ft, n):
    all_partitions = generate_partitions(n)
    permutations = Permutation.full_group(n)
    group_order = 1.0 * len(permutations)**2
    all_irreps = {p: SnIrrep(n, p).matrix_representations() for p in all_partitions}
    ift_decomps = []
    for perm1, perm2 in product(permutations, permutations):
        fourier_decomp = []
        for part1, part2 in product(all_partitions, all_partitions):
            inverse_mat1 = torch.asarray(all_irreps[part1][perm1.sigma].T).squeeze()      
            inverse_mat2 = torch.asarray(all_irreps[part2][perm2.sigma].T).squeeze()
            inv_mat = torch.kron(inverse_mat1 , inverse_mat2)
            trace = ift_trace(ft[(part1, part2)], inv_mat).unsqueeze(0)
            fourier_decomp.append(trace)
        ift_decomps.append(torch.cat(fourier_decomp).unsqueeze(0))
    return torch.cat(ift_decomps) / group_order
