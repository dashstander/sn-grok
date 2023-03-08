import torch
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
    
fft_sum = functorch.vmap(_fft_sum, in_dims=(1, None))


def slow_ft(fn_vals, n):
    all_partitions = generate_partitions(n)
    all_irreps = [SnIrrep(n, p) for p in all_partitions]
    results = {}
    for irrep in all_irreps:
        matrices = irrep.matrix_tensor()
        results[irrep.shape] = fft_sum(fn_vals, matrices).squeeze()
    return results


def _ift_trace(ft_vals, inv_rep):
    if inv_rep.dim() < 2:
        return inv_rep * ft_vals
    else:
        return torch.trace(inv_rep @ ft_vals)

ift_trace = functorch.vmap(_ift_trace, in_dims=(0, None))


def sn_fourier_basis(ft, n):
    all_partitions = generate_partitions(n)
    permutations = Permutation.full_group(n)
    all_irreps = {p: SnIrrep(n, p).matrix_representations() for p in all_partitions}
    ift_decomps = []
    for perm in permutations:
        fourier_decomp = []
        for part in all_partitions:
            inv_rep = torch.asarray(all_irreps[part][perm.sigma].T).squeeze()            
            fourier_decomp.append(ift_trace(ft[part], inv_rep).unsqueeze(0))
        ift_decomps.append(torch.cat(fourier_decomp).unsqueeze(0))
    return torch.cat(ift_decomps)