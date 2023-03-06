import torch
from functorch import vmap

from .irreps import SnIrrep
from .permutations import Permutation
from .tableau import generate_partitions


def scalar_x_mat(fx, rho):
    return fx * rho


fft_dot = vmap(scalar_x_mat, in_dims=(0, 0))

def slow_ft(fn, n):
    all_partitions = generate_partitions(n)
    permutations = Permutation.full_group(n)
    fn_vals = fn(permutations)
    all_irreps = [SnIrrep(n, p) for p in all_partitions]
    results = {}
    for irrep in all_irreps:
        matrices = irrep.tensor()
        if len(matrices.shape) == 1:
            irrep_val = torch.dot(fn_vals, matrices)
        else:
            irrep_val = fft_dot(fn_vals, matrices).sum(dim=0)
        results[irrep.shape] = irrep_val
    return results

def slow_ift(ft, n):
