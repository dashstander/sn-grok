from functools import reduce
from itertools import combinations, pairwise, permutations
import numpy as np
import torch

from .permutations import Permutation
from .tableau import enumerate_standard_tableau, YoungTableau


def adj_trans_decomp(i: int, j: int) -> list[tuple[int]]:
    center = [(i, i + 1)]
    i_to_j = list(range(i+1, j+1))
    adj_to_j = list(pairwise(i_to_j))
    return list(reversed(adj_to_j)) + center + adj_to_j


def cycle_to_one_line(cycle_rep):
    n = sum([len(c) for c in cycle_rep])
    sigma = [-1] * n
    for cycle in cycle_rep:
        first = cycle[0]
        if len(cycle) == 1:
            sigma[first] = first
        else:
            for val1, val2 in pairwise(cycle):
                sigma[val2] = val1
                lastval  = val2
            sigma[first] = lastval
    return tuple(sigma)


def trans_to_one_line(i, j, n):
    sigma = list(range(n))
    sigma[i] = j
    sigma[j] = i
    return tuple(sigma)



class SnIrrep:

    def __init__(self, n: int, partition: tuple[int]):
        self.n = n
        self.shape = partition
        self.basis = enumerate_standard_tableau(partition)
        self.permutations = Permutation.full_group(n)
        self.dim = len(self.basis)
        self._matrices = None

    def adjacent_transpositions(self):
        return pairwise(range(self.n))
    
    def non_adjacent_transpositions(self):
        return [(i, j) for i, j in combinations(range(self.n), 2) if i+1 != j]

    def adj_transposition_matrix(self, a, b):
        perm = Permutation.transposition(self.n, a, b)
        irrep = np.zeros((self.dim, self.dim))
        def fn(i, j):
            tableau = self.basis[i]
            #print(tableau)
            if i == j:
                d = tableau.transposition_dist(a, b)
                return 1. / d
            else:
                new_tab = perm * tableau
                if new_tab == self.basis[j]:
                    d = tableau.transposition_dist(a, b)**2
                    return np.sqrt(1 - (1. / d))
                else:
                    return 0.
        for x in range(self.dim):
            for y in range(self.dim):
                irrep[x, y] = fn(x, y)
        return irrep
    
    def generate_transposition_matrices(self):
        matrices = {
            (i, j): self.adj_transposition_matrix(i, j) for i, j in self.adjacent_transpositions()
        }
        for i, j in self.non_adjacent_transpositions():
            decomp = [matrices[pair] for pair in adj_trans_decomp(i, j)]
            matrices[(i, j)] = reduce(lambda x, y: x @ y, decomp)
        return matrices

    def matrix_representations(self):
        if self._matrices is not None:
            return self._matrices
        transpo_matrices = self.generate_transposition_matrices()
        matrices = {
            trans_to_one_line(*k, self.n): v for k, v in transpo_matrices.items()
        }
        matrices[tuple(range(self.n))] = np.eye(self.dim)
        for perm in self.permutations:
            if perm.sigma in matrices:
                continue
            else:
                cycle_mats = [
                    transpo_matrices[t] for t in perm.transposition_decomposition()
                ]
                perm_rep = reduce(lambda x, y: x @ y, cycle_mats)
                matrices[perm.sigma] = perm_rep
                if perm.inverse.sigma not in matrices:
                    matrices[perm.inverse.sigma] = perm_rep.T
        self._matrices = matrices
        return matrices

    def matrix_tensor(self):
        matrices = self.matrix_representations()
        tensors = [torch.asarray(matrices[perm.sigma]).unsqueeze(0) for perm in self.permutations]
        return torch.concatenate(tensors, dim=0).squeeze()
    
    def alternating_matrix_tensor(self):
        matrices = self.matrix_representations()
        tensors = [torch.asarray(matrices[perm.sigma]).unsqueeze(0) for perm in self.permutations if perm.parity == 0]
        return torch.concatenate(tensors, dim=0).squeeze()


class TrivialRep(SnIrrep):

    def __init__(self, n):
        self.n = n
        self.permutations = [
            Permutation(seq) for seq in permutations(list(range(n)))
        ]
        self.basis = [YoungTableau([list(range(n))])]
    
    def matrix_representations(self):
        return {
            perm.sigma : 1. for perm in self.permutations
        }


class AlternatingRep(SnIrrep):
    def __init__(self, n):
        self.n = n
        self.permutations = [
            Permutation(seq) for seq in permutations(list(range(n)))
        ]
        self.basis = [YoungTableau([[i] for i in range(n)])]
        self.dim = len(self.basis)
    
    @staticmethod
    def _sign(perm):
        if perm.parity == 0:
            return 1.
        else:
            return -1.
    
    def matrix_representations(self):
        return {
            perm.sigma : self._sign(perm) for perm in self.permutations
        }


def make_irrep(partition):

    if list(partition) != sorted(partition, reverse=True):
        raise ValueError(
            f'Partition {partition} is not sorted in descending order.'
        )
    
    n = sum(partition)

    if partition == (n):
        return TrivialRep(n)
    elif partition == tuple([1] * n):
        return AlternatingRep(n)
    else:
        return SnIrrep(n, partition)
