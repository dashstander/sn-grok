from functools import reduce
from itertools import combinations, pairwise
import numpy as np
from .permutations import make_permutation_dataset, Permutation
from .tableau import enumerate_standard_tableau, YoungTableau


def adj_trans_decomp(i: int, j: int) -> list[tuple[int]]:
    center = [(i, i + 1)]
    i_to_j = list(range(i+1, j+1))
    adj_to_j = list(pairwise(i_to_j))
    return list(reversed(adj_to_j)) + center + adj_to_j


class SnIrrep:

    def __init__(self, n: int, partition: tuple[int]):
        self.n = n
        self.shape = partition
        self.basis = enumerate_standard_tableau(partition)
        self.permutations, self.mult_table = make_permutation_dataset(n)
        self.dim = len(self.basis)

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


    




class TrivialRep(SnIrrep):

    def __init__(self, n):
        pass

class StandardRep(SnIrrep):

    def __init__(n):
        pass


class AlternatingRep(SnIrrep):
    def __init__(n):
        pass

def make_irrep(partition):

    if list(partition) != sorted(partition, reverse=True):
        raise ValueError(
            f'Partition {partition} is not sorted in descending order.'
        )
    
    n = sum(partition)

    if partition == (n - 1, 1):
        return StandardRep(n)
    elif partition == (n):
        return TrivialRep(n)
    elif partition == tuple([1] * n):
        return AlternatingRep(n)
    else:
        return SnIrrep(n, partition)