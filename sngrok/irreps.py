import numpy as np
from .permutations import make_permutation_dataset, Permutation
from .tableau import enumerate_standard_tableau, YoungTableau


class SnIrrep:
    def __init__(self, n: int, partition: tuple[int]):
        self.n = n
        self.shape = partition
        self.basis = enumerate_standard_tableau(partition)


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