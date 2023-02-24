import numpy as np
import torch


class Irrep:
    def __init__(self, n: int, partition: tuple[int]):
        pass


class TrivialRep(Irrep):

    def __init__(self, n):
        pass

class StandardRep(Irrep):

    def __init__(n):
        pass


class AlternatingRep(Irrep):
    def __init__(n):
        pass

def make_irrep(partition):

    if list(partition) != sorted(partition, reverse=True):
        raise ValueError(f'Partition {partition} is not sorted in descending order.')
    
    n = sum(partition)

    if partition == (n - 1, 1):
        return StandardRep(n)
    elif partition == (n):
        return TrivialRep(n)
    elif partition == tuple([1] * n):
        return AlternatingRep(n)
    else:
        return Irrep(n, partition)