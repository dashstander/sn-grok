import catalogue
from itertools import pairwise, product
import math
import polars as pl
from sngrok.permutations import Permutation


group_registry = catalogue.create("groups", entry_points=False)


def generate_subgroup(generators: list[tuple[int]]) -> list[tuple[int]]:
    group_size = 0
    all_perms = set(generators)
    while group_size < len(all_perms):
        group_size = len(all_perms)
        perms = [Permutation(p) for p in all_perms]
        for perm1, perm2 in product(perms, repeat=2):
            perm3 = perm1 * perm2
            all_perms.add(perm3.sigma)
    return list(all_perms)


def cycle_to_one_line(cycle_rep: list[tuple[int]]) -> tuple[int]:
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


def add_fixed_to_cycle(cycle, n):
    missing = []
    
    for i in range(n):
        if i not in cycle:
            missing.append((i,))
    return [cycle] + missing


def three_cycle_to_one_line(i, n):
    cycle = (0, 1, i)
    cycle = add_fixed_to_cycle(cycle, n)
    return cycle_to_one_line(cycle)


class Symmetric:
    def __init__(self, n: int):
        self.elements = Permutation.full_group(n)
        self.order = math.factorial(n)


class Alternating:

    def __init__(self, n: int):
        if n < 3:
            raise ValueError('No alternating subgroup of S2')
        self.generators = [
            three_cycle_to_one_line(i, n) for i in range(2, n)
        ]
        self.elements = generate_subgroup(self.generators)
        self.order = math.factorial(n) / 2


class Dihedral:

    def __init__(self, n: int):
        if n < 3:
            raise ValueError('We start with triangles')
        self.order = 2 * n
        n_cycle = tuple([n - 1] + [i for i in range(n - 1)])
        self.generators = [
            (n - 2, n - 1), cycle_to_one_line([n_cycle])
        ]
        self.elements = generate_subgroup(self.generators)


def _make_multiplication_table(all_permutations):
    index = {perm.sigma : i for i, perm in enumerate(all_permutations)}
    left_perms = []
    lindex = []
    rindex = []
    right_perms = []
    target_perms = []
    target_index = []
    for lperm, rperm in product(all_permutations):
        left_perms.append(str(lperm.sigma))
        right_perms.append(str(rperm.sigma))
        target = lperm * rperm
        target_perms.append(str(target.sigma))
        lindex.append(index[lperm.sigma])
        rindex.append(index[rperm.sigma])
        target_index.append(index[target.sigma])
    return pl.DataFrame({
        "index_left": lindex,
        "index_right": rindex,
        "index_target": target_index,
        "permutation_left": left_perms,
        "permutation_right": right_perms,
        "permutation_target": target_perms 
    })



@group_registry.register("Sn")
def sn_mult_table(n: int):
    Sn = Symmetric(n)
    return _make_multiplication_table(Sn.elements)


@group_registry.register("An")
def sn_mult_table(n: int):
    An = Alternating(n)
    return _make_multiplication_table(An.elements)


@group_registry.register("Dn")
def sn_mult_table(n: int):
    Dn = Dihedral(n)
    return _make_multiplication_table(Dn.elements)
    