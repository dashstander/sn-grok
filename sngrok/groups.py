import catalogue
from functools import reduce
from itertools import pairwise, product
import math
import numpy as np
from operator import mul
import polars as pl
from sngrok.dihedral import Dihedral
from sngrok.permutations import Permutation
from sngrok.product_permutations import ProductPermutation
from sngrok.fourier import slow_an_ft_1d, slow_dihedral_ft, slow_sn_ft_1d, slow_product_sn_ft, slow_cyclic_ft
from sngrok.tableau import conjugate_partition, generate_partitions
from sngrok.irreps import SnIrrep
from sngrok.dihedral_irreps import DihedralIrrep, dihedral_conjugacy_classes
from sngrok.cyclic_irreps import CyclicIrrep


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


def _make_multiplication_table(all_permutations):
    index = {perm.sigma : i for i, perm in enumerate(all_permutations)}
    left_perms = []
    lindex = []
    rindex = []
    right_perms = []
    target_perms = []
    target_index = []
    for lperm, rperm in product(all_permutations, all_permutations):
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

def _make_modular_addition_table(n):
    left_perms = []
    lindex = []
    rindex = []
    right_perms = []
    target_perms = []
    target_index = []
    for lperm, rperm in product(list(range(n)), list(range(n))):
        left_perms.append(str(lperm))
        right_perms.append(str(rperm))
        target = (lperm + rperm) % n
        target_perms.append(str(target))
        lindex.append(lperm)
        rindex.append(rperm)
        target_index.append(target)

    return pl.DataFrame({
        "index_left": lindex,
        "index_right": rindex,
        "index_target": target_index,
        "permutation_left": left_perms,
        "permutation_right": right_perms,
        "permutation_target": target_perms 
    })

class PermutationGroup:

    def make_multiplication_table(self):
        return _make_multiplication_table(self.elements)

class CyclicGroup:
    def __init__(self, n: int):
        self.n = n
        self.order = n
        self.elements = [i for i in range(n)]

    def make_modular_addition_table(self):
        return _make_modular_addition_table(self.n)
    
    def irreps(self):
        return {
            el : CyclicIrrep(self.n, el).matrix_representations() for el in self.elements
        }
    
    def fourier_transform(self, tensor):
        return slow_cyclic_ft(tensor, self.irreps(), self.n)
    

class Symmetric(PermutationGroup):
    def __init__(self, n: int):
        self.n = n
        self.elements = Permutation.full_group(n)
        self.order = math.factorial(n)
        self.group = "symmetric"

    def fourier_transform(self, tensor):
        return slow_sn_ft_1d(tensor, self.n)
    
    def irreps(self):
        partitions = generate_partitions(self.n)
        return {p: SnIrrep(self.n, p) for p in partitions}


class Alternating(PermutationGroup):

    def __init__(self, n: int):
        self.n = n
        if n < 3:
            raise ValueError('No alternating subgroup of S2')
        self.generators = [
            three_cycle_to_one_line(i, n) for i in range(2, n)
        ]
        self.elements = [
            Permutation(p) for p in generate_subgroup(self.generators)
        ]
        self.order = math.factorial(n) / 2
        self.group = "alternating"

    def fourier_transform(self, tensor):
        return slow_an_ft_1d(tensor, self.n)
    
    def irreps(self):
        all_partitions = set()
        partitions = []
        for p in generate_partitions(self.n):
            if p in all_partitions:
                continue
            else:
                partitions.append(p)
                all_partitions.add(p)
                all_partitions.add(conjugate_partition(p))
        return {
            p: SnIrrep(self.n, p) for p in partitions
        }
    

class ProductSymmetric(PermutationGroup):

    def __init__(self, ns: list[int]):
        self.ns = ns
        self.elements = ProductPermutation.full_group(ns)
        self.order = reduce(mul, [math.factorial(x) for x in ns])
        self.group = "product"
    
    def irreps(self):
        irreps = {}
        for parts in product(*[generate_partitions(n) for n in self.ns]):
            sn_irreps = [SnIrrep(n, p).matrix_representations() for n, p in zip(self.ns, parts)]
            full_mats = {}
            for perms in self.elements:
                matrices = [irrep[p] for irrep, p in zip(sn_irreps, perms.sigma)]
                full_mats[perms.sigma] = reduce(np.kron, matrices)
            irreps[parts] = full_mats
        return irreps
    
    def fourier_transform(self, tensor):
        return slow_product_sn_ft(tensor, self.irreps(), self.ns)


class DihedralGroup:

    def __init__(self, n: int):
        if n < 3:
            raise ValueError('We start with triangles')
        self.n = n
        self.order = 2 * n
        self.elements = Dihedral.full_group(n)
    
    def irreps(self):
        conj_classes = dihedral_conjugacy_classes(self.n)
        return {
            conj: DihedralIrrep(self.n, conj).matrix_representations() for conj in conj_classes
        }

    def fourier_transform(self, tensor):
        return slow_dihedral_ft(tensor, self.irreps(), self.n)
    
    def make_multiplication_table(self):
        index = {r.sigma : i for i, r in enumerate(self.elements)}
        left_elements = []
        lindex = []
        rindex = []
        right_elements = []
        target_elements = []
        target_index = []
        for lr, rr in product(self.elements, self.elements):
            left_elements.append(str(lr.sigma))
            right_elements.append(str(rr.sigma))
            target = lr * rr
            target_elements.append(str(target.sigma))
            lindex.append(index[lr.sigma])
            rindex.append(index[rr.sigma])
            target_index.append(index[target.sigma])
        return pl.DataFrame({
            "index_left": lindex,
            "index_right": rindex,
            "index_target": target_index,
            "element_left": left_elements,
            "elemenet_right": right_elements,
            "element_target": target_elements 
        })



@group_registry.register("Sn")
def sn(n: int):
    return Symmetric(n)


@group_registry.register("An")
def an(n: int):
    return Alternating(n)


@group_registry.register("Dn")
def dn(n: int):
    return DihedralGroup(n)


@group_registry.register("ProdSn")
def prod_sn(ns: list[int]):
    return ProductSymmetric(ns)

@group_registry.register("Cn")
def prod_sn(n):
    return CyclicGroup(n)
