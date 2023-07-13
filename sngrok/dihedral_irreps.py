from itertools import product
import numpy as np
from sngrok.dihedral import Dihedral


def generate_subgroup(generators, n):
    group_size = 0
    all_elements = set(generators)
    while group_size < len(all_elements):
        group_size = len(all_elements)
        rotations = [Dihedral(*p) for p in all_elements]
        for r1, r2 in product(rotations, repeat=2):
            r3 = r1 * r2
            all_elements.add(r3.sigma)
    return list(all_elements)


def dihedral_conjugacy_classes(n: int):
    conj_classes = [(0, 0), (0, 1)]
    if n % 2 == 1:
        conj_classes += [(i, 0) for i in range(1, (n + 1) // 2)]
    else:
        conj_classes += [((2, 0), (0, 1)), ((2, 1), (0, 1))]
        conj_classes += [(i, 0) for i in range(1, n // 2)]
    return conj_classes


class DihedralIrrep:

    def __init__(self, n: int, conjugacy_class):
        self.n = n
        self.conjugacy_class = conjugacy_class
        self.group = Dihedral.full_group(n)

    def _trivial_irrep(self):
        return {r.sigma: np.ones((1,)) for r in self.group}
    
    def _reflection_irrep(self):
        return {r.sigma: (-1**r.ref) * np.ones((1,)) for r in self.group}
    
    def _subgroup_irrep(self, sg):
        return {r.sigma: -1 if r in sg else 1 for r in self.group}
    
    def _matrix_irrep(self, k):
        mats = {}
        mats[(0, 0)] = np.eye(2)
        ref = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
        mats[(0, 1)] = ref
        two_pi_n = 2 * np.pi / self.n
        for l in range(1, self.n):
            sin = np.sin(two_pi_n * l * k)
            cos = np.sin(two_pi_n * l * k)
            m = np.array([[cos, -1.0 * sin], [sin, cos]])
            mats[(l, 0)] = m
            mats[(l, 1)] = ref @ m
        return mats
    
    def matrix_representations(self):
        if self.conjugacy_class == (0, 0):
            return self._trivial_irrep()
        elif self.conjugacy_class == (0, 1):
            return self._reflection_irrep()
        elif self.conjugacy_class[1] == 0:
            return self._matrix_irrep(self.conjugacy_class[0])
        elif (
            isinstance(self.conjugacy_class[0], tuple) and 
            isinstance(self.conjugacy_class[1], tuple)
        ):
            subgroup = generate_subgroup(self.conjugacy_class, self.n)
            return self._subgroup_irrep(subgroup)
        else:
            raise ValueError(
                f'Somehow {self.conjugacy_class} is not a proper conjugacy class....'
            )
