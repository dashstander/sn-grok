import pytest

from itertools import product
import math
import numpy as np
from numpy.linalg import matrix_power
from sngrok.irreps import make_irrep
from sngrok.permutations import Permutation
from sngrok.tableau import generate_partitions


@pytest.mark.parametrize('n', [3, 4, 5])
def test_matrix_reps(n):
    partitions = generate_partitions(n)
    for part in partitions:
        irrep = make_irrep(part)
        ident = np.eye(irrep.dim)
        matrices = irrep.matrix_representations()
        assert len(matrices) == math.factorial(n)
        for k, v in matrices.items():
            perm = Permutation(k)
            if isinstance(v, np.ndarray):
                assert np.allclose(v @ v.T, ident)
                assert np.allclose(matrix_power(v, perm.order), ident)
            else:
                assert isinstance(v, float)
                assert (v ** perm.order) == 1.
        for k1, k2 in product(matrices.keys(), matrices.keys()):
            perm1, perm2 = Permutation(k1), Permutation(k2)
            perm3 = perm1 * perm2
            mat3 = matrices[perm3.sigma]
            if isinstance(v, np.ndarray):
                assert np.allclose(
                    mat3,
                    matrices[perm1.sigma] @ matrices[perm2.sigma]
                )
            else:
                assert mat3 == matrices[perm1.sigma] * matrices[perm2.sigma]
