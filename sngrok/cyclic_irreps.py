import numpy as np


class CyclicIrrep:

    def __init__(self, n: int, element: int):
        self.n = n
        self.conjugacy_class = element % n
        self.group = [i for i in range(n)]

    def _trivial_irrep(self):
        return {r: np.ones((1,)) for r in self.group}
    
    def _matrix_irrep(self, k):
        mats = {}
        mats[0] = np.eye(2)
       
        two_pi_n = 2 * np.pi / self.n
        for l in range(1, self.n):
            sin = np.sin(two_pi_n * l * k)
            cos = np.cos(two_pi_n * l * k)
            m = np.array([[cos, -1.0 * sin], [sin, cos]])
            mats[l] = m

        return mats
    
    def matrix_representations(self):
        if self.conjugacy_class == 0:
            return self._trivial_irrep()
        else:
            return self._matrix_irrep(self.conjugacy_class)

