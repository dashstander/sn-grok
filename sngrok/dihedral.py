from copy import deepcopy
from functools import reduce
from itertools import product
import math
from operator import mul


class Dihedral:

    def __init__(self, rot: int, ref: int, n: int):
        self.n = n
        self.rot = rot % n
        self.ref = ref % 2

    def __repr__(self):
        return str((self.rot, self.ref))
    
    def __hash__(self):
        return hash(str(self))
    
    @classmethod
    def full_group(cls, n):
        return [
            Dihedral(r, p, n) for r, p in product(range(n), [0, 1])
        ]
    
    @property
    def order(self):
        if self.ref:
            return 2
        elif self.rot == 0:
            return 0
        elif (self.n % self.rot) == 0:
            return self.n // self.rot
        else:
            return math.lcm(self.n, self.rot) // self.rot
        
    @property
    def inverse(self):
        if self.ref:
            return deepcopy(self)
        else:
            return Dihedral(self.n - self.rot, self.ref, self.n)
    
    def __mul__(self, other):
        if (not isinstance(other, Dihedral)) or (other.n != self.n):
            raise ValueError(
                'Can only multiply a dihedral rotation with another dihedral rotation'
            )
        if self.ref:
            rot = self.rot - other.rot
        else:
            rot = self.rot + other.rot
        return Dihedral(rot, self.ref + other.ref, self.n)
    
    def __pow__(self, x: int):
        if x == -1:
            return self.inverse
        elif x == 0:
            return Dihedral(0, 0, self.n)
        elif x == 1:
            return deepcopy(self)
        else:
            return reduce(mul, [deepcopy(self) for _ in range(x)])
