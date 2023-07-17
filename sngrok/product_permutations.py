
from copy import deepcopy
from functools import reduce, total_ordering
from itertools import product
from operator import mul
from sngrok.permutations import Permutation


@total_ordering
class ProductPermutation:

    def __init__(self, sigma: tuple[tuple[int]]):

        self.sigma = tuple(sigma)
        self.num_groups = len(sigma)
        self.perms = [Permutation(p) for p in sigma]
        self.ns = [perm.n for perm in self.perms]
        self.base = [perm.base for perm in self.perms]
        self._cycle_rep = None
        self._inverse = None
        self._order = None

    def __repr__(self):
        return str(self.sigma)
    
    def __hash__(self):
        return hash(str(self.sigma))
    
    @classmethod
    def full_group(cls, ns: list[int]):
        return [
            ProductPermutation([p.sigma for p in perms]) for perms in product(*[Permutation.full_group(n) for n in ns])
        ]
    
    @classmethod
    def identity(cls, ns: list[int]):
        ids = [list(range(n)) for n in ns]
        return ProductPermutation(ids)
    
    @property
    def inverse(self):
        if self._inverse is None:
            invs = [p.inverse.sigma for p in self.perms]
            self._inverse = ProductPermutation(tuple(invs))
        return self._inverse
    
    @property
    def order(self):
        if self._order is None:
            total_order = reduce(mul, [p.order for p in self.perms])
            self._order = total_order
        return self._order
    
    @property
    def conjugacy_class(self):
        return tuple([p.conjugacy_class for p in self.perms])
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, ProductPermutation):
            return False
        else:
            return self.sigma == other.sigma
    
    def __lt__(self, other) -> bool:
        if not isinstance(other, ProductPermutation):
            raise ValueError('Must compare to another ProductPermutation')
        else:
            return self.sigma < other.sigma
    
    def __gt__(self, other) -> bool:
        if not isinstance(other, ProductPermutation):
            raise ValueError('Must compare to another ProductPermutation')
        else:
            return self.sigma > other.sigma
    
    def __mul__(self, x):
        if len(x.perms) != len(self.perms):
            raise ValueError('Can only multiply permutations from the same product group')
        prod = [(l * r).sigma for l, r in zip(self.perms, x.perms)]
        return ProductPermutation(tuple(prod))
    
    def __pow__(self, x: int):
        if x == 0:
            return ProductPermutation.identity(self.ns)
        elif x == -1:
            return self.inverse
        elif x == 1:
            return deepcopy(self)
        elif x > 1:
            copies = [deepcopy(self) for _ in range(x)]
            return reduce(mul, copies)
        else:
            raise NotImplementedError
