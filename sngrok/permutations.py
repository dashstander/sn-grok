from copy import deepcopy
from functools import reduce, total_ordering
from itertools import pairwise, permutations
import operator
from sngrok.tableau import YoungTableau


@total_ordering
class Permutation:
    
    def __init__(self, sigma):
        self.sigma = tuple(sigma)
        self.n = len(sigma)
        self.base = list(range(self.n))
        self._cycle_rep = None
        self._inverse = None
        self._order = None

    @classmethod
    def full_group(cls, n: int):
        return sorted([
            cls(seq) for seq in permutations(list(range(n)))
        ])

    @classmethod
    def identity(cls, n: int):
        return cls(list(range(n)))

    @classmethod
    def transposition(cls, n, i, j):
        assert i < j and j <= (n - 1)
        basis = list(range(n))
        basis[i] = j
        basis[j] = i
        return cls(basis)

    def is_identity(self):
        ident = tuple(list(range(self.n)))
        return ident == self.sigma
    
    def __repr__(self):
        return str(self.sigma)
    
    def __len__(self):
        return self.n

    def __hash__(self):
        return hash(str(self.sigma))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Permutation):
            return False
        return self.sigma == other.sigma

    def __lt__(self, other):
        if not isinstance(other, Permutation):
            raise ValueError('Can only compare against another permutation')
        elif self.parity != other.parity:
            return self.parity < other.parity
        else:
            return self.sigma < other.sigma
    
    def __gt__(self, other):
        if not isinstance(other, Permutation):
            raise ValueError('Can only compare against another permutation')
        elif self.parity != other.parity:
            return self.parity > other.parity
        else:
            return self.sigma > other.sigma
    
    def __mul__(self, x):
        if len(x) != len(self):
            raise ValueError(f'Permutation of length {len(self)} is ill-defined for given sequence of length {len(x)}')
        if isinstance(x, Permutation):
            sequence = x.sigma
            new_sigma = [sequence[self.sigma[i]] for i in self.base]
            return Permutation(new_sigma)
        elif isinstance(x, YoungTableau):
            # TODO(dashiell): Implement permutations acting on tableau
            #index_map = {a : x.index(a) for a in self.base}
            vals = [[-1] * s  for s in x.shape]
            for j, i in enumerate(self.sigma):
                #ix, iy = x.index(i)
                jx, jy = x.index(j)
                vals[jx][jy] = i
            return YoungTableau(vals)
                
        else:
            return [x[self.sigma[i]] for i in self.base]
    
    def __pow__(self, exponent: int):
        if not isinstance(exponent, int):
            raise ValueError('Can only raise permutations to an integer power')
        elif exponent == 0:
            return Permutation(list(range(len(self))))
        elif exponent == 1:
            return deepcopy(self)
        if exponent > 0:
            perm = deepcopy(self)
        else:
            perm = self.inverse
        perm_copies = [deepcopy(perm) for _ in range(exponent)]
        return reduce(operator.mul, perm_copies)

    def _calc_cycle_rep(self):
        elems = set(self.sigma)
        cycles = []
        i = 0
        while len(elems) > 0:
            this_cycle = []
            curr = min(elems)
            while curr not in this_cycle:
                this_cycle.append(curr)
                curr = self.sigma.index(curr)
            cycles.append(tuple(this_cycle))
            elems = elems - set(this_cycle)
            i += 1
        return sorted(cycles, key = lambda x: (len(x), *x), reverse=True)
    
    @property
    def cycle_rep(self):
        if self._cycle_rep is None:
            self._cycle_rep = self._calc_cycle_rep()
        return self._cycle_rep
    
    @property
    def parity(self):
        even_cycles = [c for c in self.cycle_rep if (len(c) % 2 == 0)]
        return len(even_cycles) % 2
    
    @property
    def conjugacy_class(self):
        cycle_lens = [len(c) for c in self.cycle_rep]
        return tuple(sorted(cycle_lens, reverse=True))

    @property
    def inverse(self):
        inv = [-1] * self.n
        for i, val in enumerate(self.sigma):
            inv[val] = i
        return Permutation(inv)    
    
    @property
    def order(self):
        if self._order is not None:
            return self._order
        perm = deepcopy(self)
        i = 1
        while not perm.is_identity():
            perm = self * perm
            i += 1
        self._order = i
        return i
    
    def transposition_decomposition(self):
        transpositions = []
        for cycle in self.cycle_rep:
            if len(cycle) > 1:
                transpositions.extend([tuple(sorted(pair)) for pair in pairwise(cycle)])
        return transpositions
    
    def adjacent_transposition_decomposition(self):
        adjacent_transpositions = []
        for transposition in self.transposition_decomposition():
            i, j = sorted(transposition)
            if j == i + 1:
                decomp = [(i, j)]
            else:
                center = [(i, i + 1)]
                i_to_j = list(range(i+1, j+1))
                adj_to_j = list(pairwise(i_to_j))
                decomp = reversed(adj_to_j) + center + adj_to_j
            adjacent_transpositions.append(decomp)
        return adjacent_transpositions
