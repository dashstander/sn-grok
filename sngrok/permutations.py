from functools import total_ordering
from itertools import permutations, product
import polars as pl


@total_ordering
class Permutation:
    
    def __init__(self, sigma):
        self.sigma = tuple(sigma)
        self.base = list(range(len(sigma)))
        self._cycle_rep = None
    
    def __repr__(self):
        return f'Permutation {list(self.sigma)}'
    
    def __len__(self):
        return len(self.sigma)

    def __hash__(self):
        return hash(self.sigma)

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
    
    def __call__(self, x):
        if len(x) != len(self):
            raise ValueError(f'Permutation of length {len(self)} is ill-defined for given sequence of length {len(x)}')
        if isinstance(x, Permutation):
            sequence = x.sigma
            new_sigma = [sequence[self.sigma[i]] for i in self.base]
            return Permutation(new_sigma)
        else:
            return [x[self.sigma[i]] for i in self.base]
    
    @property
    def cycle_rep(self):
        if self._cycle_rep is None:
            elems = set(self.sigma)
            base = list(range(len(self)))
            cycles = []
            i = 0
            while len(elems) > 0:
                this_cycle = []
                curr = min(elems)
                while curr not in this_cycle:
                    this_cycle.append(curr)
                    curr = base[self.sigma[curr]]
                cycles.append(this_cycle)
                elems = elems - set(this_cycle)
                i += 1
            self._cycle_rep = cycles
        return self._cycle_rep
    
    @property
    def parity(self):
        odd_cycles = [c for c in self.cycle_rep if (len(c) % 2 == 0)]
        return len(odd_cycles) % 2
    
    @property
    def congruency_class(self):
        cycle_lens = [len(c) for c in self.cycle_rep]
        return tuple(sorted(cycle_lens))

    def data(self):
        return {
            'permutation': self.sigma,
            'cycle_rep': self.cycle_rep,
            'congruency_class': self.congruency_class,
            'parity': self.parity
        }

def make_permutation_dataset(n: int):
    mult_table = []
    perms = []
    index = {}
    data = []
    perms = [Permutation(seq) for seq in permutations(list(range(n)))]
    perms = sorted(perms)
    for i, p in enumerate(perms):
        index[p.sigma] = i
        data.append(p.data())
    for perm1, perm2 in product(perms, repeat=2):
        q = perm1(perm2)
        mult_table.append((index[perm1.sigma], index[perm2.sigma], index[q.sigma]))
    perm_df = pl.DataFrame(data).with_row_count(name='index')
    return perm_df, mult_table
    
