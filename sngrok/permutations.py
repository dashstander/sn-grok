from copy import deepcopy
from functools import partial, reduce, total_ordering
from itertools import pairwise, permutations, product
import operator
import polars as pl
from .tableau import YoungTableau


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
    def identity(cls, n: int):
        return cls(list(range(n)))

    def is_identity(self):
        ident = tuple(list(range(self.n)))
        return ident == self.sigma
    
    def __repr__(self):
        return f'Permutation {list(self.sigma)}'
    
    def __len__(self):
        return self.n

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
    
    def __mul__(self, x):
        if len(x) != len(self):
            raise ValueError(f'Permutation of length {len(self)} is ill-defined for given sequence of length {len(x)}')
        if isinstance(x, Permutation):
            sequence = x.sigma
            new_sigma = [sequence[self.sigma[i]] for i in self.base]
            return Permutation(new_sigma)
        elif isinstance(x, YoungTableau):
            # TODO(dashiell): Implement permutations acting on tableau
            raise NotImplementedError
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
        odd_cycles = [c for c in self.cycle_rep if (len(c) % 2 == 0)]
        return len(odd_cycles) % 2
    
    @property
    def conjugacy_class(self):
        cycle_lens = [len(c) for c in self.cycle_rep]
        return tuple(sorted(cycle_lens, reverse=True))
    
    @property
    def inverse(self):
        if self._inverse is not None:
            return self._inverse
        perm = deepcopy(self)
        i = 0
        while not perm.is_identity():
            perm = self * perm
            i += 1
        self._inverse = perm
        self._order = i
        return perm
    
    @property
    def order(self):
        if self._order is not None:
            return self._order
        perm = deepcopy(self)
        i = 0
        while not perm.is_identity():
            perm = self * perm
            i += 1
        self._order = i
        return i
    
    def transposition_decomposition(self):
        transpositions = []
        for cycle in self.cycle_rep:
            if len(cycle) > 1:
                transpositions.append(list(pairwise(cycle)))
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


def get_index(df, result):
    perm = result.to_list()
    return (
        df.filter(
            pl.col('permutation').apply(lambda x: x.to_list() == perm)
        ).select('index').item()
    )


def make_permutation_dataset(n: int):
    perms = []
    index = {}
    one_lines = []
    cycle_reps = []
    conj_classes = []
    parities = []
    perms = [Permutation(seq) for seq in permutations(list(range(n)))]
    perms = sorted(perms)
    for i, p in enumerate(perms):
        one_lines.append(p.sigma)
        cycle_reps.append(p.cycle_rep)
        conj_classes.append(p.conjugacy_class)
        parities.append(p.parity)
        index[p.sigma] = i
    perm_df = pl.from_dict({
        'permutation': one_lines,
        'cycle_rep': cycle_reps,
        'conjugacy_class': conj_classes,
        'parity': parities
    }).with_row_count(name='index')
    perm_df = perm_df.with_columns(pl.col('index').cast(pl.Int32))

    _match = partial(get_index, perm_df)

    mult_df = perm_df.join(perm_df, on='permutation', how='cross')
    mult_df = mult_df.with_columns(
        pl.col('permutation_right').arr.take(pl.col('permutation')).alias('result'))
    mult_df = mult_df.with_columns(
        pl.col('result').apply(_match).alias('result_index').cast(pl.Int32)
    )
    mult_df = mult_df.join(
        perm_df.select(['index', 'conjugacy_class']),
        left_on='result_index',
        right_on='index',
        suffix='_result'
    )
    new_schema = {
        'index': 'index_left',
        'permutation': 'permutation_left',
        'cycle_rep': 'cycle_rep_left',
        'conjugacy_class': 'conjugacy_class_left',
        'parity': 'parity_left',
        'index_right': 'index_right',
        'permutation_right': 'perm_right',
        'cycle_rep_right': 'cycle_rep_right',
        'conjugacy_class_right': 'conjugacy_class_right',
        'parity_right': 'parity_right',
        'result': 'permutation_target',
        'result_index': 'index_target',
        'conjugacy_class_result': 'conjugacy_class_target'
    }
    return perm_df, mult_df.rename(new_schema)


def generate_subgroup(generators: list[tuple[int]]):
    group_size = 0
    all_perms = set(generators)
    while group_size < len(all_perms):
        group_size = len(perms)
        perms = [Permutation(p) for p in all_perms]
        for perm1, perm2 in product(perms, repeat=2):
            perm3 = perm1 * perm2
            all_perms.add(perm3.sigma)
    return list(all_perms)
