from functools import partial, total_ordering
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
    def congruency_class(self):
        cycle_lens = [len(c) for c in self.cycle_rep]
        return tuple(sorted(cycle_lens))


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
    cong_classes = []
    parities = []
    perms = [Permutation(seq) for seq in permutations(list(range(n)))]
    perms = sorted(perms)
    for i, p in enumerate(perms):
        one_lines.append(p.sigma)
        cycle_reps.append(p.cycle_rep)
        cong_classes.append(p.congruency_class)
        parities.append(p.parity)
        index[p.sigma] = i
    perm_df = pl.from_dict({
        'permutation': one_lines,
        'cycle_rep': cycle_reps,
        'congruency_class': cong_classes,
        'parity': parities
    }).with_row_count(name='index')

    _match = partial(get_index, perm_df)

    mult_df = perm_df.join(perm_df, on='permutation', how='cross')
    mult_df = mult_df.with_columns(
        pl.col('permutation_right').arr.take(pl.col('permutation')).alias('result'))
    mult_df = mult_df.with_columns(
        pl.col('result').apply(_match).alias('result_index')
    )
    return perm_df, mult_df


def generate_subgroup(generators: list[tuple[int]]):
    group_size = 0
    all_perms = set(generators)
    while group_size < len(all_perms):
        group_size = len(perms)
        perms = [Permutation(p) for p in all_perms]
        for perm1, perm2 in product(perms, repeat=2):
            perm3 = perm1(perm2)
            all_perms.add(perm3.sigma)
    return list(all_perms)
