class Permutation:
    
    def __init__(self, sigma):
        self.sigma = tuple(sigma)
        self.base = list(range(len(sigma)))
        self._cycle_rep = None
    
    def __len__(self):
        return len(self.sigma)
    
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
    
    def congruency_class(self):
        cycle_lens = [len(c) for c in self.cycle_rep]
        return tuple(sorted(cycle_lens))
    
    @property
    def parity(self):
        odd_cycles = [c for c in self.cycle_rep if (len(c) % 2 == 0)]
        return len(odd_cycles) % 2

