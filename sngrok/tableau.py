from copy import deepcopy
from functools import cache, total_ordering


def _check_shape(partition_shape):
    for i in range(len(partition_shape) - 1):
        j = i + 1
        if partition_shape[j] > partition_shape[i]:
            raise ValueError(
                f'Partition {partition_shape} is not in decreasing order.'
            )

def _subpartitions(part):
    assert len(part) >= 2
    new_parts = []
    if part[0] == 1:
        yield part
    part = list(part)
    for i in range(len(part) - 1):
        currval = part[i]
        nextval = part[i+1]
        if (currval - nextval) > 1:
            new_part = deepcopy(part)
            new_part[i] = currval - 1
            new_part[i + 1] = nextval + 1
            new_parts.append(tuple(new_part))
        if currval > 1 and nextval == 1:
            new_part = deepcopy(part)
            new_part[i] = currval - 1
            new_part.append(1)
            new_parts.append(tuple(new_part))
    if part[-1] > 1:
        new_part = deepcopy(part)
        lastval = new_part[-1]
        new_part[-1] = lastval - 1
        new_part.append(1)
        new_parts.append(tuple(new_part))
    for subpart in new_parts:
        yield subpart
        for subsub in _subpartitions(subpart):
            yield subsub


@cache
def _generate_partitions(n):
    if n == 3:
        partitions = [(3,), (2, 1), (1, 1, 1)]
    elif n == 2:
        partitions = [(2,), (1, 1)]
    elif n == 1:
        partitions = [(1,)]
    elif n == 0:
        return ()
    else:
        partitions = [(n,)]
        for k in range(n):
            m = n - k
            partitions.extend(
                tuple(
                    sorted((m, *p), reverse=True)
                ) for p in _generate_partitions(k)
            )
    return partitions


def check_parity(partition):
    even_cycles = [c for c in partition if (c % 2 == 0)]
    return len(even_cycles) % 2


def generate_partitions(n):
    return sorted(list(set(_generate_partitions(n))))


def conjugate_partition(partition):
    n = sum(partition)
    conj_part = []
    for i in range(n):
        reverse = [j for j in partition if j > i]
        if reverse:
            conj_part.append(len(reverse))
    return tuple(conj_part)
    

@total_ordering
class YoungTableau:

    def __init__(self, values: list[list[int]]):
        self.values = values
        self.shape = tuple([len(row) for row in values])
        self.n = sum(self.shape)

    def __repr__(self):
        strrep = []
        for row in self.values:
            strrep.append('|' + '|'.join([str(v) for v in row]) + '|' )
        return '\n'.join(strrep)

    def __len__(self):
        return self.n
    
    def __getitem__(self, key):
        i, j = key
        return self.values[i][j]
    
    def __setitem__(self, key, value):
        i, j = key
        self.values[i][j] = value

    def __eq__(self, other):
        if not isinstance(other, YoungTableau):
            other = YoungTableau(other)
        if (self.n != other.n) or (self.shape != other.shape):
            return False
        for row1, row2 in zip(self.values, other.values):
            if row1 != row2:
                return False
        return True

    def __lt__(self, other):
        if not isinstance(other, YoungTableau):
            other = YoungTableau(other)
        if (self.n != other.n) or (self.shape != other.shape):
            raise ValueError('Can only compare two tableau of the same shape')
        for row1, row2 in zip(self.values, other.values):
            if row1 == row2:
                continue
            for v1, v2 in zip(row1, row2):
                if v1 == v2:
                    continue
                return v1 < v2

    def index(self, val):
        for r, row in enumerate(self.values):
            if val in row:
                c = row.index(val)
                return r, c
        raise ValueError(f'{val} could not be found in tableau.')
    
    def transposition_dist(self, x: int, y: int) -> int:
        #assert y == x + 1
        row_x, col_x = self.index(x)
        row_y, col_y = self.index(y)
        row_dist = row_x - row_y
        col_dist = col_y - col_x
        return row_dist + col_dist


def _enumerate_next_placements(unfinished_syt):
    indices = []
    for i, row in enumerate(unfinished_syt):
        for j, el in enumerate(row):
            if el >= 0:
                continue
            elif (i == 0) or (unfinished_syt[i-1][j] >= 0):
                indices.append((i, j))
                break
    return indices


def _fill_unfinished_tableau(tableau, numbers):
    possible_placements = _enumerate_next_placements(tableau)
    val = numbers.pop()
    new_tableaus = []
    for i, j in possible_placements:
        new_tableau = deepcopy(tableau)
        new_tableau[i][j] = val
        new_tableaus.append(new_tableau)
    if len(numbers) == 0:
        return [YoungTableau(t) for t in new_tableaus]
    else:
        all_tableaus = []
        for t in new_tableaus:
            all_tableaus += _fill_unfinished_tableau(t, deepcopy(numbers))
        return all_tableaus


def enumerate_standard_tableau(partition_shape: tuple[int]) -> list[YoungTableau]:
    _check_shape(partition_shape)
    n = sum(partition_shape)
    base_tableau = [[-1] * l for l in partition_shape]
    numbers = list(range(n))
    numbers.reverse()
    base_tableau[0][0] = numbers.pop()
    all_tableaus = _fill_unfinished_tableau(base_tableau, numbers)
    return sorted(all_tableaus)
