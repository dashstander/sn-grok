from copy import deepcopy
from functools import total_ordering
from itertools import chain


def _check_shape(partition_shape):
    for i in range(len(partition_shape) - 1):
        j = i + 1
        if partition_shape[j] > partition_shape[i]:
            raise ValueError(
                f'Partition {partition_shape} is not in decreasing order.'
            )


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
        for i, row in enumerate(self.values):
            if val in row:
                j = row.index(val)
                return i, j
        raise ValueError(f'{val} could not be found in tableau.')
    
    def transposition_dist(self, x: int, y: int) -> int:
        #assert y == x + 1
        ix, jx = self.index(x)
        iy, jy = self.index(y)
        row_dist = iy - ix
        col_dist = jx - jy
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
        return new_tableaus
    else:
        return [_fill_unfinished_tableau(t, deepcopy(numbers)) for t in new_tableaus]


def enumerate_standard_tableau(partition_shape: tuple[int]) -> list[YoungTableau]:
    _check_shape(partition_shape)
    n = sum(partition_shape)
    base_tableau = [[-1] * l for l in partition_shape]
    numbers = list(range(n))
    numbers.reverse()
    base_tableau[0][0] = numbers.pop()
    all_tableaus = _fill_unfinished_tableau(base_tableau, numbers)
    return [YoungTableau(t) for t in chain(all_tableaus)]
