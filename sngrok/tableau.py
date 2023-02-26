from copy import deepcopy
from itertools import chain


def _check_shape(partition_shape):
    for i in range(len(partition_shape) - 1):
        j = i + 1
        if partition_shape[j] > partition_shape[i]:
            raise ValueError(f'Partition {partition_shape} is not in decreasing order.')


class YoungTableau:

    def __init__(self, values):
        self.values = values
        self.shape = tuple([len(row) for row in values])
        self.n = sum(self.shape)
    
    def __getitem__(self, i, j):
        return self.values[i][j]

    def index(self, val):
        for i, row in enumerate(self.values):
            if val in row:
                j = row.index(val)
                return i, j
        raise ValueError(f'{val} could not be found in tableau.')
    
    def adjacent_transposition_dist(self, x, y):
        assert y == x + 1
        ix, jx = self.index(x)
        iy, jy = self.index(y)
        row_dist = iy - ix
        col_dist = jx - jy
        return row_dist + col_dist


def _enumerate_next_placements(unfinished_syt):
    indices = []
    for i, row in enumerate(unfinished_syt):
        for j, el in enumerate(row):
            if el > 0:
                continue
            elif (i == 0) or (unfinished_syt[i-1][j] > 0):
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


def enumerate_standard_tableau(partition_shape):
    _check_shape(partition_shape)
    n = sum(partition_shape)
    base_tableau = [[0] * l for l in partition_shape]
    numbers = list(range(1, n+1))
    numbers.reverse()
    base_tableau[0][0] = numbers.pop()
    all_tableaus = _fill_unfinished_tableau(base_tableau, numbers)
    return [YoungTableau(t) for t in chain(all_tableaus)]
