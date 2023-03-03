import polars as pl

_permutations = [(1, 2, 3), (1, 3, 2), (2, 1, 3), (3, 1, 2), (2, 3, 1), (3, 2, 1)]
_city_ranks = [1, 1, 2, 3, 2, 3]
_subrubs_ranks = [2, 3, 1, 1, 3, 2]
_country_ranks = [3, 2, 3, 2, 1, 1]
_count = [242, 28, 170, 628, 12, 359]

survey_data = pl.DataFrame({
    'permutation': _permutations,
    'city': _city_ranks,
    'suburbs': _subrubs_ranks,
    'country': _country_ranks,
    'count': _count
})
