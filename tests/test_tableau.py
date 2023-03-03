from sngrok.tableau import enumerate_standard_tableau, generate_partitions, YoungTableau



def test_generate_partitions():
    parts10 = [
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        (2, 1, 1, 1, 1, 1, 1, 1, 1),
        (2, 2, 1, 1, 1, 1, 1, 1),
        (2, 2, 2, 1, 1, 1, 1),
        (2, 2, 2, 2, 1, 1),
        (2, 2, 2, 2, 2),
        (3, 1, 1, 1, 1, 1, 1, 1),
        (3, 2, 1, 1, 1, 1, 1),
        (3, 2, 2, 1, 1, 1),
        (3, 2, 2, 2, 1),
        (3, 3, 1, 1, 1, 1),
        (3, 3, 2, 1, 1),
        (3, 3, 2, 2),
        (3, 3, 3, 1),
        (4, 1, 1, 1, 1, 1, 1),
        (4, 2, 1, 1, 1, 1),
        (4, 2, 2, 1, 1),
        (4, 2, 2, 2),
        (4, 3, 1, 1, 1),
        (4, 3, 2, 1),
        (4, 3, 3),
        (4, 4, 1, 1),
        (4, 4, 2),
        (5, 1, 1, 1, 1, 1),
        (5, 2, 1, 1, 1),
        (5, 2, 2, 1),
        (5, 3, 1, 1),
        (5, 3, 2),
        (5, 4, 1),
        (5, 5),
        (6, 1, 1, 1, 1),
        (6, 2, 1, 1),
        (6, 2, 2),
        (6, 3, 1),
        (6, 4),
        (7, 1, 1, 1),
        (7, 2, 1),
        (7, 3),
        (8, 1, 1),
        (8, 2),
        (9, 1),
        (10,)
    ]
    for p1, p2 in zip(generate_partitions(10), parts10):
        assert p1 == p2 


def test_enumerate_standard_tableau():
    expected_41 = [
        YoungTableau([[0, 1, 2, 3], [4]]),
        YoungTableau([[0, 1, 2, 4], [3]]),
        YoungTableau([[0, 1, 3, 4], [2]]),
        YoungTableau([[0, 2, 3, 4], [1]])
    ]
    expected_32 = [
        YoungTableau([[0, 1, 2], [3, 4]]),
        YoungTableau([[0, 1, 3], [2, 4]]),
        YoungTableau([[0, 1, 4], [2, 3]]),
        YoungTableau([[0, 2, 3], [1, 4]]),
        YoungTableau([[0, 2, 4], [1, 3]])
    ]
    expected_33 = [
        YoungTableau([[0, 1, 2], [3, 4, 5]]),
        YoungTableau([[0, 1, 3], [2, 4, 5]]),
        YoungTableau([[0, 1, 4], [2, 3, 5]]),
        YoungTableau([[0, 2, 3], [1, 4, 5]]),
        YoungTableau([[0, 2, 4], [1, 3, 5]])
    ]
    expected_42 = [
        YoungTableau([[0, 1, 2, 3], [4, 5]]),
        YoungTableau([[0, 1, 2, 4], [3, 5]]),
        YoungTableau([[0, 1, 2, 5], [3, 4]]),
        YoungTableau([[0, 1, 3, 4], [2, 5]]),
        YoungTableau([[0, 1, 3, 5], [2, 4]]),
        YoungTableau([[0, 1, 4, 5], [2, 3]]),
        YoungTableau([[0, 2, 3, 4], [1, 5]]),
        YoungTableau([[0, 2, 3, 5], [1, 4]]),
        YoungTableau([[0, 2, 4, 5], [1, 3]])
    ]
    assert enumerate_standard_tableau((4, 1)) == sorted(expected_41)
    assert enumerate_standard_tableau((3, 2)) == sorted(expected_32)
    assert enumerate_standard_tableau((3, 3)) == sorted(expected_33)
    assert enumerate_standard_tableau((4, 2)) == sorted(expected_42)
