"""Small reproducible instances and solver-independent assignment oracles."""

from collections import Counter
from itertools import combinations, permutations, product

import numpy as np
import pytest


ALGORITHMS = ("Am", "Bm", "Cm", "Dm", "Em", "Fm")
DTYPES = (pytest.param(np.float64, id="float"), pytest.param(np.int64, id="int"))


def random_instance(k, *, seed=0, dtype=np.float64, equal=True, n=3, sizes=None):
    """Return pair matrices in combinations order, using a fresh seeded RNG.

    Unequal instances cycle through sizes n-1, n, n+1. Explicit ``sizes``
    can instead describe a particular regression instance.
    """
    if sizes is None:
        sizes = (n,) * k if equal else tuple(n - 1 + p % 3 for p in range(k))
    if len(sizes) != k or min(sizes) < 1:
        raise ValueError("Expected one positive size per partite")
    rng = np.random.default_rng(seed)
    matrices = []
    for a, b in combinations(range(k), 2):
        shape = (sizes[a], sizes[b])
        if np.issubdtype(dtype, np.integer):
            matrix = rng.integers(-20, 40, size=shape).astype(dtype)
        else:
            matrix = rng.uniform(-20, 40, size=shape).astype(dtype)
        matrices.append(matrix)
    return matrices


def clique_cost(cost_matrices, k, matches):
    """Sum all original pairwise edges, without kap's indexing or cost helpers."""
    pair_costs = dict(zip(combinations(range(k), 2), cost_matrices))
    return sum(
        float(pair_costs[a, b][i, j])
        for match in matches
        for (a, i), (b, j) in combinations(sorted(match), 2)
    )


def brute_force_optimum(cost_matrices, k):
    """Enumerate (n!)**(k-1) complete equal-size assignments, anchoring partite 0.

    This deliberately shares no matching or quotient logic with kap. Keep
    callers tiny: at most (k, n) = (3, 4), (4, 3), or (5, 2).
    """
    n = cost_matrices[0].shape[0]
    if len(cost_matrices) != k * (k - 1) // 2 or any(
        matrix.shape != (n, n) for matrix in cost_matrices
    ):
        raise ValueError("Brute force requires equal-size square pair matrices")
    orderings = tuple(permutations(range(n)))
    return min(
        clique_cost(
            cost_matrices,
            k,
            [list(enumerate(nodes)) for nodes in zip(range(n), *others)],
        )
        for others in product(orderings, repeat=k - 1)
    )


def assert_valid_assignment(cost_matrices, sizes, result):
    """Check coverage, clique structure, and every independently computed cost."""
    k = len(sizes)
    assert result.free is not None
    assert len(result.matches) == len(result.matching_costs)
    used = []
    for match, reported_cost in zip(result.matches, result.matching_costs):
        assert 2 <= len(match) <= k
        partites = [p for p, _ in match]
        assert partites == sorted(set(partites))
        used.extend(tuple(node) for node in match)
        assert np.isclose(reported_cost, clique_cost(cost_matrices, k, [match]))
    used.extend(tuple(node) for node in result.free)
    expected = [(p, node) for p, size in enumerate(sizes) for node in range(size)]
    assert Counter(used) == Counter(expected)
    assert np.isclose(sum(result.matching_costs), clique_cost(cost_matrices, k, result.matches))
    if len(set(sizes)) == 1:
        assert result.free == []
        assert len(result.matches) == sizes[0]
        assert all(len(match) == k for match in result.matches)


@pytest.fixture
def instance_factory():
    return random_instance


@pytest.fixture
def scipy_assignment():
    return pytest.importorskip("scipy.optimize").linear_sum_assignment


@pytest.fixture(params=("scipy", "lap", "lapjv", "munkres"))
def backend(request):
    pytest.importorskip(request.param)
    return request.param


@pytest.fixture(params=("scipy", "lap", "munkres"))
def rectangular_backend(request):
    pytest.importorskip(request.param)
    return request.param


@pytest.fixture
def paper_costs():
    return [
        np.array([[9, 6, 4], [4, 4, 6], [3, 9, 2]]),
        np.array([[8, 2, 9], [5, 0, 4], [8, 7, 4]]),
        np.array([[4, 0, 9], [9, 9, 6], [8, 9, 5]]),
    ]
