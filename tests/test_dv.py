"""Native Shift and dimensionwise variation, with an independent LAP oracle."""

import itertools
import sys

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

import kap
from kap import _core
from conftest import assert_valid_assignment, random_instance

BASE = ("Am", "Bm", "Cm", "Dm", "Em", "Fm")
SEARCHES = [(algo, None) for algo in ("shift", "1dv", "sdv")]
SEARCHES += list(itertools.product(BASE, ("1dv", "sdv")))


@pytest.mark.parametrize("algo,refine", SEARCHES)
@pytest.mark.parametrize("k", (2, 3, 4))
@pytest.mark.parametrize("equal", (False, True))
@pytest.mark.parametrize("maximize", (False, True))
def test_valid_and_never_worse(algo, refine, k, equal, maximize):
    sizes = [3] * k if equal else [2 + p % 3 for p in range(k)]
    matrices = random_instance(k, sizes=sizes, seed=85 + k, dtype=np.int64)
    options = dict(maximize=maximize, return_free=True, random_state=42, n_trials=2)
    actual = kap.k_assignment(matrices, algo=algo, refine=refine, **options)
    assert_valid_assignment(matrices, sizes, actual)
    base_algo = algo if refine else "Am"
    base = kap.k_assignment(matrices, algo=base_algo, **options)
    cost, base_cost = sum(actual.matching_costs), sum(base.matching_costs)
    assert cost >= base_cost if maximize else cost <= base_cost
    if k == 3 and algo == "1dv":
        assert actual == kap.k_assignment(matrices, algo="sdv", refine=refine, **options)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("k", (2, 3, 4, 5))
@pytest.mark.parametrize("maximize", (False, True))
def test_independent_single_part_local_optimum(seed, k, maximize):
    n = 4
    matrices = random_instance(k, n=n, seed=seed, dtype=np.int64)
    pair_costs = dict(zip(itertools.combinations(range(k), 2), matrices))
    result = kap.k_assignment(matrices, algo="1dv", maximize=maximize)
    # Read the returned matches only: each row fixes the vertices of all parts
    # except p, and each column offers a vertex of p for that partial clique.
    vertices = np.array([[v for _, v in sorted(clique)] for clique in result.matches])
    assert vertices.shape == (n, k)
    for p in range(k):
        costs = np.zeros((n, n), dtype=np.int64)
        for q in range(k):
            if p < q:
                costs += pair_costs[p, q][:, vertices[:, q]].T
            elif q < p:
                costs += pair_costs[q, p][vertices[:, q], :]
        rows, cols = linear_sum_assignment(costs, maximize=maximize)
        assert costs[np.arange(n), vertices[:, p]].sum() == costs[rows, cols].sum()


@pytest.mark.parametrize("alias,canonical", (
    ("sHiFtAm", "shift"), ("SHIFT", "shift"), ("DV1", "1dv"),
    ("1dV", "1dv"), ("DvS", "sdv"), ("SDV", "sdv"),
))
def test_aliases(alias, canonical):
    matrices = random_instance(4, n=3, seed=91, dtype=np.int64)
    assert kap.k_assignment(matrices, algo=alias) == kap.k_assignment(matrices, algo=canonical)
    assert _core.k_assignment(matrices, alias) == _core.k_assignment(matrices, canonical)
    if canonical != "shift":
        assert kap.k_assignment(matrices, refine=alias) == kap.k_assignment(matrices, refine=canonical)
        assert _core.k_assignment(matrices, "Am", refine=alias) == _core.k_assignment(matrices, "Am", refine=canonical)


@pytest.mark.parametrize("options", (
    {"refine": "unknown"}, {"refine": "shift"}, {"refine": 1},
))
def test_bad_refine(options):
    with pytest.raises(ValueError, match="refine"):
        kap.k_assignment([np.zeros((2, 2), dtype=np.int64)], **options)


@pytest.mark.parametrize("options", SEARCHES)
@pytest.mark.parametrize("missing_core", (False, True))
def test_native_only(monkeypatch, options, missing_core):
    algo, refine = options
    if missing_core:
        monkeypatch.setitem(sys.modules, "kap._core", None)
    with pytest.raises(ValueError, match='backend="kap".*native core'):
        kap.k_assignment([np.zeros((2, 2), dtype=np.int64)], algo=algo, refine=refine,
                         backend="kap" if missing_core else "scipy")


@pytest.mark.parametrize("algo", ("1dv", "sdv"))
@pytest.mark.parametrize("max_iter", (0, -2))
def test_zero_sweeps(algo, max_iter):
    matrices = random_instance(4, n=3, seed=38, dtype=np.int64)
    assert kap.k_assignment(matrices, algo=algo, max_iter=max_iter) == kap.k_assignment(matrices, algo="Am")
    assert kap.k_assignment(matrices, algo="Am", refine=algo, max_iter=max_iter) == kap.k_assignment(matrices, algo="Am")


@pytest.mark.parametrize("options", ({"algo": "sdv"}, {"algo": "Am", "refine": "sdv"}))
def test_sdv_cap(options):
    matrices = [np.zeros((0, 0), dtype=np.int64)] * (17 * 16 // 2)
    with pytest.raises(ValueError, match="k <= 16"):
        kap.k_assignment(matrices, **options)


def test_core_bad_options():
    matrix = [np.zeros((2, 2), dtype=np.int64)]
    with pytest.raises(ValueError, match="refine"):
        _core.k_assignment(matrix, "Am", refine="shift")
    with pytest.raises(ValueError, match="Shift.*1DV.*sDV"):
        _core.k_assignment(matrix, "unknown")


@pytest.mark.parametrize("algo,refine,limit,expected", (
    ("Am", None, -2, 100), ("shift", None, -2, 100),
    ("Cm", None, -2, 0), ("1dv", None, -2, 0), ("sdv", None, 7, 7),
    ("Am", "1dv", -2, 0), ("Bm", "sdv", 7, 7),
    ("Em", "1dv", 7, 7), ("Fm", "sdv", 7, 7),
))
def test_iteration_forwarding(monkeypatch, algo, refine, limit, expected):
    seen = []

    def recording(*args, max_iter, **kwargs):
        seen.append(max_iter)
        return [], [], []

    monkeypatch.setattr(_core, "k_assignment", recording)
    kap.k_assignment([np.zeros((2, 2))], algo=algo, refine=refine, max_iter=limit)
    assert seen == [expected]


@pytest.mark.parametrize("algo,option", (("Cm", "max_iter"), ("Em", "max_iter"),
                                         ("Em", "n_trials"), ("Fm", "max_iter"),
                                         ("1dv", "max_iter"), ("sdv", "max_iter")))
def test_noninteger_search_options(algo, option):
    with pytest.raises(TypeError):
        kap.k_assignment([np.zeros((2, 2), dtype=np.int64)], algo=algo, **{option: 1.5})
