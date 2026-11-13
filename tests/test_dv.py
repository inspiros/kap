"""Native Shift and dimensionwise variation, with an independent LAP oracle."""

import itertools
import sys
import time

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

import kap
from kap import _core
from conftest import assert_valid_assignment, random_instance

BASE = ("Am", "Bm", "Cm", "Dm", "Em", "Fm")
SEARCHES = [(algo, None) for algo in ("shift", "1dv", "sdv")]
SEARCHES += list(itertools.product(BASE, ("1dv", "sdv")))
SEARCHES += [("ils", None), ("ils", "sdv")]
SEARCHES += [("memetic", None), ("memetic", "sdv")]


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
    base_algo = ("sdv" if refine else "1dv") if algo == "ils" else (algo if refine else "Am")
    if algo == "memetic":
        base = kap.k_assignment(matrices, algo="shift", refine=refine or "1dv", **options)
    else:
        base = kap.k_assignment(matrices, algo=base_algo, **options)
    cost, base_cost = sum(actual.matching_costs), sum(base.matching_costs)
    assert cost >= base_cost if maximize else cost <= base_cost
    if k == 3 and algo == "1dv":
        assert actual == kap.k_assignment(matrices, algo="sdv", refine=refine, **options)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("k", (2, 3, 4, 5))
@pytest.mark.parametrize("maximize", (False, True))
@pytest.mark.parametrize("algo", ("1dv", "ils", "memetic"))
def test_independent_single_part_local_optimum(seed, k, maximize, algo):
    n = 4
    matrices = random_instance(k, n=n, seed=seed, dtype=np.int64)
    pair_costs = dict(zip(itertools.combinations(range(k), 2), matrices))
    result = kap.k_assignment(matrices, algo=algo, maximize=maximize, random_state=42)
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
    ("iLs", "ils"), ("MeMeTiC", "memetic"),
))
def test_aliases(alias, canonical):
    matrices = random_instance(4, n=3, seed=91, dtype=np.int64)
    assert kap.k_assignment(matrices, algo=alias, random_state=42) == kap.k_assignment(matrices, algo=canonical, random_state=42)
    assert _core.k_assignment(matrices, alias) == _core.k_assignment(matrices, canonical)
    if canonical in ("1dv", "sdv"):
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
    ("ils", None, -2, 0), ("ils", "sdv", 7, 7),
    ("memetic", None, -2, 0), ("memetic", "sdv", 7, 7),
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
                                         ("1dv", "max_iter"), ("sdv", "max_iter"),
                                         ("ils", "max_iter"), ("ils", "n_trials"),
                                         ("memetic", "max_iter")))
def test_noninteger_search_options(algo, option):
    with pytest.raises(TypeError):
        kap.k_assignment([np.zeros((2, 2), dtype=np.int64)], algo=algo, **{option: 1.5})


@pytest.mark.parametrize("sizes", ([4, 4, 4], [2, 2, 3], [1, 4, 2, 3], [27, 28, 26]))
@pytest.mark.parametrize("maximize", (False, True))
@pytest.mark.parametrize("refine", (None, "sdv"))
def test_ils_valid_reproducible_and_chain_budget(sizes, maximize, refine):
    matrices = random_instance(len(sizes), sizes=sizes, seed=196, dtype=np.int64)
    options = dict(algo="ils", maximize=maximize, return_free=True, random_state=42,
                   refine=refine, max_iter=20)
    one = kap.k_assignment(matrices, n_trials=1, **options)
    assert_valid_assignment(matrices, sizes, one)
    assert one == kap.k_assignment(matrices, n_trials=1, **options)
    four = kap.k_assignment(matrices, n_trials=4, **options)
    assert_valid_assignment(matrices, sizes, four)
    dv = kap.k_assignment(matrices, algo="sdv" if refine else "1dv",
                          maximize=maximize, max_iter=10_000, return_free=True)
    costs = [sum(result.matching_costs) for result in (dv, one, four)]
    assert costs == sorted(costs, reverse=not maximize)
    start = kap.k_assignment(matrices, **{**options, "max_iter": 0})
    assert start == dv
    assert kap.k_assignment(matrices, **{**options, "max_iter": -5}) == start


@pytest.mark.parametrize("trials", (0, -1))
def test_ils_bad_trials(trials):
    with pytest.raises(ValueError, match="n_trials"):
        kap.k_assignment([np.zeros((2, 2), dtype=np.int64)], algo="ils", n_trials=trials)


@pytest.mark.parametrize("seconds", (0, -1, float("nan"), float("inf"), -float("inf"), 1e30))
def test_ils_bad_time_limit(seconds):
    matrices = [np.zeros((2, 2), dtype=np.int64)]
    for solve in (kap.k_assignment, _core.k_assignment):
        with pytest.raises(ValueError, match="time_limit"):
            solve(matrices, algo="ils", time_limit=seconds)


def test_ils_far_future_deadline():
    # Fits Duration but exceeds Instant's range on common platforms.
    matrices = [np.zeros((2, 2), dtype=np.int64)]
    # Explicit iteration limits apply alongside a time limit.
    assert kap.k_assignment(matrices, algo="ils", max_iter=0, time_limit=1e19, random_state=0) == \
        kap.k_assignment(matrices, algo="ils", max_iter=0, random_state=0)


@pytest.mark.parametrize("algo", ("ils", "memetic"))
def test_time_limit_is_the_budget(algo):
    # The default max_iter must not end a time-limited run early.
    matrices = random_instance(3, sizes=[30, 30, 30], seed=7, dtype=np.int64)
    before = time.monotonic()
    actual = kap.k_assignment(matrices, algo=algo, random_state=0, time_limit=0.3,
                              return_free=True)
    elapsed = time.monotonic() - before
    assert 0.25 < elapsed < 5
    assert_valid_assignment(matrices, [30, 30, 30], actual)


def test_time_limit_requires_ils_and_native_core(monkeypatch):
    matrices = [np.zeros((2, 2), dtype=np.int64)]
    for solve in (kap.k_assignment, _core.k_assignment):
        with pytest.raises(ValueError, match="time_limit"):
            solve(matrices, algo="Cm", time_limit=1)
    with pytest.raises(ValueError, match='backend="kap".*native core'):
        kap.k_assignment(matrices, algo="ils", backend="scipy", time_limit=1)
    monkeypatch.setitem(sys.modules, "kap._core", None)
    with pytest.raises(ValueError, match='backend="kap".*native core'):
        kap.k_assignment(matrices, algo="ils", time_limit=1)


@pytest.mark.parametrize("maximize", (False, True))
def test_ils_tiny_deadline(maximize):
    matrices = random_instance(4, sizes=[1, 4, 2, 3], seed=25, dtype=np.int64)
    options = dict(algo="ils", maximize=maximize, random_state=42, n_trials=4)
    start = kap.k_assignment(matrices, max_iter=0, **options)
    before = time.monotonic()
    actual = kap.k_assignment(matrices, max_iter=10**12, time_limit=1e-9, **options)
    assert time.monotonic() - before < 5
    assert actual == start


def test_ils_sdv_cap_and_core_zero_trials():
    matrices = [np.zeros((0, 0), dtype=np.int64)] * (17 * 16 // 2)
    for solve in (kap.k_assignment, _core.k_assignment):
        with pytest.raises(ValueError, match="k <= 16"):
            solve(matrices, algo="ils", refine="sdv", max_iter=0)
    with pytest.raises(ValueError, match="n_trials"):
        _core.k_assignment([np.zeros((0, 0), dtype=np.int64)], "ils", n_trials=0)


@pytest.mark.parametrize("sizes", ([4, 4, 4], [2, 2, 3], [1, 4, 2, 3]))
@pytest.mark.parametrize("maximize", (False, True))
@pytest.mark.parametrize("refine", (None, "sdv"))
def test_memetic_valid_reproducible_and_start(sizes, maximize, refine):
    matrices = random_instance(len(sizes), sizes=sizes, seed=196, dtype=np.int64)
    options = dict(algo="memetic", maximize=maximize, return_free=True, random_state=42,
                   refine=refine, max_iter=5, population=4)
    actual = kap.k_assignment(matrices, **options)
    assert_valid_assignment(matrices, sizes, actual)
    assert actual == kap.k_assignment(matrices, **options)
    assert actual == kap.k_assignment(matrices, n_trials=0, **options)
    start = kap.k_assignment(matrices, algo="shift", refine=refine or "1dv",
                            maximize=maximize, max_iter=10_000)
    cost, start_cost = sum(actual.matching_costs), sum(start.matching_costs)
    assert cost >= start_cost if maximize else cost <= start_cost
    zero = kap.k_assignment(matrices, **{**options, "max_iter": 0})
    assert zero == kap.k_assignment(matrices, **{**options, "max_iter": -5})
    assert_valid_assignment(matrices, sizes, zero)
    zero_cost = sum(zero.matching_costs)
    assert zero_cost >= start_cost if maximize else zero_cost <= start_cost


@pytest.mark.parametrize("options", (
    {"algo": "memetic", "population": 1},
    {"algo": "memetic", "population": 0},
    {"algo": "ils", "population": 4},
    {"algo": "memetic", "population": 2.5},
    {"algo": "memetic", "population": "4"},
    {"algo": "memetic", "population": True},
))
def test_memetic_population_errors(options):
    with pytest.raises(ValueError, match="population"):
        kap.k_assignment([np.zeros((2, 2), dtype=np.int64)], **options)


@pytest.mark.parametrize("population", (None, 4))
@pytest.mark.parametrize("maximize", (False, True))
def test_memetic_time_limit(population, maximize):
    sizes = [6, 8, 5, 7]
    matrices = random_instance(len(sizes), sizes=sizes, seed=196, dtype=np.int64)
    before = time.monotonic()
    actual = kap.k_assignment(matrices, algo="memetic", population=population,
                              maximize=maximize, return_free=True, random_state=42,
                              time_limit=0.05, max_iter=10**9)
    assert time.monotonic() - before < 3
    assert_valid_assignment(matrices, sizes, actual)


@pytest.mark.parametrize("n", (0, 1, 2))
@pytest.mark.parametrize("population", (2, 3))
def test_memetic_collapsed_population(n, population):
    matrices = [np.zeros((n, n), dtype=np.int64)] * 3
    actual = kap.k_assignment(matrices, algo="memetic", population=population,
                              random_state=42, max_iter=5, return_free=True)
    assert_valid_assignment(matrices, [n] * 3, actual)


def test_memetic_sdv_cap():
    refine = "sdv"
    matrices = [np.zeros((0, 0), dtype=np.int64)] * (17 * 16 // 2)
    for solve in (kap.k_assignment, _core.k_assignment):
        with pytest.raises(ValueError, match="k <= 16"):
            solve(matrices, algo="memetic", refine=refine, max_iter=0)


def test_default_algo_is_ils_natively_and_cm_otherwise(monkeypatch):
    matrices = random_instance(4, sizes=[6, 6, 6, 6], seed=3, dtype=np.int64)
    assert kap.k_assignment(matrices, random_state=1) == kap.k_assignment(
        matrices, algo="ils", random_state=1)
    assert kap.k_assignment(matrices, backend="scipy") == kap.k_assignment(
        matrices, algo="Cm", backend="scipy")
    monkeypatch.setitem(sys.modules, "kap._core", None)
    assert kap.k_assignment(matrices) == kap.k_assignment(matrices, algo="Cm")
