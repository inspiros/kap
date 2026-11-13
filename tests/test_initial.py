"""Warm starts from complete solutions and rejection of incomplete partitions."""

from itertools import combinations

import numpy as np
import pytest

import kap
from kap import _core

from conftest import assert_valid_assignment, clique_cost, random_instance


def canonical(result):
    return sorted((tuple(sorted(row)), cost)
                  for row, cost in zip(result.matches, result.matching_costs))


@pytest.mark.parametrize("k", (3, 4, 5))
@pytest.mark.parametrize("unequal", (False, True))
@pytest.mark.parametrize("dtype", (np.int64, np.float64))
@pytest.mark.parametrize("maximize", (False, True))
@pytest.mark.parametrize("neighborhood", ("1dv", "sdv"))
def test_warm_start_quality_and_local_optimum(k, unequal, dtype, maximize, neighborhood):
    sizes = [3 + (p % 3 if unequal else 0) for p in range(k)]
    matrices = random_instance(k, sizes=sizes, seed=1729 + k, dtype=dtype)
    initial = kap.multiple_hungarian(matrices, "Am", maximize=maximize, return_free=True)
    options = dict(maximize=maximize, return_free=True, neighborhood=neighborhood)
    local = kap.local_search(matrices, initial=initial, max_sweeps=10_000, **options)
    again = kap.local_search(matrices, initial=local, max_sweeps=10_000, **options)
    assert [row for row, _ in canonical(again)] == [row for row, _ in canonical(local)]
    np.testing.assert_allclose([cost for _, cost in canonical(again)],
                               [cost for _, cost in canonical(local)], rtol=1e-14, atol=1e-12)
    assert sorted(again.free) == sorted(local.free)
    zero = kap.ils(matrices, initial=initial, max_iter=0, n_chains=3, random_state=42, **options)
    assert zero == local
    searched = kap.ils(matrices, initial=initial, max_iter=60, n_chains=3, random_state=42, **options)
    assert searched == kap.ils(matrices, initial=initial, max_iter=60, n_chains=3,
                               random_state=42, **options)
    for result in (local, searched):
        assert_valid_assignment(matrices, sizes, result)
        assert len(result.matches) + len(result.free) == max(sizes)
        start_cost = clique_cost(matrices, k, initial.matches)
        cost = sum(result.matching_costs)
        assert cost >= start_cost - 1e-10 if maximize else cost <= start_cost + 1e-10
    cost, local_cost = sum(searched.matching_costs), sum(local.matching_costs)
    assert cost >= local_cost - 1e-10 if maximize else cost <= local_cost + 1e-10


@pytest.mark.parametrize("solve,options", ((kap.local_search, {}), (kap.ils, {"max_iter": 5, "random_state": 42})))
@pytest.mark.parametrize("sizes", ([4, 4, 4], [1, 4, 2, 3]))
def test_result_matches_and_iterables_agree(solve, options, sizes):
    matrices = random_instance(len(sizes), sizes=sizes, seed=6, dtype=np.int64)
    initial = kap.local_search(matrices, return_free=True)
    expected = solve(matrices, initial=initial.matches, return_free=True, **options)
    assert solve(matrices, initial=initial, return_free=True, **options) == expected
    # Both outer and inner generators are accepted; numpy integer indices too.
    rows = ((tuple(np.int64(v) for v in node) for node in row) for row in initial.matches)
    assert solve(matrices, initial=rows, return_free=True, **options) == expected


@pytest.mark.parametrize("sizes", ([0, 0, 0], [0, 3, 0], [2, 2, 2]))
@pytest.mark.parametrize("neighborhood", ("1dv", "sdv"))
def test_complete_zero_costs_and_empty_parts(sizes, neighborhood):
    matrices = [np.zeros((sizes[a], sizes[b]), dtype=np.float64)
                for a, b in combinations(range(3), 2)]
    initial = [[(p, v) for p, n in enumerate(sizes) if v < n] for v in range(max(sizes))]
    initial = [row for row in initial if len(row) > 1]
    options = dict(initial=initial, neighborhood=neighborhood, return_free=True)
    for result in (kap.local_search(matrices, **options), kap.ils(matrices, max_iter=0, **options)):
        assert_valid_assignment(matrices, sizes, result)
        assert len(result.matches) + len(result.free) == max(sizes)


@pytest.mark.parametrize("k", (2, 3, 4, 5))
@pytest.mark.parametrize("explicit_singletons", (False, True))
@pytest.mark.parametrize("solve", (kap.local_search, kap.ils))
def test_incomplete_singleton_starts_raise(k, explicit_singletons, solve):
    matrices = [np.zeros((2, 2), dtype=np.int64)] * (k * (k - 1) // 2)
    initial = [[(p, v)] for p in range(k) for v in range(2)] if explicit_singletons else []
    message = f"initial must be a complete solution: 2 nodes expected, {2 * k} given"
    with pytest.raises(ValueError) as public_error:
        solve(matrices, initial=initial)
    assert str(public_error.value) == message
    with pytest.raises(ValueError) as core_error:
        _core.k_assignment(matrices, "ils", initial=initial)
    assert str(core_error.value) == message


@pytest.mark.parametrize("solve", (kap.local_search, kap.ils))
def test_incomplete_unmergeable_partition_raises(solve):
    matrices = [np.zeros((2, 2), dtype=np.int64)] * 3
    initial = [[(0, 0), (1, 0)], [(0, 1), (2, 1)], [(1, 1), (2, 0)]]
    message = "initial must be a complete solution: 2 nodes expected, 3 given"
    with pytest.raises(ValueError) as public_error:
        solve(matrices, initial=initial)
    assert str(public_error.value) == message
    with pytest.raises(ValueError) as core_error:
        _core.k_assignment(matrices, "Am", refine="1dv", initial=initial)
    assert str(core_error.value) == message


@pytest.mark.parametrize("solve", (kap.local_search, kap.ils))
def test_missing_vertices_can_make_start_incomplete(solve):
    matrices = random_instance(3, sizes=[1, 4, 2], seed=17, dtype=np.int64)
    with pytest.raises(ValueError, match="complete solution: 4 nodes expected, 5 given"):
        solve(matrices, initial=[[(0, 0), (1, 0), (2, 0)]])


@pytest.mark.parametrize("initial,message", (
    (3, "iterable"), ([3], "iterable"), ([[]], "empty clique"),
    ([[0]], "pairs"), ([[(0,)]], "pairs"), ([[(0, 0, 1)]], "pairs"),
    ([[(0.0, 0)]], "integers"), ([[(0, 1.5)]], "integers"),
    ([[(True, 0)]], "integers"), ([[(0, np.bool_(True))]], "integers"),
    ([[("0", 0)]], "integers"), ([[(0, float("nan"))]], "integers"),
    ([[(-1, 0)]], "part index"), ([[(3, 0)]], "part index"),
    ([[(0, -1)]], "vertex index"), ([[(1, 3)]], "vertex index"),
    ([[(0, 0)], [(0, 0)]], "used more than once"),
    ([[(0, 0), (0, 0)]], "used more than once"),
    ([[(0, 0), (0, 1)]], "more than one vertex of part"),
))
@pytest.mark.parametrize("solve", (kap.local_search, kap.ils))
def test_invalid_initial(initial, message, solve):
    matrices = [np.zeros((3, 3), dtype=np.int64)] * 3
    with pytest.raises(ValueError, match=message):
        solve(matrices, initial=initial)


def test_start_and_initial_exclusive_and_none_is_am():
    matrices = random_instance(3, n=3, seed=71, dtype=np.int64)
    assert kap.local_search(matrices) == kap.local_search(matrices, start=None)
    assert kap.local_search(matrices) == kap.local_search(matrices, start="Am")
    for start in ("Am", "Bm", "Dm", "Shift"):
        with pytest.raises(ValueError, match="start and initial"):
            kap.local_search(matrices, start=start, initial=[])


@pytest.mark.parametrize("initial", ([[]], [[(3, 0)]], [[(0, 3)]], [[(0, -1)]],
                                     [[(0, 0)], [(0, 0)]], [[(0, 0), (0, 1)]], [[(0.5, 0)]]))
def test_core_invalid_initial_raises_value_error(initial):
    with pytest.raises(ValueError):
        _core.k_assignment([np.zeros((3, 3), dtype=np.int64)] * 3, "ils", initial=initial)


@pytest.mark.parametrize("algo", ("Cm", "Em", "Fm", "memetic"))
def test_core_rejects_initial_for_unsupported_algorithms(algo):
    with pytest.raises(ValueError, match="initial"):
        _core.k_assignment([np.zeros((3, 3), dtype=np.int64)] * 3, algo, initial=[])


def test_zero_sweeps_preserve_supplied_cliques():
    sizes = [1, 4, 2]
    matrices = random_instance(3, sizes=sizes, seed=17, dtype=np.int64)
    initial = [[(2, 1), (0, 0), (1, 1)], [(1, 0), (2, 0)]]
    result = kap.local_search(matrices, initial=initial, max_sweeps=0, return_free=True)
    assert result.matches == [sorted(row) for row in initial]
    assert_valid_assignment(matrices, sizes, result)
    assert len(result.matches) + len(result.free) == max(sizes)
    assert sorted(result.free) == [(1, 2), (1, 3)]
    assert initial == [[(2, 1), (0, 0), (1, 1)], [(1, 0), (2, 0)]]
    # Free vertices are restored even when the AssignmentResult omits `free`.
    without_free = kap.local_search(matrices, initial=initial, max_sweeps=0)
    for warm in (result, result.matches, without_free):
        assert kap.local_search(matrices, initial=warm, max_sweeps=0, return_free=True) == result


@pytest.mark.parametrize('maximize', [False, True])
@pytest.mark.parametrize('n', [1, 2])
def test_warm_start_integer_overflow(maximize, n):
    # n=1 overflows within a clique; n=2 overflows only in the total.
    value = 2**60 if n == 1 else 2**59
    matrices = [np.full((n, n), value, dtype=np.int64)] * 10
    initial = [[(p, v) for p in range(5)] for v in range(n)]
    with pytest.raises(ValueError, match='cost overflow'):
        kap.local_search(matrices, initial=initial, max_sweeps=0, maximize=maximize)


@pytest.mark.parametrize('maximize,value', [(False, np.nan), (True, np.nan),
                                           (False, -np.inf), (True, np.inf),
                                           (False, 2**61), (True, -(2**63))])
def test_warm_start_validates_all_entries(maximize, value):
    dtype = np.int64 if isinstance(value, int) else np.float64
    matrix = np.zeros((2, 2), dtype=dtype)
    matrix[0, 1] = value  # Not selected by the initial solution.
    initial = [[(0, v), (1, v)] for v in range(2)]
    with pytest.raises(ValueError, match=r'invalid cost entry at \(0, 1\)'):
        kap.local_search([matrix], initial=initial, max_sweeps=0, maximize=maximize)


@pytest.mark.parametrize("neighborhood", ("1dv", "sdv"))
def test_local_search_initial_always_requests_refinement(monkeypatch, neighborhood):
    matrices = random_instance(3, n=3, seed=17, dtype=np.int64)
    initial = kap.multiple_hungarian(matrices, "Am")
    native = _core.k_assignment
    calls = []

    def recording(costs, algo, **kwargs):
        calls.append((algo, kwargs["refine"], kwargs["initial"]))
        return native(costs, algo, **kwargs)

    monkeypatch.setattr(_core, "k_assignment", recording)
    kap.local_search(matrices, initial=initial, neighborhood=neighborhood)
    assert calls == [("a", neighborhood, initial.matches)]
