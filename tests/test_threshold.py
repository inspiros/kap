"""Optional matching checked against an independent transversal-partition oracle."""

from itertools import combinations

import numpy as np
import pytest

import kap
from kap import _assignment, _core, _reference


CASES = [
    pytest.param(kap.ils, dict(max_iter=6, n_chains=2, random_state=31, neighborhood=dv),
                 id=f"ils-{dv}") for dv in ("1dv", "sdv")
] + [
    pytest.param(kap.memetic, dict(generations=6, population=4, random_state=31, neighborhood=dv),
                 id=f"memetic-{dv}") for dv in ("1dv", "sdv")
] + [
    pytest.param(kap.local_search, dict(neighborhood=dv), id=f"local-{dv}")
    for dv in ("1dv", "sdv")
] + [
    pytest.param(kap.multiple_hungarian, dict(method=method, max_iter=6, n_trials=2, random_state=31),
                 id=method) for method in ("Am", "Bm", "Cm", "Dm", "Em", "Fm", "Shift")
] + [
    pytest.param(kap.k_assignment, dict(algo=algo, max_iter=6, random_state=31),
                 id=f"dispatcher-{algo}")
    for algo in (None, "Am", "Bm", "Cm", "Dm", "Em", "Fm", "Shift", "1dv", "sdv", "ils", "memetic")
] + [
    pytest.param(kap.k_assignment, dict(algo=algo, refine="sdv", max_iter=6, random_state=31),
                 id=f"dispatcher-refine-{algo}") for algo in ("Am", "1dv", "ils", "memetic")
]


def edge_table(matrices, sizes):
    return dict(zip(combinations(range(len(sizes)), 2), matrices))


def edge(table, x, y):
    (p, u), (q, v) = sorted((x, y))
    return table[p, q][u, v].item()


def partition_optimum(matrices, sizes, tau, maximize):
    """Enumerate each set partition once, excluding repeated parts and forbidden pairs.

    This uses no kap helpers, LAP solver, or assumption about the number of cliques.
    """
    vertices = [(p, v) for p, n in enumerate(sizes) for v in range(n)]
    table = edge_table(matrices, sizes)
    best = 0
    cliques = []

    def visit(index, score):
        nonlocal best
        if index == len(vertices):
            best = max(best, score) if maximize else min(best, score)
            return
        vertex = vertices[index]
        for row in cliques:
            if any(other[0] == vertex[0] for other in row):
                continue
            costs = [edge(table, vertex, other) for other in row]
            if not all(np.isfinite(cost) for cost in costs):
                continue
            row.append(vertex)
            visit(index + 1, score + sum(cost - tau for cost in costs))
            row.pop()
        cliques.append([vertex])
        visit(index + 1, score)
        cliques.pop()

    visit(0, 0)
    return best


def validate(matrices, sizes, result, tau):
    table = edge_table(matrices, sizes)
    seen = []
    shifted = 0
    assert len(result.matches) == len(result.matching_costs)
    for row, reported in zip(result.matches, result.matching_costs):
        assert len(row) >= 2
        assert len({p for p, _ in row}) == len(row)
        costs = [edge(table, x, y) for x, y in combinations(row, 2)]
        assert all(np.isfinite(cost) for cost in costs), "forbidden pair selected"
        assert reported == pytest.approx(sum(costs), abs=1e-12)
        shifted += sum(cost - tau for cost in costs)
        seen.extend(row)
    seen.extend(result.free)
    assert len(seen) == len(set(seen))
    assert set(seen) == {(p, v) for p, n in enumerate(sizes) for v in range(n)}
    return shifted


def instance(sizes, seed, dtype):
    rng = np.random.default_rng(seed)
    return [rng.integers(-8, 10, size=(sizes[p], sizes[q])).astype(dtype)
            for p, q in combinations(range(len(sizes)), 2)]


@pytest.mark.parametrize("solve,options", CASES)
@pytest.mark.parametrize("maximize", (False, True))
@pytest.mark.parametrize("sizes", ((0, 0), (0, 4), (1, 3), (2, 4), (3, 2), (4, 4),
                                    (0, 2, 3), (1, 1, 1), (1, 3, 2), (2, 2, 3), (3, 3, 3)))
def test_partition_oracle(solve, options, maximize, sizes):
    for seed, dtype, tau in ((1729, np.int64, 2), (71, np.float64, 0.75)):
        matrices = instance(sizes, seed, dtype)
        optimum = partition_optimum(matrices, sizes, tau, maximize)
        result = solve(matrices, threshold=tau, maximize=maximize, return_free=True, **options)
        score = validate(matrices, sizes, result, tau)
        assert score <= optimum + 1e-10 if maximize else score >= optimum - 1e-10
        if len(sizes) == 2:
            assert score == pytest.approx(optimum)


@pytest.mark.parametrize("solve,options", CASES)
@pytest.mark.parametrize("maximize", (False, True))
def test_no_improving_pairs(solve, options, maximize):
    sizes = (2, 3, 1)
    matrices = instance(sizes, 31, np.int64)
    tau = 10 if maximize else -9
    result = solve(matrices, threshold=tau, maximize=maximize, return_free=True, **options)
    assert result.matches == result.matching_costs == []
    validate(matrices, sizes, result, tau)
    assert len(result.free) == sum(sizes)


@pytest.mark.parametrize("solve,options", CASES)
@pytest.mark.parametrize("maximize", (False, True))
@pytest.mark.parametrize("sizes", ((3, 4), (2, 3, 3)))
def test_forbidden_pairs(solve, options, maximize, sizes):
    rng = np.random.default_rng(493)
    for all_forbidden in (False, True):
        matrices = instance(sizes, 71, np.float64)
        for matrix in matrices:
            matrix[np.ones(matrix.shape, dtype=bool) if all_forbidden else
                   rng.random(matrix.shape) < 0.35] = -np.inf if maximize else np.inf
        result = solve(matrices, threshold=1.5, maximize=maximize, return_free=True, **options)
        score = validate(matrices, sizes, result, 1.5)
        optimum = partition_optimum(matrices, sizes, 1.5, maximize)
        assert score <= optimum + 1e-10 if maximize else score >= optimum - 1e-10
        if len(sizes) == 2:
            assert score == pytest.approx(optimum)
        if all_forbidden:
            assert result.matches == result.matching_costs == []
            assert len(result.free) == sum(sizes)


@pytest.mark.parametrize("solve", (kap.local_search, kap.ils))
@pytest.mark.parametrize("maximize", (False, True))
@pytest.mark.parametrize("initial", ([], [[(0, 0), (1, 1)]]))
def test_partial_initial(solve, maximize, initial):
    sizes = (2, 3, 3)
    matrices = instance(sizes, 31, np.int64)
    options = dict(initial=initial, maximize=maximize, return_free=True)
    result = solve(matrices, threshold=2, **options)
    score = validate(matrices, sizes, result, 2)
    table = edge_table(matrices, sizes)
    start = sum(edge(table, x, y) - 2 for row in initial for x, y in combinations(row, 2))
    assert score >= start if maximize else score <= start
    assert solve(matrices, initial=result, threshold=2, maximize=maximize) is not None
    with pytest.raises(ValueError, match="complete solution"):
        solve(matrices, **options)


@pytest.mark.parametrize("solve", (kap.local_search, kap.ils))
@pytest.mark.parametrize("maximize", (False, True))
def test_forbidden_initial(solve, maximize):
    matrix = np.array([[1., -np.inf if maximize else np.inf], [2., 3.]])
    with pytest.raises(ValueError, match="initial places a forbidden pair in one clique"):
        solve([matrix], threshold=2, initial=[[(0, 0), (1, 1)]], maximize=maximize)


@pytest.mark.parametrize("solve,options", CASES)
@pytest.mark.parametrize("maximize", (False, True))
def test_integer_threshold_and_none_compatibility(solve, options, maximize):
    matrices = instance((3, 2, 3), 512, np.int64)
    original = [matrix.copy() for matrix in matrices]
    common = dict(maximize=maximize, return_free=True, **options)
    integral = solve(matrices, threshold=np.int64(2), **common)
    assert integral.matching_costs
    assert all(isinstance(cost, int) for cost in integral.matching_costs)
    assert solve(matrices, threshold=2.0, **common) == integral
    with pytest.raises(ValueError, match="fractional threshold.*float64"):
        solve(matrices, threshold=2.5, **common)
    fractional = solve([m.astype(np.float64) for m in matrices], threshold=2.5, **common)
    assert all(isinstance(cost, float) for cost in fractional.matching_costs)
    assert solve(matrices, threshold=None, **common) == solve(matrices, **common)
    for matrix, before in zip(matrices, original):
        np.testing.assert_array_equal(matrix, before)


@pytest.mark.parametrize("solve,options", CASES)
@pytest.mark.parametrize("tau,error", ((np.nan, ValueError), (np.inf, ValueError), (-np.inf, ValueError),
                                      (True, TypeError), (np.bool_(False), TypeError),
                                      ("2", TypeError), (1j, TypeError), ([], TypeError)))
def test_bad_threshold(solve, options, tau, error):
    with pytest.raises(error, match="threshold"):
        solve([np.zeros((2, 2))], threshold=tau, **options)


@pytest.mark.parametrize("solve,options", CASES)
@pytest.mark.parametrize("maximize", (False, True))
def test_bad_entries(solve, options, maximize):
    for invalid in (np.nan, np.inf if maximize else -np.inf):
        matrix = np.zeros((2, 2))
        matrix[1, 0] = invalid
        with pytest.raises(ValueError, match="invalid cost entry"):
            solve([matrix], threshold=1, maximize=maximize, **options)


@pytest.mark.parametrize("solve", (kap.multiple_hungarian, kap.k_assignment))
@pytest.mark.parametrize("backend", ("scipy", "lap", "lapjv", "munkres", "unknown"))
def test_other_backends_reject_threshold(solve, backend):
    with pytest.raises(ValueError, match="threshold needs the native core"):
        solve([np.zeros((2, 2))], threshold=1, backend=backend)


def test_reference_rejects_threshold():
    with pytest.raises(ValueError, match="threshold needs the native core"):
        _reference.k_assignment([np.zeros((2, 2))], threshold=1)
    matrices = [np.zeros((2, 2))]
    assert _reference.k_assignment(matrices, threshold=None) == _reference.k_assignment(matrices)


@pytest.mark.parametrize("solve,options", CASES)
def test_missing_core_rejects_threshold(monkeypatch, solve, options):
    monkeypatch.setattr(_assignment, "_core", lambda: None)
    with pytest.raises(ValueError, match="threshold needs the native core"):
        solve([np.zeros((2, 2))], threshold=1, **options)


def test_native_error_mapping_and_threshold_range():
    with pytest.raises(ValueError, match="threshold"):
        _core.k_assignment([np.zeros((1, 1), dtype=np.int64)], "Am", threshold=0.5)
    with pytest.raises(ValueError, match="cost overflow"):
        kap.multiple_hungarian([np.array([[np.iinfo(np.int64).max]])], "Am", threshold=-1)
    with pytest.raises(ValueError, match="threshold"):
        kap.ils([np.zeros((1, 1), dtype=np.int64)], threshold=2**63)
    with pytest.raises(ValueError, match="exactly representable"):
        kap.ils([np.zeros((1, 1), dtype=np.int64)], threshold=2**53 + 1)


@pytest.mark.parametrize("maximize", (False, True))
def test_soft_gate_and_original_costs(maximize):
    # A triangle may contain a pair above tau if its other two pairs pay for it.
    sign = -1 if maximize else 1
    matrices = [np.array([[sign * c]], dtype=np.float64) for c in (1, 1, 12)]
    result = kap.ils(matrices, threshold=sign * 10, maximize=maximize, return_free=True)
    assert result.matches == [[(0, 0), (1, 0), (2, 0)]]
    assert result.matching_costs == [sign * 14]
    validate(matrices, (1, 1, 1), result, sign * 10)
    # Recost directly from originals, even if shifting erases their small magnitudes.
    result = kap.ils(matrices, threshold=sign * 1e20, maximize=maximize)
    assert result.matching_costs == [sign * 14]


@pytest.mark.parametrize("solve,options", CASES)
def test_integral_threshold_preserves_large_integer_costs(solve, options):
    value = 2**54 + 1
    matrices = [np.array([[value]], dtype=np.int64)]
    result = solve(matrices, threshold=2**55, **options)
    assert result.matching_costs == [value]
    assert isinstance(result.matching_costs[0], int)


@pytest.mark.parametrize("solve", (kap.local_search, kap.ils))
@pytest.mark.parametrize("initial,message", (([[]], "empty clique"),
                                             ([[(0, 0)], [(0, 0)]], "used more than once"),
                                             ([[(0, 0), (0, 1)]], "more than one vertex"),
                                             ([[(0, True)]], "integers"),
                                             ([[(2, 0)]], "part index"),
                                             ([[(0, 2)]], "vertex index")))
def test_threshold_keeps_initial_structure_checks(solve, initial, message):
    with pytest.raises(ValueError, match=message):
        solve([np.zeros((2, 2))], initial=initial, threshold=1)


def test_exact_and_bound_have_no_threshold():
    import inspect

    assert "threshold" not in inspect.signature(kap.exact).parameters
    assert "threshold" not in inspect.signature(kap.lower_bound).parameters


def test_fractional_real_threshold_rejected_before_rounding():
    from fractions import Fraction

    # The fractional part rounds away in float64, but the original threshold
    # is fractional, so integer input must still be rejected.
    tau = Fraction(1) + Fraction(1, 2**54)
    matrices = [np.zeros((1, 1), dtype=np.int64)]
    with pytest.raises(ValueError, match="fractional threshold.*float64"):
        kap.ils(matrices, threshold=tau)
    result = kap.ils([m.astype(np.float64) for m in matrices], threshold=tau)
    assert result.matching_costs == [0.0]
    assert isinstance(result.matching_costs[0], float)


@pytest.mark.parametrize("dtype", (np.int64, np.uint64, np.bool_))
def test_fractional_threshold_requires_explicit_float_input(dtype):
    matrices = [np.ones((1, 1), dtype=dtype) for _ in range(3)]
    with pytest.raises(ValueError, match="fractional threshold.*float64"):
        kap.ils(matrices, threshold=1.5)
    # A mixed input already selects floating-point arithmetic independently of tau.
    matrices[0] = matrices[0].astype(np.float64)
    result = kap.ils(matrices, threshold=1.5)
    assert result.matching_costs == [3.0]
    assert isinstance(result.matching_costs[0], float)
