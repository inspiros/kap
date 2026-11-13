import itertools
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")
import scipy.optimize
import kap


def instance(k, n, seed, dtype):
    rng = np.random.default_rng(seed)
    return [rng.integers(-20, 21, (n, n)).astype(dtype) if dtype is int
            else rng.normal(size=(n, n)) for _ in itertools.combinations(range(k), 2)]


def brute(matrices, k, n, maximize):
    pairs = list(itertools.combinations(range(k), 2))
    scores = []
    for permutations in itertools.product(itertools.permutations(range(n)), repeat=k - 1):
        vertices = [range(n), *permutations]
        scores.append(sum(m[vertices[p][i], vertices[q][i]].item()
                          for i in range(n) for m, (p, q) in zip(matrices, pairs)))
    return (max if maximize else min)(scores)


def check_partition(result, k, n):
    assert len(result.cliques) == n
    assert all([p for p, _ in match] == list(range(k)) for match in result.cliques)
    for p in range(k):
        assert sorted(match[p][1] for match in result.cliques) == list(range(n))


@pytest.mark.parametrize('k,n', [(3, 2), (3, 3), (3, 4), (4, 2), (4, 3)])
@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('dtype', [int, float])
@pytest.mark.parametrize('maximize', [False, True])
def test_brute_force(k, n, seed, dtype, maximize):
    matrices = instance(k, n, seed, dtype)
    result = kap.exact(matrices, maximize=maximize, return_free=True)
    optimum = brute(matrices, k, n, maximize)
    if dtype is int:
        assert sum(result.costs) == optimum
    else:
        assert sum(result.costs) == pytest.approx(optimum)
    check_partition(result, k, n)
    assert result.free == []
    if dtype is int:
        assert all(isinstance(cost, int) for cost in result.costs)


@pytest.mark.parametrize('maximize', [False, True])
def test_two_parts_and_structure(maximize):
    matrices = instance(2, 5, 3, int)
    result = kap.exact(matrices, maximize=maximize, return_free=True)
    rows, cols = kap.linear_sum_assignment(matrices[0], maximize=maximize)
    assert sum(result.costs) == matrices[0][rows, cols].sum()
    reference = kap.ils(matrices, return_free=True, random_state=0)
    assert type(result) is type(reference)
    assert [type(result.cliques), type(result.costs), type(result.free)] == [type(reference.cliques), type(reference.costs), type(reference.free)]
    assert all(isinstance(match, list) and all(isinstance(node, tuple) for node in match) for match in result.cliques)
    assert kap.exact(matrices).free is None


@pytest.mark.parametrize('k,n', [(3, 6), (4, 4), (5, 3)])
@pytest.mark.parametrize('maximize', [False, True])
def test_heuristics(k, n, maximize):
    matrices = instance(k, n, 9, int)
    optimum = sum(kap.exact(matrices, maximize=maximize).costs)
    results = [kap.ils(matrices, max_iter=3, random_state=0, maximize=maximize),
               kap.memetic(matrices, generations=2, population=4, random_state=0, maximize=maximize),
               kap.local_search(matrices, max_sweeps=3, maximize=maximize)]
    results += [kap.multiple_hungarian(matrices, method=method, max_iter=3, n_trials=2,
                                       random_state=0, maximize=maximize)
                for method in ('Am', 'Bm', 'Cm', 'Dm', 'Em', 'Fm')]
    for result in results:
        check_partition(result, k, n)
        score = sum(result.costs)
        assert score <= optimum if maximize else score >= optimum


def test_validation():
    with pytest.raises(ValueError, match=r'n\*\*k.*max_variables'):
        kap.exact(instance(3, 3, 0, int), max_variables=26)
    with pytest.raises(ValueError, match='equal part sizes'):
        kap.exact([np.zeros((2, 3))])
    for value in (0, -1, np.inf, np.nan, 'x'):
        with pytest.raises(ValueError, match='time_limit'):
            kap.exact([np.zeros((2, 2))], time_limit=value)
    assert kap.exact([np.zeros((0, 0))], return_free=True).free == []


@pytest.mark.parametrize('maximize', [False, True])
def test_timeout(monkeypatch, maximize):
    def fake(*args, **kwargs):
        assert kwargs['options'] == {'disp': False, 'mip_rel_gap': 0, 'time_limit': 3}
        return SimpleNamespace(status=1, message='time limit', mip_dual_bound=-10, fun=-8)
    monkeypatch.setattr(scipy.optimize, 'milp', fake)
    with pytest.raises(kap.ExactTimeoutError) as error:
        kap.exact([np.zeros((2, 2))], maximize=maximize, time_limit=3)
    assert isinstance(error.value, TimeoutError)
    assert error.value.bound == (10 if maximize else -10)
    assert error.value.best_cost == (8 if maximize else -8)


def test_invalid_solver_partition(monkeypatch):
    monkeypatch.setattr(scipy.optimize, 'milp', lambda *a, **kw: SimpleNamespace(status=0, x=np.ones(4)))
    with pytest.raises(RuntimeError, match='perfect partition'):
        kap.exact([np.zeros((2, 2))])


@pytest.mark.parametrize('maximize', [False, True])
@pytest.mark.parametrize('sign', [-1, 1])
def test_original_integer_costs(maximize, sign):
    # Distinct objectives one unit apart, immediately below the float64 guard.
    base = 2**52 - 2
    matrices = [sign * np.array([[base, base + 1], [base, base]], dtype=np.int64)]
    optimum = brute(matrices, 2, 2, maximize)
    assert sum(kap.exact(matrices, maximize=maximize).costs) == optimum


@pytest.mark.parametrize('value', [2**53, -(2**53), np.iinfo(np.int64).min])
def test_integer_precision_guard(value):
    matrices = [np.array([[value, value + 1], [value, value]], dtype=np.int64)]
    with pytest.raises(OverflowError, match='integer costs.*too large.*float64.*rescale'):
        kap.exact(matrices)


def test_integer_precision_guard_sums_with_python_ints():
    # Per-matrix maxima fit int64, but their sum does not.
    with pytest.raises(OverflowError, match='integer costs.*float64'):
        kap.exact([np.full((1, 1), 2**62, dtype=np.int64)] * 6)
    assert kap.exact([np.array([[2**53]], dtype=np.int64)]).costs == [2**53]


@pytest.mark.parametrize('scale', [1, 1e8, 1e-200, 1e200])
@pytest.mark.parametrize('maximize', [False, True])
def test_float_objective_scale(scale, maximize):
    matrices = scale * 1e-8 * np.array([[[0, 2], [1, 1]], [[1, 2], [0, 2]], [[0, 0], [1, 2]]])
    optimum = brute(matrices, 3, 2, maximize)
    cost = sum(kap.exact(matrices, maximize=maximize).costs)
    assert cost == pytest.approx(optimum, rel=1e-9, abs=0)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('maximize', [False, True])
def test_nonfinite_costs(monkeypatch, value, maximize):
    def unexpected(*args, **kwargs):
        pytest.fail('non-finite costs must be rejected before calling milp')
    monkeypatch.setattr(scipy.optimize, 'milp', unexpected)
    with pytest.raises(ValueError, match='finite costs.*forbidden'):
        kap.exact([np.array([[0, value], [0, 0]])], maximize=maximize)


@pytest.mark.parametrize('scale', [1e-200, 1, 1e200])
@pytest.mark.parametrize('maximize', [False, True])
def test_timeout_original_scale(monkeypatch, scale, maximize):
    matrix = scale * np.array([[1., 2.], [3., 4.]])

    def fake(objective, **kwargs):
        assert 512 <= np.max(np.abs(objective)) < 1024
        # Two valid permutation objectives expressed in the solver's scale.
        return SimpleNamespace(status=1, message='time limit',
                               mip_dual_bound=objective[0] + objective[3],
                               fun=objective[1] + objective[2])
    monkeypatch.setattr(scipy.optimize, 'milp', fake)
    with pytest.raises(kap.ExactTimeoutError) as error:
        kap.exact([matrix], maximize=maximize, time_limit=3)
    assert error.value.bound == pytest.approx(5 * scale, rel=1e-15, abs=0)
    assert error.value.best_cost == pytest.approx(5 * scale, rel=1e-15, abs=0)


def test_integer_objective_is_not_scaled(monkeypatch):
    matrix = np.array([[2, 3], [5, 7]], dtype=np.int64)

    def fake(objective, **kwargs):
        np.testing.assert_array_equal(objective, matrix.ravel())
        return SimpleNamespace(status=1, message='time limit', mip_dual_bound=8, fun=9)
    monkeypatch.setattr(scipy.optimize, 'milp', fake)
    with pytest.raises(kap.ExactTimeoutError) as error:
        kap.exact([matrix])
    assert error.value.bound == 8
    assert error.value.best_cost == 9


def test_missing_scipy_is_lazy(monkeypatch):
    import builtins
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith('scipy'):
            raise ImportError('missing scipy')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', blocked)
    with pytest.raises(ImportError, match='pip install scipy'):
        kap.exact([np.zeros((2, 2))])


def test_other_solver_status(monkeypatch):
    monkeypatch.setattr(scipy.optimize, 'milp', lambda *a, **kw: SimpleNamespace(status=2, message='infeasible'))
    with pytest.raises(RuntimeError, match='infeasible'):
        kap.exact([np.zeros((2, 2))])
