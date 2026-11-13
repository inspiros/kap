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
        scores.append(sum(m[vertices[p][i], vertices[q][i]]
                          for i in range(n) for m, (p, q) in zip(matrices, pairs)))
    return (max if maximize else min)(scores)


def check_partition(result, k, n):
    assert len(result.matches) == n
    assert all([p for p, _ in match] == list(range(k)) for match in result.matches)
    for p in range(k):
        assert sorted(match[p][1] for match in result.matches) == list(range(n))


@pytest.mark.parametrize('k,n', [(3, 2), (3, 3), (3, 4), (4, 2), (4, 3)])
@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('dtype', [int, float])
@pytest.mark.parametrize('maximize', [False, True])
def test_brute_force(k, n, seed, dtype, maximize):
    matrices = instance(k, n, seed, dtype)
    result = kap.exact(matrices, maximize=maximize, return_free=True)
    assert sum(result.matching_costs) == pytest.approx(brute(matrices, k, n, maximize))
    check_partition(result, k, n)
    assert result.free == []
    if dtype is int:
        assert all(isinstance(cost, int) for cost in result.matching_costs)


@pytest.mark.parametrize('maximize', [False, True])
def test_two_parts_and_structure(maximize):
    matrices = instance(2, 5, 3, int)
    result = kap.exact(matrices, maximize=maximize, return_free=True)
    rows, cols = kap.linear_sum_assignment(matrices[0], maximize=maximize)
    assert sum(result.matching_costs) == matrices[0][rows, cols].sum()
    reference = kap.ils(matrices, return_free=True, random_state=0)
    assert type(result) is type(reference)
    assert [type(result.matches), type(result.matching_costs), type(result.free)] == [type(reference.matches), type(reference.matching_costs), type(reference.free)]
    assert all(isinstance(match, list) and all(isinstance(node, tuple) for node in match) for match in result.matches)
    assert kap.exact(matrices).free is None


@pytest.mark.parametrize('k,n', [(3, 6), (4, 4), (5, 3)])
@pytest.mark.parametrize('maximize', [False, True])
def test_heuristics(k, n, maximize):
    matrices = instance(k, n, 9, int)
    optimum = sum(kap.exact(matrices, maximize=maximize).matching_costs)
    results = [kap.ils(matrices, max_iter=3, random_state=0, maximize=maximize),
               kap.memetic(matrices, generations=2, population=4, random_state=0, maximize=maximize),
               kap.local_search(matrices, max_sweeps=3, maximize=maximize)]
    results += [kap.multiple_hungarian(matrices, method=method, max_iter=3, n_trials=2,
                                       random_state=0, maximize=maximize)
                for method in ('Am', 'Bm', 'Cm', 'Dm', 'Em', 'Fm')]
    for result in results:
        check_partition(result, k, n)
        score = sum(result.matching_costs)
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


def test_original_integer_costs():
    # Individual costs remain exact even when their sum exceeds float precision.
    matrices = [np.full((2, 2), 2**54 + i, dtype=np.int64) for i in range(3)]
    assert kap.exact(matrices).matching_costs == [3 * 2**54 + 3] * 2


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
