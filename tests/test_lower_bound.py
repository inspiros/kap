"""Public and native bounds, checked against exact and feasible solutions."""

import itertools
import sys

import numpy as np
import pytest

import kap
from kap import _core


def instance(k, n, seed, dtype):
    rng = np.random.default_rng(seed)
    return [rng.integers(-20, 21, (n, n)) if dtype is int
            else rng.normal(size=(n, n)) for _ in itertools.combinations(range(k), 2)]


def tolerance(cost, dtype):
    return 1e-9 * max(1, abs(cost)) if dtype is float else 0


SIZES = [(3, n) for n in range(3, 9)] + [(4, n) for n in range(3, 7)] + [(5, n) for n in range(3, 5)]


@pytest.mark.parametrize('k,n', SIZES)
@pytest.mark.parametrize('seed', [0, 1, 7])
@pytest.mark.parametrize('dtype', [int, float])
@pytest.mark.parametrize('maximize', [False, True])
def test_exact_optimum(k, n, seed, dtype, maximize):
    pytest.importorskip("scipy")
    matrices = instance(k, n, seed, dtype)
    optimum = sum(kap.exact(matrices, maximize=maximize).matching_costs)
    bound = kap.lower_bound(matrices, maximize=maximize)
    tol = tolerance(optimum, dtype)
    assert bound >= optimum - tol if maximize else bound <= optimum + tol


@pytest.mark.parametrize('k,n', [(3, 30), (4, 15), (6, 8)])
@pytest.mark.parametrize('seed', [0, 1, 7])
@pytest.mark.parametrize('dtype', [int, float])
@pytest.mark.parametrize('maximize', [False, True])
def test_heuristic_cost(k, n, seed, dtype, maximize):
    matrices = instance(k, n, seed, dtype)
    cost = sum(kap.ils(matrices, random_state=seed, maximize=maximize).matching_costs)
    bound = kap.lower_bound(matrices, maximize=maximize)
    tol = tolerance(cost, dtype)
    assert bound >= cost - tol if maximize else bound <= cost + tol


@pytest.mark.parametrize('n', [0, 1, 8])
@pytest.mark.parametrize('dtype', [int, float])
@pytest.mark.parametrize('maximize', [False, True])
def test_two_parts(n, dtype, maximize):
    matrices = instance(2, n, 42, dtype)
    rows, cols = kap.linear_sum_assignment(matrices[0], maximize=maximize)
    optimum = matrices[0][rows, cols].sum().item()
    bound = kap.lower_bound(matrices, maximize=maximize)
    if dtype is int:
        assert bound == optimum
    else:
        assert bound == pytest.approx(optimum, rel=1e-9, abs=1e-12)
    assert type(bound) is dtype


@pytest.mark.parametrize('k,n', [(3, 8), (4, 6), (5, 4), (6, 8)])
@pytest.mark.parametrize('seed', [0, 1, 7])
@pytest.mark.parametrize('dtype', [int, float])
def test_pairwise_floor_and_monotonicity(k, n, seed, dtype):
    matrices = instance(k, n, seed, dtype)
    pairwise = sum(m[kap.linear_sum_assignment(m)].sum().item() for m in matrices)
    bounds = [kap.lower_bound(matrices, max_iter=iterations) for iterations in (0, 10, 300)]
    assert all(type(bound) is dtype for bound in bounds)
    assert all(bound >= pairwise - tolerance(pairwise, dtype) for bound in bounds)
    assert bounds == sorted(bounds)


@pytest.mark.parametrize('k', [2, 3, 4, 6])
@pytest.mark.parametrize('dtype', [int, float])
def test_planted_zero(k, dtype):
    rng = np.random.default_rng(k)
    vertices = [rng.permutation(8) for _ in range(k)]
    matrices = []
    for p, q in itertools.combinations(range(k), 2):
        matrix = rng.integers(1, 100, size=(8, 8)).astype(dtype)
        matrix[vertices[p], vertices[q]] = 0
        matrices.append(matrix)
    assert kap.lower_bound(matrices) == 0


@pytest.mark.parametrize('k', [3, 4, 6])
@pytest.mark.parametrize('dtype', [int, float])
def test_empty_parts(k, dtype):
    assert kap.lower_bound([np.zeros((0, 0), dtype=dtype)] * (k * (k - 1) // 2)) == 0


def test_large_integer_exactness():
    value = 2**53 + 1
    matrices = [np.full((2, 2), value, dtype=np.int64)] * 3
    assert kap.lower_bound(matrices) == 6 * value
    assert kap.lower_bound(matrices, maximize=True) == 6 * value


@pytest.mark.parametrize('dtype', [np.int64, np.float64])
@pytest.mark.parametrize('layout', ['c', 'f', 'transpose', 'slice', 'reverse'])
def test_native_layouts(dtype, layout):
    matrices = [m.astype(dtype) for m in instance(4, 5, 13, int)]
    if layout == 'f':
        matrices = [np.asfortranarray(m) for m in matrices]
    elif layout == 'transpose':
        matrices = [m.T for m in matrices]
    elif layout == 'slice':
        padded = [np.repeat(np.repeat(m, 2, axis=0), 2, axis=1) for m in matrices]
        matrices = [m[::2, ::2] for m in padded]
    elif layout == 'reverse':
        matrices = [m[::-1, ::-1] for m in matrices]
    expected = _core.lower_bound([np.array(m, order='C') for m in matrices])
    assert _core.lower_bound(matrices) == expected
    assert kap.lower_bound(matrices) == expected


@pytest.mark.parametrize('kind', ['int32', 'bool', 'float16', 'uint64', 'mixed', 'byteswap', 'lists', 'stack'])
def test_promotion(monkeypatch, kind):
    matrices = [np.arange(9).reshape(3, 3)] * 3
    if kind == 'mixed':
        matrices = [matrices[0], matrices[1].astype(np.float32), matrices[2]]
    elif kind == 'byteswap':
        matrices = [m.astype(m.dtype.newbyteorder('S')) for m in matrices]
    elif kind == 'lists':
        matrices = [m.tolist() for m in matrices]
    elif kind == 'stack':
        matrices = np.stack(matrices)
    else:
        matrices = [m.astype(kind) for m in matrices]
    native = _core.lower_bound
    calls = []

    def recording(costs, **kwargs):
        calls.append(costs)
        return native(costs, **kwargs)

    monkeypatch.setattr(_core, 'lower_bound', recording)
    bound = kap.lower_bound(matrices, max_iter=np.int64(10))
    dtype = np.float64 if kind in ('mixed', 'float16') else np.int64
    assert all(m.dtype == dtype and m.dtype.isnative for m in calls[0])
    assert type(bound) is (float if dtype is np.float64 else int)


def test_uint64_overflow():
    big = np.full((2, 2), 2**63, dtype=np.uint64)
    with pytest.raises(OverflowError, match='uint64.*int64'):
        kap.lower_bound([big])
    # Mixed integer/float inputs follow _native's float promotion.
    assert type(kap.lower_bound([big, np.zeros((2, 2)), big])) is float


@pytest.mark.parametrize('matrices', [[], [np.zeros((2, 2))] * 2, [np.zeros(3)],
                                     [np.zeros((2, 2)), np.zeros((2, 3)), np.zeros((4, 3))]])
def test_bad_shapes(matrices):
    with pytest.raises(ValueError):
        kap.lower_bound(matrices)


@pytest.mark.parametrize('matrices', [[np.zeros((2, 3))],
                                     [np.zeros((2, 3)), np.zeros((2, 4)), np.zeros((3, 4))]])
@pytest.mark.parametrize('call', [kap.lower_bound, _core.lower_bound])
def test_unequal_parts(matrices, call):
    with pytest.raises(ValueError, match='equal part sizes'):
        call(matrices)


@pytest.mark.parametrize('value', [-1, np.int64(-1)])
def test_negative_max_iter(value):
    with pytest.raises(ValueError, match='max_iter'):
        kap.lower_bound([np.zeros((2, 2))], max_iter=value)


@pytest.mark.parametrize('value', [True, np.bool_(False), 1.5, '10', None, 1j])
def test_noninteger_max_iter(value):
    with pytest.raises(TypeError, match='max_iter'):
        kap.lower_bound([np.zeros((2, 2))], max_iter=value)


@pytest.mark.parametrize('dtype', [complex, object, 'U4', 'datetime64[D]'])
def test_unsupported_dtype(dtype):
    with pytest.raises(TypeError, match='unsupported cost matrix dtype'):
        kap.lower_bound([np.zeros((2, 2)).astype(dtype)])


@pytest.mark.parametrize('value', [np.nan, -np.inf])
@pytest.mark.parametrize('call', [kap.lower_bound, _core.lower_bound])
def test_invalid_costs_are_value_errors(value, call):
    with pytest.raises(ValueError):
        call([np.full((2, 2), value)])


def test_native_contract():
    matrices = instance(3, 4, 2, int)
    assert type(_core.lower_bound(matrices)) is int
    assert type(_core.lower_bound([m.astype(float) for m in matrices])) is float
    assert _core.lower_bound(matrices) == kap.lower_bound(matrices, max_iter=300)
    with pytest.raises(TypeError):
        _core.lower_bound(matrices, upper=0)
    with pytest.raises(TypeError, match='float64 or all int64'):
        _core.lower_bound([matrices[0].astype(np.int32)])
    with pytest.raises(ValueError):
        _core.lower_bound([])


def test_missing_core(monkeypatch):
    monkeypatch.setitem(sys.modules, 'kap._core', None)
    with pytest.raises(RuntimeError, match='native core.*required'):
        kap.lower_bound([np.zeros((2, 2))])
