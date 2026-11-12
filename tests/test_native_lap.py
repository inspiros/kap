"""Cross-check kap's native LAP solver against scipy.optimize.linear_sum_assignment.

Both implement the same algorithm with the same tie-breaking, so the
assignments themselves (not only the costs) are expected to be identical.
"""

import itertools

import numpy as np
import pytest

import kap

pytest.importorskip("kap._core")
scipy_lsa = pytest.importorskip("scipy.optimize").linear_sum_assignment

SHAPES = [(0, 0), (0, 3), (3, 0), (1, 1), (1, 5), (5, 1), (4, 4), (3, 8), (8, 3), (20, 20), (17, 31)]


def assert_same(cost, maximize=False, expected_cost=None):
    rows, cols = kap.linear_sum_assignment(cost, maximize=maximize)
    ref = np.asarray(expected_cost if expected_cost is not None else cost)
    srows, scols = scipy_lsa(ref, maximize=maximize)
    assert rows.dtype == np.int64 and cols.dtype == np.int64
    np.testing.assert_array_equal(rows, srows)
    np.testing.assert_array_equal(cols, scols)


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("maximize", (False, True), ids=("min", "max"))
@pytest.mark.parametrize("seed", range(5))
def test_random_float(shape, maximize, seed):
    cost = np.random.default_rng(seed).normal(size=shape)
    assert_same(cost, maximize)


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("maximize", (False, True), ids=("min", "max"))
@pytest.mark.parametrize("seed", range(5))
def test_random_int_with_ties(shape, maximize, seed):
    # A narrow range gives many ties, which exercises tie-breaking.
    cost = np.random.default_rng(seed).integers(-3, 4, size=shape)
    assert_same(cost, maximize)


@pytest.mark.parametrize(
    "dtype",
    [np.float64, np.float32, np.float16, np.longdouble, np.int64, np.int32, np.int16,
     np.int8, np.uint64, np.uint32, np.uint16, np.uint8, np.bool_],
    ids=lambda d: np.dtype(d).name,
)
def test_dtypes(dtype):
    raw = np.random.default_rng(1).integers(0, 50, size=(6, 9))
    cost = raw.astype(dtype)
    assert_same(cost, expected_cost=cost.astype(np.float64))
    assert_same(cost, maximize=True, expected_cost=cost.astype(np.float64))


def test_large_uint64_falls_back_to_float():
    cost = np.array([[2**63 + 2, 0], [0, 2**63 + 2]], dtype=np.uint64)
    rows, cols = kap.linear_sum_assignment(cost)
    np.testing.assert_array_equal(cols, [1, 0])


@pytest.mark.parametrize("byteorder", (">", "<"))
def test_byte_order(byteorder):
    cost = np.random.default_rng(2).normal(size=(5, 7)).astype(np.dtype("f8").newbyteorder(byteorder))
    assert_same(cost, expected_cost=cost.astype("=f8"))


@pytest.mark.parametrize(
    "view",
    [
        lambda a: a.T,
        lambda a: np.asfortranarray(a),
        lambda a: a[::2, 1::3],
        lambda a: a[::-1, ::-1],
        lambda a: a[:, ::-2].T,
    ],
    ids=("T", "fortran", "step", "reversed", "neg-step-T"),
)
def test_strided_views(view):
    base = np.random.default_rng(3).normal(size=(12, 15))
    cost = view(base)
    assert_same(cost, expected_cost=np.ascontiguousarray(cost))


def test_list_input():
    cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    rows, cols = kap.linear_sum_assignment(cost)
    np.testing.assert_array_equal(rows, [0, 1, 2])
    np.testing.assert_array_equal(cols, [1, 0, 2])


def test_brute_force_small():
    rng = np.random.default_rng(4)
    for _ in range(200):
        n, m = map(int, rng.integers(1, 6, size=2))
        cost = rng.integers(-10, 10, size=(n, m))
        rows, cols = kap.linear_sum_assignment(cost)
        small, large = sorted((n, m))
        best = min(
            sum(cost[i, p[i]] if n <= m else cost[p[i], i] for i in range(small))
            for p in itertools.permutations(range(large), small)
        )
        assert cost[rows, cols].sum() == best


def test_forbidden_entries():
    inf = np.inf
    cost = np.array([[inf, 1.0, 5.0], [2.0, inf, 1.0], [1.0, 3.0, inf]])
    assert_same(cost)
    assert_same(-cost, maximize=True)


def test_infeasible():
    cost = np.array([[np.inf, 1.0], [np.inf, 2.0]])
    with pytest.raises(ValueError, match="infeasible"):
        kap.linear_sum_assignment(cost)


@pytest.mark.parametrize("bad, maximize", [(np.nan, False), (-np.inf, False), (np.inf, True)])
def test_invalid_entries(bad, maximize):
    cost = np.array([[1.0, bad], [0.0, 1.0]])
    with pytest.raises(ValueError, match="invalid numeric entries"):
        kap.linear_sum_assignment(cost, maximize=maximize)


@pytest.mark.parametrize("cost", [np.ones(3), np.ones((2, 2, 2)), np.float64(1.0)], ids=("1d", "3d", "0d"))
def test_rejects_non_matrix(cost):
    with pytest.raises(ValueError, match="2-D"):
        kap.linear_sum_assignment(cost)


def test_rejects_non_numeric():
    with pytest.raises(TypeError, match="dtype"):
        kap.linear_sum_assignment(np.array([["a", "b"], ["c", "d"]]))


def test_default_backend_is_native():
    cost = np.random.default_rng(5).normal(size=(6, 4))
    default = kap.linear_assignment(cost)
    native = kap.linear_assignment(cost, backend="kap")
    np.testing.assert_array_equal(default.matches, native.matches)
    rows, cols = scipy_lsa(cost)
    np.testing.assert_array_equal(default.matches, np.stack([rows, cols]).T)
