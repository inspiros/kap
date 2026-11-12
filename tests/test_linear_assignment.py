"""Compare each available LAP wrapper with SciPy's exact solver."""

import numpy as np
import pytest

import kap

from conftest import DTYPES


def assert_lap_result(cost, result, scipy_assignment, maximize):
    rows, columns = scipy_assignment(cost, maximize=maximize)
    assert np.isclose(sum(result.matching_costs), cost[rows, columns].sum())
    matches = np.asarray(result.matches)
    assert matches.shape == (min(cost.shape), 2)
    assert np.issubdtype(matches.dtype, np.integer)
    assert len(set(matches[:, 0])) == len(matches)
    assert len(set(matches[:, 1])) == len(matches)
    assert np.all((0 <= matches[:, 0]) & (matches[:, 0] < cost.shape[0]))
    assert np.all((0 <= matches[:, 1]) & (matches[:, 1] < cost.shape[1]))
    np.testing.assert_array_equal(
        result.matching_costs, cost[matches[:, 0], matches[:, 1]]
    )
    expected_free = {
        (0, i) for i in set(range(cost.shape[0])) - set(matches[:, 0])
    } | {(1, j) for j in set(range(cost.shape[1])) - set(matches[:, 1])}
    assert set(result.free) == expected_free
    assert len(result.free) == len(expected_free)


@pytest.mark.parametrize("n", (1, 5, 8))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("maximize", (False, True), ids=("min", "max"))
def test_square_optimum(backend, scipy_assignment, instance_factory, n, dtype, maximize):
    cost = instance_factory(2, n=n, dtype=dtype, seed=210 + n)[0]
    result = kap.linear_assignment(cost, backend=backend, maximize=maximize, return_free=True)
    assert_lap_result(cost, result, scipy_assignment, maximize)


@pytest.mark.parametrize("shape", ((3, 7), (7, 3)), ids=("wide", "tall"))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("maximize", (False, True), ids=("min", "max"))
def test_rectangular_optimum(
    rectangular_backend, scipy_assignment, instance_factory, shape, dtype, maximize
):
    cost = instance_factory(2, sizes=shape, dtype=dtype, seed=321)[0]
    result = kap.linear_assignment(
        cost, backend=rectangular_backend, maximize=maximize, return_free=True
    )
    assert_lap_result(cost, result, scipy_assignment, maximize)


@pytest.mark.parametrize("shape", ((3, 7), (7, 3)), ids=("wide", "tall"))
@pytest.mark.parametrize("dtype", DTYPES)
def test_lapjv_rejects_rectangular_input(instance_factory, shape, dtype):
    pytest.importorskip("lapjv")
    cost = instance_factory(2, sizes=shape, dtype=dtype, seed=321)[0]
    with pytest.raises(ValueError, match="square"):
        kap.linear_assignment(cost, backend="lapjv")


def test_free_is_none_by_default(backend, instance_factory):
    cost = instance_factory(2, n=3, seed=75)[0]
    assert kap.linear_assignment(cost, backend=backend).free is None


def test_unknown_backend_raises_value_error():
    with pytest.raises(ValueError, match="backend"):
        kap.linear_assignment(np.ones((2, 2)), backend="unknown")
