"""Native multiple Hungarian heuristics against the Python quotient oracle."""

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import assert_valid_assignment, random_instance
from kap import _core, _reference


def strided(matrices, layout):
    if layout == "fortran":
        return [np.asfortranarray(m) for m in matrices]
    if layout == "negative":
        return [m[::-1, ::-1] for m in matrices]
    if layout == "stepped":
        result = []
        for matrix in matrices:
            backing = np.empty(tuple(2 * d for d in matrix.shape), dtype=matrix.dtype)
            view = backing[::2, ::2]
            view[:] = matrix
            result.append(view)
        return result
    return matrices


@pytest.mark.parametrize("algo", ["Am", "Bm", "Dm"])
@pytest.mark.parametrize("k", range(2, 7))
@pytest.mark.parametrize("equal", [True, False])
@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("seed,layout", [(0, "c"), (4, "fortran"), (17, "negative")])
@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_oracle(algo, k, equal, maximize, seed, layout, dtype):
    matrices = strided(random_instance(k, seed=seed, dtype=dtype, equal=equal), layout)
    actual = _core.k_assignment(matrices, algo, maximize=maximize)
    expected = _reference.k_assignment(
        matrices, algo=algo, maximize=maximize, return_free=True
    )
    if dtype == np.int64:
        assert actual == tuple(expected)
    else:
        np.testing.assert_allclose(sum(actual[1]), sum(expected.costs), rtol=1e-9)
    sizes = [matrices[0].shape[0]] + [matrices[p - 1].shape[1] for p in range(1, k)]
    assert_valid_assignment(
        matrices, sizes,
        SimpleNamespace(cliques=actual[0], costs=actual[1], free=actual[2]),
    )
    assert all(isinstance(part, list) for part in actual)
    assert all(isinstance(clique, list) for clique in actual[0])
    assert all(isinstance(vertex, tuple) for clique in actual[0] for vertex in clique)


@pytest.mark.parametrize("algo", ["Am", "Bm", "Dm"])
@pytest.mark.parametrize("maximize", [False, True])
def test_ties_and_stepped_inputs(algo, maximize):
    for sizes in [(3, 3, 3, 3, 3), (4, 2, 3, 1, 4)]:
        matrices = random_instance(5, sizes=sizes, dtype=np.int64)
        for m in matrices:
            m.fill(0)
        matrices = strided(matrices, "stepped")
        expected = _reference.k_assignment(matrices, algo=algo, maximize=maximize, return_free=True)
        assert _core.k_assignment(matrices, algo, maximize) == tuple(expected)
    matrices = strided(random_instance(4, equal=False, dtype=np.int64), "stepped")
    expected = _reference.k_assignment(matrices, algo=algo, maximize=maximize, return_free=True)
    assert _core.k_assignment(matrices, algo, maximize) == tuple(expected)


@pytest.mark.parametrize("algo", ["Am", "Bm", "Dm"])
@pytest.mark.parametrize("sizes", [(0, 0), (0, 0, 0), (0, 2, 3), (3, 0, 1, 2)])
@pytest.mark.parametrize("maximize", [False, True])
def test_empty_partites(algo, sizes, maximize):
    matrices = [np.zeros((a, b), dtype=np.int64)
                for p, a in enumerate(sizes) for b in sizes[p + 1:]]
    matches, costs, free = _core.k_assignment(matrices, algo, maximize)
    assert_valid_assignment(
        matrices, sizes, SimpleNamespace(cliques=matches, costs=costs, free=free)
    )


@pytest.mark.parametrize("algo", ["Cm", "Em", "Fm"])
@pytest.mark.parametrize("k", range(2, 6))
@pytest.mark.parametrize("equal", [True, False])
@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_local_search(algo, k, equal, maximize, dtype):
    matrices = random_instance(k, seed=3, dtype=dtype, equal=equal, n=4)
    sizes = [matrices[0].shape[0]] + [m.shape[1] for m in matrices[: k - 1]]
    matches, costs, free = _core.k_assignment(matrices, algo, maximize, seed=7)
    assert_valid_assignment(
        matrices, sizes, SimpleNamespace(cliques=matches, costs=costs, free=free)
    )
    bm_cost = sum(_core.k_assignment(matrices, "Bm", maximize)[1])
    assert (sum(costs) >= bm_cost - 1e-9) if maximize else (sum(costs) <= bm_cost + 1e-9)
    assert _core.k_assignment(matrices, algo, maximize, seed=7) == (matches, costs, free)


def test_bad_options():
    with pytest.raises(ValueError, match="max_iter"):
        _core.k_assignment([np.zeros((2, 2))], "Fm", max_iter=0)
    with pytest.raises(ValueError, match="n_trials"):
        _core.k_assignment([np.zeros((2, 2))], "Em", n_trials=0)


def test_input_errors():
    with pytest.raises(ValueError, match="matrix count"):
        _core.k_assignment([], "Am")
    with pytest.raises(ValueError, match="matrix count"):
        _core.k_assignment([np.zeros((2, 2))] * 2, "Bm")
    with pytest.raises(ValueError, match="partite 1"):
        _core.k_assignment([np.zeros((2, 3))] * 3, "Dm")
    for matrices in [
        [np.zeros((2, 2), dtype=np.float32)],
        [np.zeros(2)],
        [np.zeros((2, 2), dtype=np.int64), np.zeros((2, 2)), np.zeros((2, 2))],
    ]:
        with pytest.raises(TypeError, match="all float64 or all int64"):
            _core.k_assignment(matrices, "Am")
    with pytest.raises(ValueError, match="algo must be"):
        _core.k_assignment([np.zeros((2, 2))], "unknown")
    for value in [np.nan, np.inf]:
        with pytest.raises(ValueError):
            _core.k_assignment([np.full((1, 1), value)], "Am")
