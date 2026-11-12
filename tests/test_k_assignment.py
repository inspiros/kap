"""Exercise clique costs, coverage, heuristics, and public input validation."""

from itertools import product

import numpy as np
import pytest

import kap

from conftest import ALGORITHMS, DTYPES, assert_valid_assignment, brute_force_optimum, clique_cost


pytestmark = pytest.mark.usefixtures("scipy_assignment")

# A short iteration budget keeps Fm's exploration of tied integer solutions
# bounded. These are correctness checks, not a heuristic quality benchmark.
SOLVER_OPTIONS = dict(backend="scipy", return_free=True, random_state=42, max_iter=5)


def random_cases():
    for algo, k, equal, dtype in product(
        ALGORITHMS, (3, 4, 5), (True, False), (np.float64, np.int64)
    ):
        yield pytest.param(
            algo, k, equal, dtype,
            id=f"{algo}-k{k}-{'equal' if equal else 'unequal'}-{np.dtype(dtype).name}",
        )


def test_paper_example(paper_costs):
    # Explicit edges also sanity-check the independent oracle against the README.
    paper_matches = [
        [(0, 0), (1, 1), (2, 0)],
        [(0, 1), (1, 0), (2, 1)],
        [(0, 2), (1, 2), (2, 2)],
    ]
    assert [clique_cost(paper_costs, 3, [m]) for m in paper_matches] == [23, 4, 11]
    result = kap.k_assignment(paper_costs, algo="Em", **SOLVER_OPTIONS)
    assert_valid_assignment(paper_costs, (3, 3, 3), result)
    assert sum(result.matching_costs) == 38


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_paper_example_valid_for_every_algorithm(paper_costs, algo):
    result = kap.k_assignment(paper_costs, algo=algo, **SOLVER_OPTIONS)
    assert_valid_assignment(paper_costs, (3, 3, 3), result)


@pytest.mark.parametrize("algo,k,equal,dtype", tuple(random_cases()))
def test_clique_costs_and_node_coverage(instance_factory, algo, k, equal, dtype):
    sizes = (3,) * k if equal else tuple(2 + p % 3 for p in range(k))
    costs = instance_factory(k, sizes=sizes, dtype=dtype, seed=1729 + k)
    result = kap.k_assignment(costs, algo=algo, **SOLVER_OPTIONS)
    # Checking each clique against original matrices catches quotient copies
    # that incorrectly use renumbered partites, especially for k >= 4.
    assert_valid_assignment(costs, sizes, result)


@pytest.mark.parametrize("algo", ("Cm", "Em", "Fm"))
def test_unequal_sizes_with_empty_fixed_pair(instance_factory, algo):
    # Regression: a partite pair with no fixed edges used to crash quotient().
    sizes = (2, 3, 4, 2)
    costs = instance_factory(4, sizes=sizes, dtype=np.int64, seed=104)
    result = kap.k_assignment(costs, algo=algo, **SOLVER_OPTIONS)
    assert_valid_assignment(costs, sizes, result)


@pytest.fixture(
    params=((3, 3, 3), (3, 4, 7), (4, 3, 11), (5, 2, 19)),
    ids=("k3-n3-seed3", "k3-n4-seed7", "k4-n3-seed11", "k5-n2-seed19"),
    scope="module",
)
def tiny_instance(request):
    # Compute the factorial oracle once per instance, not once per algorithm.
    from conftest import random_instance

    k, n, seed = request.param
    costs = random_instance(k, n=n, seed=seed)
    return costs, k, n, brute_force_optimum(costs, k)


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_against_brute_force(tiny_instance, algo):
    costs, k, n, optimum = tiny_instance
    result = kap.k_assignment(costs, algo=algo, **SOLVER_OPTIONS)
    assert_valid_assignment(costs, (n,) * k, result)
    true_cost = clique_cost(costs, k, result.matches)
    assert true_cost >= optimum - 1e-8
    # Empirically verified on these four seeds; no general optimality claim.
    if algo in ("Bm", "Cm", "Em", "Fm"):
        assert np.isclose(true_cost, optimum)


@pytest.mark.parametrize("algo,k,equal,dtype", tuple(random_cases()))
def test_maximize(instance_factory, algo, k, equal, dtype):
    sizes = (3,) * k if equal else tuple(2 + p % 3 for p in range(k))
    costs = instance_factory(k, sizes=sizes, dtype=dtype, seed=789 + k)
    minimum = kap.k_assignment(costs, algo=algo, **SOLVER_OPTIONS)
    maximum = kap.k_assignment(costs, algo=algo, maximize=True, **SOLVER_OPTIONS)
    assert_valid_assignment(costs, sizes, minimum)
    assert_valid_assignment(costs, sizes, maximum)
    assert clique_cost(costs, k, maximum.matches) >= clique_cost(costs, k, minimum.matches) - 1e-8


@pytest.mark.parametrize("algo", ALGORITHMS)
@pytest.mark.parametrize("shape", ((4, 4), (2, 5), (5, 2)))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("maximize", (False, True), ids=("min", "max"))
def test_two_partites_equal_lap_optimum(
    instance_factory, scipy_assignment, algo, shape, dtype, maximize
):
    costs = instance_factory(2, sizes=shape, dtype=dtype, seed=675)
    rows, cols = scipy_assignment(costs[0], maximize=maximize)
    result = kap.k_assignment(costs, algo=algo, maximize=maximize, **SOLVER_OPTIONS)
    assert_valid_assignment(costs, shape, result)
    assert np.isclose(clique_cost(costs, 2, result.matches), costs[0][rows, cols].sum())


@pytest.mark.parametrize("algo", ("Em", "Fm"))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("maximize", (False, True), ids=("min", "max"))
def test_random_state_reproducibility(instance_factory, algo, dtype, maximize):
    costs = instance_factory(4, n=3, dtype=dtype, seed=53)
    options = dict(SOLVER_OPTIONS, n_trials=3, maximize=maximize)
    first = kap.k_assignment(costs, algo=algo, **options)
    second = kap.k_assignment(costs, algo=algo, **options)
    assert_valid_assignment(costs, (3,) * 4, first)
    assert_valid_assignment(costs, (3,) * 4, second)
    assert first.matches == second.matches
    np.testing.assert_array_equal(first.matching_costs, second.matching_costs)
    assert first.free == second.free


@pytest.mark.parametrize("algo", ("Am", "Bm", "Dm"))
@pytest.mark.parametrize("backend", ("kap", "scipy", "lap", "munkres"))
@pytest.mark.parametrize("seed", (23, 91))
def test_backend_parity(instance_factory, algo, backend, seed):
    pytest.importorskip("kap._core" if backend == "kap" else backend)
    costs = instance_factory(4, n=3, seed=seed)
    reference = kap.k_assignment(costs, algo=algo, **SOLVER_OPTIONS)
    result = kap.k_assignment(costs, algo=algo, **dict(SOLVER_OPTIONS, backend=backend))
    assert_valid_assignment(costs, (3,) * 4, result)
    assert np.isclose(sum(result.matching_costs), sum(reference.matching_costs))


@pytest.mark.parametrize(
    "costs",
    (
        pytest.param([], id="empty"),
        pytest.param([np.ones((2, 2))] * 2, id="invalid-matrix-count"),
        pytest.param([np.ones(3)], id="one-dimensional"),
        pytest.param([np.ones((2, 2, 2))], id="three-dimensional"),
        pytest.param([np.array(1)], id="scalar"),
        pytest.param(
            [np.ones((2, 3)), np.ones((4, 2)), np.ones((3, 2))],
            id="inconsistent-row-size",
        ),
        pytest.param(
            [np.ones((2, 3)), np.ones((2, 4)), np.ones((3, 5))],
            id="inconsistent-column-size",
        ),
    ),
)
def test_invalid_cost_matrices_raise_value_error(costs):
    with pytest.raises(ValueError):
        kap.k_assignment(costs)


@pytest.mark.parametrize("option", ({"algo": "unknown"}, {"backend": "unknown"}))
def test_unknown_solver_option_raises_value_error(paper_costs, option):
    with pytest.raises(ValueError, match=next(iter(option))):
        kap.k_assignment(paper_costs, **option)


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_free_is_none_by_default(paper_costs, algo):
    assert kap.k_assignment(paper_costs, algo=algo, random_state=42, max_iter=5).free is None


# Regression tests for 0.2.3 fixes.

from conftest import random_instance  # noqa: E402
import kap._reference.assignment as _assignment  # noqa: E402


def test_lap_calls_never_request_free(monkeypatch, paper_costs):
    """matching_filter must not land in linear_assignment's return_free slot."""
    original = _assignment.linear_assignment

    def checked(cost_matrix, maximize=False, return_free=False, backend="scipy"):
        assert return_free is False
        return original(cost_matrix, maximize, return_free, backend)

    monkeypatch.setattr(_assignment, "linear_assignment", checked)
    G = _assignment.KPartiteGraph([np.asarray(c) for c in paper_costs], 3)
    _assignment.solve_Am(G, matching_filter=object(), backend="scipy")
    _assignment.solve_Bm(G, matching_filter=object(), backend="scipy")


def test_em_trials_are_independent(monkeypatch):
    """Every Em trial starts from the Bm solution, not from the previous trial."""
    costs = random_instance(4, seed=0, n=6)  # Em improves on Bm here
    G = _assignment.KPartiteGraph(costs, 4)
    bm = _assignment.solve_Bm(G, backend="scipy").vertices()
    starts = []
    original = _assignment._Em_loop

    def recording(G_, G_init, *args, **kwargs):
        starts.append(G_init.vertices())
        return original(G_, G_init, *args, **kwargs)

    monkeypatch.setattr(_assignment, "_Em_loop", recording)
    _assignment.solve_Em(G, n_trials=4, backend="scipy", random_state=0)
    assert starts == [bm] * 4


def test_em_more_trials_never_worse():
    costs = random_instance(4, seed=5, n=6)
    one = kap.k_assignment(costs, algo="Em", n_trials=1, random_state=1)
    many = kap.k_assignment(costs, algo="Em", n_trials=10, random_state=1)
    assert sum(many.matching_costs) <= sum(one.matching_costs)


def test_fm_stops_on_a_plateau(monkeypatch):
    """Fm must stop when no different solution is at least as good, not run to max_iter."""
    costs = random_instance(4, seed=0, n=5, dtype=np.int64)
    G = _assignment.KPartiteGraph(costs, 4)
    calls = []
    original = _assignment.solve_Bm

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_assignment, "solve_Bm", counting)
    max_iter = 1000
    result = _assignment.solve_Fm(G, max_iter=max_iter, backend="scipy", random_state=0)
    n_pairs = 6
    assert len(calls) < 50 * n_pairs  # was ~max_iter * (n_pairs - 1) before the fix
    assert result.vertices_cost() <= original(G, backend="scipy").vertices_cost()
