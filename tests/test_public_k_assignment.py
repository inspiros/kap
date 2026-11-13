"""Public native dispatch, dtype promotion, RNGs, and reference compatibility."""

import inspect
import sys

import numpy as np
import pytest

import kap
from kap import _core, _reference
from kap._reference.assignment import AssignmentResult

from conftest import ALGORITHMS, assert_valid_assignment, random_instance


def instance(k, equal, dtype):
    sizes = (3,) * k if equal else tuple(2 + p % 3 for p in range(k))
    matrices = random_instance(k, sizes=sizes, seed=1729 + k, dtype=dtype)
    if dtype == np.bool_:
        rng = np.random.default_rng(1729 + k)
        matrices = [rng.integers(0, 2, size=m.shape).astype(dtype) for m in matrices]
    elif np.issubdtype(dtype, np.floating):
        # Exact binary fractions keep merge-order roundoff from reordering tied
        # assignments, allowing full result parity for both floating dtypes.
        matrices = [m.astype(dtype) / 8 for m in
                    random_instance(k, sizes=sizes, seed=1729 + k, dtype=np.int64)]
    return matrices, sizes


@pytest.mark.parametrize("algo", ("Am", "Bm", "Dm"))
@pytest.mark.parametrize("k", range(2, 7))
@pytest.mark.parametrize("equal", (True, False))
@pytest.mark.parametrize("dtype", (np.int64, np.float64, np.int32, np.float32, np.bool_))
@pytest.mark.parametrize("maximize", (False, True))
def test_reference_parity(algo, k, equal, dtype, maximize):
    matrices, sizes = instance(k, equal, dtype)
    options = dict(algo=algo, maximize=maximize, return_free=True, random_state=42)
    actual = kap.k_assignment(matrices, **options)
    # Boolean quotient sums in the reference use logical OR. Compare arithmetic
    # costs to its int64 path, matching the public API's required promotion.
    oracle_matrices = [m.astype(np.int64) for m in matrices] if dtype == np.bool_ else matrices
    expected = _reference.k_assignment(oracle_matrices, **options)
    assert isinstance(actual, AssignmentResult)
    assert actual.matches == expected.matches
    assert actual.free == expected.free
    if np.issubdtype(dtype, np.floating):
        np.testing.assert_allclose(actual.matching_costs, expected.matching_costs, rtol=1e-6)
    else:
        assert actual == expected
    assert_valid_assignment(matrices, sizes, actual)
    assert isinstance(actual.matches, list)
    assert isinstance(actual.matching_costs, list)
    assert isinstance(actual.free, list)
    assert all(isinstance(match, list) for match in actual.matches)
    assert all(isinstance(node, tuple) for match in actual.matches for node in match)
    assert all(isinstance(node, tuple) for node in actual.free)


@pytest.mark.parametrize("algo", ("Am", "Bm", "Dm"))
@pytest.mark.parametrize("k", range(2, 7))
@pytest.mark.parametrize("equal", (True, False))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("maximize", (False, True))
def test_arbitrary_floating_costs(algo, k, equal, dtype, maximize):
    matrices = random_instance(k, equal=equal, dtype=dtype, seed=1729 + k)
    options = dict(algo=algo, maximize=maximize, return_free=True)
    actual = kap.k_assignment(matrices, **options)
    expected = _reference.k_assignment(matrices, **options)
    # Different summation orders can select tied merge trees and reorder the
    # cliques. Compare the same cliques and their associated costs in order.
    actual_pairs = sorted(zip(actual.matches, actual.matching_costs))
    expected_pairs = sorted(zip(expected.matches, expected.matching_costs))
    assert [m for m, _ in actual_pairs] == [m for m, _ in expected_pairs]
    np.testing.assert_allclose([c for _, c in actual_pairs], [c for _, c in expected_pairs],
                               rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(sum(actual.matching_costs), sum(expected.matching_costs),
                               rtol=1e-6, atol=1e-5)
    sizes = [matrices[0].shape[0]] + [m.shape[1] for m in matrices[:k - 1]]
    assert_valid_assignment(matrices, sizes, actual)


@pytest.mark.parametrize("algo", ("Cm", "Em", "Fm"))
@pytest.mark.parametrize("k", range(2, 7))
@pytest.mark.parametrize("equal", (True, False))
@pytest.mark.parametrize("dtype", (np.int64, np.float64, np.int32, np.float32, np.bool_))
@pytest.mark.parametrize("maximize", (False, True))
def test_local_search_valid_and_reproducible(algo, k, equal, dtype, maximize):
    matrices, sizes = instance(k, equal, dtype)
    options = dict(algo=algo, maximize=maximize, return_free=True,
                   random_state=42, max_iter=5, n_trials=3)
    actual = kap.k_assignment(matrices, **options)
    assert_valid_assignment(matrices, sizes, actual)
    assert actual == kap.k_assignment(matrices, **options)


@pytest.mark.parametrize("algo", ("Cm", "Em", "Fm"))
@pytest.mark.parametrize("rng_factory", (np.random.RandomState, np.random.default_rng))
@pytest.mark.parametrize("maximize", (False, True))
def test_seeded_rng_reproducibility(algo, rng_factory, maximize):
    matrices = random_instance(4, equal=False, seed=53)
    options = dict(algo=algo, maximize=maximize, return_free=True, max_iter=5, n_trials=3)
    first = kap.k_assignment(matrices, random_state=rng_factory(42), **options)
    second = kap.k_assignment(matrices, random_state=rng_factory(42), **options)
    assert first == second
    assert_valid_assignment(matrices, (2, 3, 4, 2), first)


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_return_free_false(paper_costs, algo):
    assert kap.k_assignment(paper_costs, algo=algo, random_state=42, max_iter=5).free is None


def test_signature_docstring_and_linear_assignment_unchanged():
    public = inspect.signature(kap.k_assignment)
    reference = inspect.signature(_reference.k_assignment)
    shared = list(public.parameters.values())[:-3]
    assert [p.name for p in shared] == list(reference.parameters)
    assert [p.default for p in shared if p.name not in ("algo", "max_iter")] == \
        [p.default for p in reference.parameters.values() if p.name not in ("algo", "max_iter")]
    assert public.parameters["max_iter"].default is None
    assert public.parameters["algo"].default is None
    assert list(public.parameters)[-3:] == ["refine", "time_limit", "population"]
    assert public.parameters["population"].default is None
    assert public.parameters["time_limit"].default is None
    assert public.parameters["refine"].default is None
    assert "1dv" in kap.k_assignment.__doc__
    assert 'backend="kap"' in kap.k_assignment.__doc__
    assert kap.linear_assignment is _reference.linear_assignment


def test_import_failure_delegates_unchanged(monkeypatch, paper_costs):
    options = dict(algo="Em", max_iter=5, n_trials=3, maximize=True,
                   return_free=True, backend="kap", random_state=42)
    expected = _reference.k_assignment(paper_costs, **options)
    reference = _reference.k_assignment
    calls = []

    def recording(matrices, **kwargs):
        calls.append((matrices, kwargs))
        return reference(matrices, **kwargs)

    monkeypatch.setattr(_reference, "k_assignment", recording)
    monkeypatch.setitem(sys.modules, "kap._core", None)
    assert kap.k_assignment(paper_costs, **options) == expected
    assert len(calls) == 1
    assert calls[0][0] is paper_costs
    assert calls[0][1] == options


@pytest.mark.parametrize("matrices", (
    [],
    [np.ones((2, 2))] * 2,
    [np.ones(3)],
    [np.ones((2, 2, 2))],
    [np.array(1)],
    [np.ones((2, 3)), np.ones((4, 2)), np.ones((3, 2))],
    [np.ones((2, 3)), np.ones((2, 4)), np.ones((3, 5))],
))
def test_matrix_error_messages_match_reference(matrices):
    with pytest.raises(ValueError) as expected:
        _reference.k_assignment(matrices)
    with pytest.raises(ValueError) as actual:
        kap.k_assignment(matrices)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("options", (
    {"algo": "unknown"}, {"algo": "Zm"}, {"algo": ""},
    {"algo": "Em", "max_iter": 0}, {"algo": "Fm", "max_iter": -1},
    {"algo": "Em", "n_trials": 0},
))
def test_option_error_messages_match_reference(paper_costs, options):
    with pytest.raises(ValueError) as expected:
        _reference.k_assignment(paper_costs, **options)
    with pytest.raises(ValueError) as actual:
        kap.k_assignment(paper_costs, **options)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("dtype", (np.complex128, object, "U4", "datetime64[D]"))
def test_unsupported_dtypes(dtype):
    matrices = [np.zeros((2, 2)).astype(dtype)]
    with pytest.raises(TypeError, match="unsupported cost matrix dtype"):
        kap.k_assignment(matrices)


@pytest.mark.parametrize("layout", ("fortran", "negative", "stepped", "byte-swapped"))
@pytest.mark.parametrize("dtype", (np.int32, np.float32))
def test_layouts_and_native_byte_order(layout, dtype):
    from test_native_k_assignment import strided

    matrices = random_instance(4, equal=False, dtype=dtype)
    if layout == "byte-swapped":
        matrices = [m.astype(m.dtype.newbyteorder("S")) for m in matrices]
    else:
        matrices = strided(matrices, layout)
    actual = kap.k_assignment(matrices, algo="Am", return_free=True)
    expected = _reference.k_assignment(matrices, algo="Am", return_free=True)
    assert actual.matches == expected.matches
    np.testing.assert_allclose(actual.matching_costs, expected.matching_costs, rtol=1e-6)
    assert_valid_assignment(matrices, (2, 3, 4, 2), actual)


@pytest.mark.parametrize("kind", ("mixed", "float16", "lists", "stacked"))
def test_conversion_and_native_dispatch(monkeypatch, kind):
    matrices = random_instance(3, n=2, dtype=np.int32)
    if kind == "mixed":
        matrices[1] = matrices[1].astype(np.float32)
    elif kind == "float16":
        matrices = [m.astype(np.float16) for m in matrices]
    elif kind == "lists":
        matrices = [m.tolist() for m in matrices]
    else:
        matrices = np.stack(matrices)
    calls = []
    native = _core.k_assignment

    def recording(costs, *args, **kwargs):
        calls.append(costs)
        return native(costs, *args, **kwargs)

    monkeypatch.setattr(_core, "k_assignment", recording)
    actual = kap.k_assignment(matrices, algo="aM", backend="KAP", return_free=True)
    assert len(calls) == 1
    expected_dtype = np.float64 if kind in ("mixed", "float16") else np.int64
    assert all(m.dtype == np.dtype(expected_dtype) and m.dtype.isnative for m in calls[0])
    assert_valid_assignment([np.asarray(m) for m in matrices], (2, 2, 2), actual)


@pytest.mark.parametrize("random_state", (0, np.int64(42), (1 << 80) + 7,
                                        np.random.RandomState(42), np.random.default_rng(42)))
def test_seed_translation(monkeypatch, paper_costs, random_state):
    if isinstance(random_state, np.random.RandomState):
        expected = int(np.random.RandomState(42).randint(0, 1 << 64, dtype=np.uint64))
    elif isinstance(random_state, np.random.Generator):
        expected = int(np.random.default_rng(42).integers(0, 1 << 64, dtype=np.uint64))
    else:
        expected = int(random_state) & ((1 << 64) - 1)
    seeds = []

    def recording(*args, seed, **kwargs):
        seeds.append(seed)
        return [], [], []

    monkeypatch.setattr(_core, "k_assignment", recording)
    kap.k_assignment(paper_costs, random_state=random_state)
    assert seeds == [expected]


def test_none_draws_a_fresh_global_seed(monkeypatch, paper_costs):
    draws = iter((11, 29))
    seeds = []

    def randint(low, high, dtype):
        assert (low, high, dtype) == (0, 1 << 64, np.uint64)
        return next(draws)

    def recording(*args, seed, **kwargs):
        seeds.append(seed)
        return [], [], []

    monkeypatch.setattr(np.random, "randint", randint)
    monkeypatch.setattr(_core, "k_assignment", recording)
    kap.k_assignment(paper_costs)
    kap.k_assignment(paper_costs)
    assert seeds == [11, 29]


def test_negative_seed_rejected(paper_costs):
    with pytest.raises(ValueError, match="random_state must be non-negative"):
        kap.k_assignment(paper_costs, random_state=-1)


def test_invalid_random_state_rejected(paper_costs):
    with pytest.raises(TypeError, match="random_state"):
        kap.k_assignment(paper_costs, random_state=1.5)


def test_uint64_beyond_int64_rejected():
    big = np.full((2, 2), np.iinfo(np.int64).max, dtype=np.uint64) + np.uint64(1)
    with pytest.raises(OverflowError, match="int64 range"):
        kap.k_assignment([big, big, big], algo="Am")
    small = [np.arange(4, dtype=np.uint64).reshape(2, 2)] * 3
    assert kap.k_assignment(small, algo="Am") == kap._reference.k_assignment(small, algo="Am")
