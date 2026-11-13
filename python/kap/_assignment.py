"""Public k-assignment dispatch to the native core or the Python reference."""

import sys
from typing import Optional, Sequence, Union

import numpy as np

from . import _reference
from ._reference._types import Array2DLike
from ._reference.assignment import (
    AssignmentResult,
    _validate_algo,
    _validate_cost_matrices,
    _validate_search_options,
)

__all__ = ["ils", "memetic", "local_search", "multiple_hungarian", "k_assignment"]


def _seed(random_state):
    if isinstance(random_state, (int, np.integer)):
        if random_state < 0:
            raise ValueError(f"random_state must be non-negative. Got {random_state}.")
        return int(random_state) & ((1 << 64) - 1)
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, 1 << 64, dtype=np.uint64))
    if random_state is None:
        random_state = np.random
    elif not isinstance(random_state, np.random.RandomState):
        raise TypeError("random_state must be None, an int, RandomState, or Generator")
    return int(random_state.randint(0, 1 << 64, dtype=np.uint64))


_ALIASES = {"shift": "shift", "shiftam": "shift", "1dv": "1dv", "dv1": "1dv",
            "sdv": "sdv", "dvs": "sdv", "ils": "ils", "memetic": "memetic"}


def _core():
    try:
        from ._core import k_assignment
        return k_assignment
    except ImportError:
        return None


def _neighborhood(value, name="neighborhood"):
    if not isinstance(value, str) or value.lower() not in ("1dv", "dv1", "sdv", "dvs"):
        raise ValueError(f"{name} must be 1DV | sDV")
    return _ALIASES[value.lower()]


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}")
    return int(value)


def _budget(value, time_limit, name):
    if time_limit is not None:
        try:
            valid = bool(np.isfinite(time_limit)) and time_limit > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("time_limit must be finite and positive")
    return (sys.maxsize if time_limit is not None else 100) if value is None else _integer(value, name, 0)


def _initial_cliques(initial, sizes):
    if isinstance(initial, AssignmentResult):
        initial = initial.matches
    try:
        cliques = [list(clique) for clique in initial]
    except TypeError:
        raise ValueError("initial must be an iterable of cliques of (part, vertex) pairs") from None
    used = set()
    result = []
    for clique in cliques:
        if not clique:
            raise ValueError("initial contains an empty clique")
        parts = set()
        row = []
        for node in clique:
            try:
                part, vertex = node
            except (TypeError, ValueError):
                raise ValueError("initial vertices must be (part, vertex) pairs") from None
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                   for v in (part, vertex)):
                raise ValueError("initial part and vertex indices must be integers")
            part, vertex = int(part), int(vertex)
            if not 0 <= part < len(sizes):
                raise ValueError(f"initial part index {part} is out of range")
            if not 0 <= vertex < sizes[part]:
                raise ValueError(f"initial vertex index {vertex} is out of range for part {part}")
            if (part, vertex) in used:
                raise ValueError(f"initial vertex {(part, vertex)} is used more than once")
            if part in parts:
                raise ValueError(f"initial clique contains more than one vertex of part {part}")
            parts.add(part)
            used.add((part, vertex))
            row.append((part, vertex))
        result.append(row)
    expected = max(sizes)
    actual = len(result) + sum(sizes) - len(used)
    if actual != expected:
        raise ValueError(f"initial must be a complete solution: {expected} nodes expected, {actual} given")
    return result


def _promote_cost_matrices(cost_matrices):
    """Promote validated matrices to the native core's common numeric dtype."""
    for matrix in cost_matrices:
        if matrix.dtype.kind not in "biuf":
            raise TypeError(f"unsupported cost matrix dtype {matrix.dtype}")
    dtype = np.int64 if all(m.dtype.kind in "biu" for m in cost_matrices) else np.float64
    if dtype is np.int64:
        for matrix in cost_matrices:
            if (matrix.dtype.kind == "u" and matrix.dtype.itemsize == 8
                    and matrix.size and int(matrix.max()) > np.iinfo(np.int64).max):
                raise OverflowError("uint64 cost matrix values exceed the int64 range "
                                    "supported by the native solver; convert to float64 "
                                    "or use another backend.")
    return [np.asarray(matrix, dtype=dtype) for matrix in cost_matrices]


def _native(cost_matrices, algo, *, max_iter=100, n_trials=1, maximize=False,
            return_free=False, random_state=None, refine=None, time_limit=None, population=None,
            initial=None):
    native = _core()
    if native is None:
        raise RuntimeError("the native core is required")
    cost_matrices, n_partites = _validate_cost_matrices(cost_matrices)
    if initial is not None:
        sizes = [cost_matrices[0].shape[0]] + [m.shape[1] for m in cost_matrices[:n_partites - 1]]
        initial = _initial_cliques(initial, sizes)
    matrices = _promote_cost_matrices(cost_matrices)
    matches, costs, free = native(matrices, algo, maximize=bool(maximize), max_iter=max_iter,
                                  n_trials=n_trials, seed=_seed(random_state), refine=refine,
                                  time_limit=time_limit, population=population, initial=initial)
    return AssignmentResult(matches, costs, free if return_free else None)


def ils(cost_matrices, *, max_iter=None, n_chains=1, time_limit=None, neighborhood="1dv",
        maximize=False, return_free=False, random_state=None, initial=None):
    """Run iterated local search, perturbing and reoptimizing independent chains.

    The best solution found is returned. The native core is required.

    Args:
        cost_matrices: Pairwise cost matrices in combinations order.
        max_iter: Nonnegative perturbations per chain; default 100, or unlimited with a time limit.
        n_chains: Positive number of independent chains.
        time_limit: Finite positive seconds; applies alongside max_iter when both are set.
        neighborhood: "1dv" or "sdv" (also "dv1" or "dvs"), case-insensitive.
        maximize: Maximize the score instead of minimizing it.
        return_free: Include unmatched vertices in the result.
        random_state: None, integer, RandomState or Generator; use an integer for reproducibility.
        initial: Iterable of cliques of (part, vertex) pairs, or an AssignmentResult
            whose matches are used, replacing the Am start. Must be a complete
            solution such as a previous result: after unlisted vertices are added
            as singletons, there must be exactly max(part sizes) nodes.
            DV refines this start before all chains.
            The result is never worse than the start for integer costs; for float
            costs this holds up to floating-point rounding of the cost sums.

    Returns:
        AssignmentResult: Matches, their costs, and optionally free vertices.

    A time limit makes results dependent on machine speed.
    """
    iterations = _budget(max_iter, time_limit, "max_iter")
    chains = _integer(n_chains, "n_chains", 1)
    return _native(cost_matrices, "ils", max_iter=iterations, n_trials=chains,
                   time_limit=time_limit, refine=_neighborhood(neighborhood), maximize=maximize,
                   return_free=return_free, random_state=random_state, initial=initial)


def memetic(cost_matrices, *, generations=None, population=None, time_limit=None,
            neighborhood="1dv", maximize=False, return_free=False, random_state=None):
    """Evolve a population of locally optimized solutions using mutation and crossover.

    Each candidate is refined with dimensionwise variation. The native core is required.

    Args:
        cost_matrices: Pairwise cost matrices in combinations order.
        generations: Nonnegative generation limit; default 100, or unlimited with a time limit.
        population: Population size, an integer >= 2, or None for the native default.
        time_limit: Finite positive seconds; applies alongside generations when both are set.
        neighborhood: "1dv" or "sdv" (also "dv1" or "dvs"), case-insensitive.
        maximize: Maximize the score instead of minimizing it.
        return_free: Include unmatched vertices in the result.
        random_state: None, integer, RandomState or Generator; use an integer for reproducibility.

    Returns:
        AssignmentResult: Matches, their costs, and optionally free vertices.

    A time limit makes results dependent on machine speed.
    """
    iterations = _budget(generations, time_limit, "generations")
    if population is not None:
        if isinstance(population, (bool, np.bool_)) or not isinstance(population, (int, np.integer)) or population < 2:
            raise ValueError("population must be an int >= 2")
        population = int(population)
    return _native(cost_matrices, "memetic", max_iter=iterations, population=population,
                   time_limit=time_limit, refine=_neighborhood(neighborhood), maximize=maximize,
                   return_free=return_free, random_state=random_state)


def local_search(cost_matrices, *, neighborhood="1dv", start=None, initial=None, max_sweeps=100,
                 maximize=False, return_free=False):
    """Build a deterministic starting solution and improve it with DV sweeps.

    Search ends at a local optimum or the sweep limit. The native core is required.

    Args:
        cost_matrices: Pairwise cost matrices in combinations order.
        neighborhood: "1dv" or "sdv" (also "dv1" or "dvs"), case-insensitive.
        start: Starting method "Am", "Bm", "Dm", or "Shift", case-insensitive;
            None selects Am. Cannot be supplied together with initial.
        initial: Iterable of cliques of (part, vertex) pairs, or an AssignmentResult
            whose matches are used, replacing the constructed start. Must be a complete
            solution such as a previous result: after unlisted vertices are added
            as singletons, there must be exactly max(part sizes) nodes.
            The result is never worse than the start for integer costs; for float
            costs this holds up to floating-point rounding of the cost sums.
        max_sweeps: Nonnegative maximum number of sweeps.
        maximize: Maximize the score instead of minimizing it.
        return_free: Include unmatched vertices in the result.

    Returns:
        AssignmentResult: Matches, their costs, and optionally free vertices.

    This method is deterministic and needs no random seed.
    """
    starts = {"am": "a", "bm": "b", "dm": "d", "shift": "shift"}
    if start is not None and initial is not None:
        raise ValueError("start and initial cannot both be supplied")
    if start is None:
        start = "Am"
    if not isinstance(start, str) or start.lower() not in starts:
        raise ValueError("start must be Am | Bm | Dm | Shift")
    return _native(cost_matrices, starts[start.lower()], max_iter=_integer(max_sweeps, "max_sweeps", 0),
                   refine=_neighborhood(neighborhood), maximize=maximize, return_free=return_free,
                   initial=initial)


def multiple_hungarian(cost_matrices, method="Cm", *, max_iter=100, n_trials=1,
                       maximize=False, return_free=False, backend="kap", random_state=None):
    """Solve using a multiple Hungarian method from the paper, or cyclic Shift.

    Am–Fm use the selected LAP backend and fall back to the Python reference when
    the native core is unavailable. Shift requires the native core and backend="kap".

    Args:
        cost_matrices: Pairwise cost matrices in combinations order.
        method: "Am" through "Fm", or "Shift", case-insensitive.
        max_iter: Move limit for Cm, Em and Fm; ignored for other methods.
        n_trials: Number of independent Em trials.
        maximize: Maximize the score instead of minimizing it.
        return_free: Include unmatched vertices in the result.
        backend: LAP backend: "kap", "scipy", "lap", "lapjv", or "munkres".
        random_state: None, integer, RandomState or Generator; use an integer for reproducible Em/Fm results.

    Returns:
        AssignmentResult: Matches, their costs, and optionally free vertices.
    """
    shift = isinstance(method, str) and method.lower() in ("shift", "shiftam")
    if not shift:
        try:
            algo = _validate_algo(method)
        except (ValueError, AttributeError):
            raise ValueError(f"method must be Am | Bm | Cm | Dm | Em | Fm | Shift; got {method}") from None
    else:
        algo = "shift"
    native = isinstance(backend, str) and backend.lower() == "kap" and _core() is not None
    if not native:
        if shift:
            raise ValueError('Shift needs backend="kap" (native core)')
        return _reference.k_assignment(cost_matrices, algo=method, max_iter=max_iter, n_trials=n_trials,
                                       maximize=maximize, return_free=return_free, backend=backend,
                                       random_state=random_state)
    _validate_search_options(algo, max_iter, n_trials)
    iterations = max_iter if algo in ("e", "f") else max(0, max_iter) if algo == "c" else 100
    return _native(cost_matrices, algo, max_iter=iterations, n_trials=n_trials if algo == "e" else 1,
                   maximize=maximize, return_free=return_free, random_state=random_state)


def k_assignment(cost_matrices: Sequence[Array2DLike], algo: Optional[str] = None,
                 max_iter: Optional[int] = None, n_trials: int = 1, maximize: bool = False,
                 return_free: bool = False, backend: str = "kap",
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
                 refine: Optional[str] = None, time_limit: Optional[float] = None,
                 population: Optional[int] = None) -> AssignmentResult:
    """Dispatch a k-assignment problem to a search or multiple Hungarian method.

    Args:
        cost_matrices: Pairwise cost matrices in combinations order.
        algo: Am–Fm, Shift, 1dv, sdv, ils or memetic; default ils with the
            native core and backend="kap", otherwise Cm. Aliases are case-insensitive.
        max_iter: Algorithm limit; None selects 100, or unlimited for a timed search.
            Explicit limits also apply with time_limit. Negative limits are treated
            as zero except for Em and Fm, which require positive limits.
        n_trials: Independent Em trials or ILS chains.
        maximize: Maximize total weight instead of minimizing it.
        return_free: Include unmatched vertices.
        backend: LAP backend; newer methods and refinement require backend="kap".
        random_state: None, integer, RandomState or Generator; integers reproduce seeded runs.
        refine: 1dv/dv1 or sdv/dvs; polish the result, or select the ILS/memetic neighborhood.
        time_limit: Finite positive seconds for ILS or memetic; results depend on machine speed.
        population: Memetic population size, at least 2, or None for the native default.

    Returns:
        AssignmentResult: Matches, their costs, and optionally free vertices.
    """
    native = isinstance(backend, str) and backend.lower() == "kap" and _core() is not None
    if algo is None:
        algo = "ils" if native else "Cm"
    new_algo = _ALIASES.get(algo.lower()) if isinstance(algo, str) else None
    if time_limit is not None and new_algo not in ("ils", "memetic"):
        raise ValueError('time_limit requires algo="ils" or "memetic"')
    if population is not None and new_algo != "memetic":
        raise ValueError('population requires algo="memetic"')
    if refine is not None:
        refine = _neighborhood(refine, "refine")
    if not native and (new_algo is not None or refine is not None or time_limit is not None):
        raise ValueError('Shift, 1DV, sDV, ILS, memetic, refine and time_limit need backend="kap" (native core)')
    common = dict(maximize=maximize, return_free=return_free)
    if new_algo in ("ils", "memetic", "1dv", "sdv"):
        # Preserve the dispatcher's historical treatment of negative limits.
        limit = None if max_iter is None else max(0, max_iter)
        if new_algo == "ils":
            _validate_search_options("e", 1, n_trials)
            return ils(cost_matrices, max_iter=limit, n_chains=n_trials, time_limit=time_limit,
                       neighborhood=refine or "1dv", random_state=random_state, **common)
        if new_algo == "memetic":
            return memetic(cost_matrices, generations=limit, population=population, time_limit=time_limit,
                           neighborhood=refine or "1dv", random_state=random_state, **common)
        if refine is None:
            return local_search(cost_matrices, neighborhood=new_algo, max_sweeps=100 if limit is None else limit, **common)
        return _native(cost_matrices, new_algo, max_iter=100 if limit is None else limit,
                       refine=refine, random_state=random_state, **common)
    limit = 100 if max_iter is None else max_iter
    if refine is not None:
        normalized = new_algo or _validate_algo(algo)
        _validate_search_options(normalized, limit, n_trials)
        return _native(cost_matrices, normalized, max_iter=max(0, limit),
                       n_trials=n_trials if normalized == "e" else 1, refine=refine,
                       random_state=random_state, **common)
    if new_algo is None and native:
        # Preserve the reference's error text for dispatcher callers.
        _validate_algo(algo)
    return multiple_hungarian(cost_matrices, method=algo, max_iter=limit, n_trials=n_trials,
                              backend=backend, random_state=random_state, **common)
