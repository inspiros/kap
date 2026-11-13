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

__all__ = ["k_assignment"]


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


def k_assignment(cost_matrices: Sequence[Array2DLike],
                 algo: Optional[str] = None,
                 max_iter: int = 100,
                 n_trials: int = 1,
                 maximize: bool = False,
                 return_free: bool = False,
                 backend: str = "kap",
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
                 refine: Optional[str] = None,
                 time_limit: Optional[float] = None,
                 population: Optional[int] = None) -> AssignmentResult:
    r"""
    Solve the k-Assignment Problem using merge heuristics and local search.
    Gabrovšek's multiple Hungarian methods are described in ref. [1]_. This
    calls the LAP solver multiple times. See :func:`linear_assignment` for details about the available backends.

    Args:
        cost_matrices (sequence of ndarray): List of pairwise cost matrices,
            ordered as in ``itertools.combination(n, 2)``.
        algo (str): "Am", "Bm", "Cm", "Dm", "Em", "Fm", "shift" (or "ShiftAm"),
            "1dv" (or "dv1"), "sdv" (or "dvs"), "ils", or "memetic", case-insensitive.
            Shift keeps the best Am over cyclic part orders. 1DV and sDV start
            from Am and re-match one part or a subset against the rest.
            ILS perturbs an incumbent and runs DV to convergence at each step.
            Memetic evolves a population of DV local optima from Shift.
            The new algorithms require backend="kap" (native core).
            sDV supports at most 16 partites. Defaults to None: "ils" with
            backend="kap" (native core), otherwise "Cm". ILS is randomised;
            pass an int random_state for reproducible results.
        max_iter (int): Maximum number of moves for Cm, Em (per trial) and Fm.
            For memetic this caps generations (zero returns the best of the first
            generation). For ILS this caps perturbations per chain; for 1DV, sDV and other
            algorithms with refine it caps full sweeps. Zero returns the start.
            Negative values are treated as zero except for Em/Fm, which require
            a positive limit. Ignored otherwise. Defaults to 100.
        n_trials (int): Positive number of independent trials for Em or chains for ILS.
            This is ignored if other algorithm is used. Defaults to 1.
        maximize (bool): Calculates a maximum weight matching if true.
            Defaults to False.
        return_free (bool): Return unmatched vertices if true.
            Defaults to False.
        backend (str): The backend used to solve the linear assignment problem.
            Supported backends are "kap", "scipy", "lap", "lapjv", "munkres".
            Defaults to "kap".
        random_state (RandomState, Generator, int, optional): Controls the pseudo random
            number generation for shuffling or choosing random partites pair in
            Em, Fm, ILS and memetic algorithms. Pass an int for reproducible output across
            multiple function calls. Defaults to None.
        refine (str, optional): "1dv"/"dv1" or "sdv"/"dvs", case-insensitive.
            For ILS and memetic, select the DV neighbourhood (default 1DV); otherwise run
            dimensionwise variation after the selected algorithm. Moves
            strictly improve cost, including with unequal part sizes. Requires
            backend="kap" (native core). Defaults to None.
        time_limit (float, optional): Finite positive ILS or memetic budget in seconds.
            ILS checks between perturbations after the start; memetic counts from
            entry and checks between first-generation batches and generations.
            When given, it is the only budget and max_iter is ignored.
            Requires backend="kap" (native core). A time limit makes results
            depend on machine speed. Defaults to None.
        population (int, optional): Memetic only, at least 2. Without time_limit,
            use this fixed size (default 12) and max_iter generations, reproducibly
            per seed. With time_limit, use this size or estimate it from the budget
            and measured local-search time.
            n_trials is ignored for memetic. Defaults to None.

    Returns:
        matching_results: A namedtuple of ``matches``, ``matching_costs``,
            and optionally ``free`` vertices if ``return_free`` is True.

    References:
        [1] Multiple Hungarian Method for k-Assignment Problem.
            *Mathematics*, 8(11), 2050, November 2020, :doi:`10.3390/math8112050`
    """
    if algo is None:
        algo = "Cm"
        if isinstance(backend, str) and backend.lower() == "kap":
            try:
                from ._core import k_assignment as _  # noqa: F401
                algo = "ils"
            except ImportError:
                pass
    aliases = {"shift": "shift", "shiftam": "shift", "1dv": "1dv",
               "dv1": "1dv", "sdv": "sdv", "dvs": "sdv", "ils": "ils",
               "memetic": "memetic"}
    new_algo = aliases.get(algo.lower()) if isinstance(algo, str) else None
    if time_limit is not None:
        if new_algo not in ("ils", "memetic"):
            raise ValueError('time_limit requires algo="ils" or "memetic"')
        if not np.isfinite(time_limit) or time_limit <= 0:
            raise ValueError("time_limit must be finite and positive")
    if population is not None:
        if new_algo != "memetic":
            raise ValueError('population requires algo="memetic"')
        if isinstance(population, (bool, np.bool_)) or not isinstance(population, (int, np.integer)) or population < 2:
            raise ValueError("population must be an int >= 2")
        population = int(population)
    if refine is not None:
        if not isinstance(refine, str) or refine.lower() not in ("1dv", "dv1", "sdv", "dvs"):
            raise ValueError("refine must be 1DV | sDV")
        refine = aliases[refine.lower()]
    native_k_assignment = None
    if isinstance(backend, str) and backend.lower() == "kap":
        try:
            from ._core import k_assignment as native_k_assignment
        except ImportError:
            pass
    if native_k_assignment is None:
        if new_algo is not None or refine is not None or time_limit is not None:
            raise ValueError('Shift, 1DV, sDV, ILS, memetic, refine and time_limit need backend="kap" (native core)')
        return _reference.k_assignment(
            cost_matrices, algo=algo, max_iter=max_iter, n_trials=n_trials,
            maximize=maximize, return_free=return_free, backend=backend,
            random_state=random_state,
        )

    cost_matrices, _ = _validate_cost_matrices(cost_matrices)
    algo = new_algo or _validate_algo(algo)
    if algo == "ils":
        # Reuse Em's positive trial-count validation; ILS allows zero iterations.
        _validate_search_options("e", 1, n_trials)
    else:
        _validate_search_options(algo, max_iter, n_trials)
    for matrix in cost_matrices:
        if matrix.dtype.kind not in "biuf":
            raise TypeError(f"unsupported cost matrix dtype {matrix.dtype}")
    dtype = np.int64 if all(m.dtype.kind in "biu" for m in cost_matrices) else np.float64
    if dtype is np.int64:
        int64_max = np.iinfo(np.int64).max
        for matrix in cost_matrices:
            if matrix.dtype == np.uint64 and matrix.size and matrix.max() > int64_max:
                raise OverflowError("uint64 cost matrix values exceed the int64 range "
                                    "supported by the native solver; convert to float64 "
                                    "or use another backend.")
    matrices = [np.asarray(matrix, dtype=dtype) for matrix in cost_matrices]
    # Ignore options unused by the selected algorithm, as the reference does.
    iterations = 100
    if algo in ("e", "f"):
        iterations = max_iter
    elif algo in ("c", "1dv", "sdv", "ils", "memetic") or refine is not None:
        iterations = max(0, max_iter)
    if time_limit is not None:
        # The time limit is the budget; the default max_iter must not end the run early.
        iterations = sys.maxsize
    trials = n_trials if algo in ("e", "ils") else 1
    matches, matching_costs, free = native_k_assignment(
        matrices, algo, maximize=bool(maximize), max_iter=iterations,
        n_trials=trials, seed=_seed(random_state), refine=refine, time_limit=time_limit,
        population=population,
    )
    return AssignmentResult(matches, matching_costs, free if return_free else None)
