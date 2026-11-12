"""Public k-assignment dispatch to the native core or the Python reference."""

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
                 algo: str = "Cm",
                 max_iter: int = 100,
                 n_trials: int = 1,
                 maximize: bool = False,
                 return_free: bool = False,
                 backend: str = "kap",
                 random_state: Optional[Union[np.random.RandomState, int]] = None) -> AssignmentResult:
    native_k_assignment = None
    if isinstance(backend, str) and backend.lower() == "kap":
        try:
            from ._core import k_assignment as native_k_assignment
        except ImportError:
            pass
    if native_k_assignment is None:
        return _reference.k_assignment(
            cost_matrices, algo=algo, max_iter=max_iter, n_trials=n_trials,
            maximize=maximize, return_free=return_free, backend=backend,
            random_state=random_state,
        )

    cost_matrices, _ = _validate_cost_matrices(cost_matrices)
    algo = _validate_algo(algo)
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
    if algo == "c":
        iterations = max(0, max_iter)
    elif algo in ("e", "f"):
        iterations = max_iter
    trials = n_trials if algo == "e" else 1
    matches, matching_costs, free = native_k_assignment(
        matrices, algo, maximize=bool(maximize), max_iter=iterations,
        n_trials=trials, seed=_seed(random_state),
    )
    return AssignmentResult(matches, matching_costs, free if return_free else None)


k_assignment.__doc__ = _reference.k_assignment.__doc__
