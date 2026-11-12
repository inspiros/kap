"""Native linear sum assignment (``kap._core``), scipy-compatible interface."""

import numpy as np

from . import _core

__all__ = ["linear_sum_assignment"]

_NATIVE_DTYPES = frozenset(np.dtype(t) for t in (
    np.float64, np.float32, np.int64, np.int32, np.int16, np.int8,
    np.uint32, np.uint16, np.uint8))
_INT64_MAX = np.iinfo(np.int64).max


def _as_native(cost: np.ndarray) -> np.ndarray:
    dtype = cost.dtype
    if dtype in _NATIVE_DTYPES:  # dtype equality includes the byte order
        return cost
    if dtype.kind in "fiu" and dtype.newbyteorder("=") in _NATIVE_DTYPES:
        return cost.astype(dtype.newbyteorder("="))
    if dtype.kind == "b":
        return cost.astype(np.uint8)
    if dtype.kind in "iu" and (cost.size == 0 or cost.max() <= _INT64_MAX):
        return cost.astype(np.int64)  # uint64 that fits
    if dtype.kind in "iuf":
        return cost.astype(np.float64)  # float16, longdouble, large uint64
    raise TypeError(f"unsupported cost matrix dtype {dtype}")


def linear_sum_assignment(cost_matrix, maximize: bool = False):
    """Solve the linear sum assignment problem with kap's native solver.

    Same interface and tie-breaking as :func:`scipy.optimize.linear_sum_assignment`:
    returns ``(row_ind, col_ind)`` with ``row_ind`` increasing and
    ``min(cost_matrix.shape)`` pairs. Entries equal to ``inf`` (``-inf`` when
    maximizing) are forbidden pairs. Integer matrices are solved exactly in
    64-bit arithmetic.
    """
    cost = np.asarray(cost_matrix)
    if cost.ndim != 2:
        raise ValueError(f"expected a matrix (2-D array), got a {cost.ndim}-D array")
    return _core.linear_sum_assignment(_as_native(cost), bool(maximize))
