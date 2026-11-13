"""Exact equal-size k-assignment using SciPy's HiGHS MILP solver."""

import itertools
import math

import numpy as np

from ._reference.assignment import AssignmentResult, _validate_cost_matrices
from ._assignment import _promote_cost_matrices


class ExactTimeoutError(TimeoutError):
    """The solver stopped before proving optimality.

    Attributes:
        bound: Dual bound in the original objective sense, or None.
        best_cost: Incumbent objective in the original sense, or None.
    """

    def __init__(self, message, *, bound=None, best_cost=None):
        super().__init__(message)
        self.bound = bound
        self.best_cost = best_cost


def exact(cost_matrices, *, maximize=False, time_limit=None, return_free=False,
          max_variables=150_000) -> AssignmentResult:
    """Solve equal-size k-assignment exactly with SciPy's HiGHS MILP solver.

    Integer costs are solved exactly when n times the sum of the largest
    absolute entry in each matrix is at most 2**53; larger inputs raise
    OverflowError (rescale them). Float objectives are scaled by a power of two
    so the largest absolute tuple cost lies in [512, 1024), unless all are zero.
    Float results are optimal within HiGHS's absolute tolerances on that scaled
    problem, about 1e-9 times the largest tuple cost magnitude. Costs must be
    finite; +inf forbidden entries are not supported. The practical limit is
    about 1e5 tuples (k=3 n<=45, k=4 n<=17, k=5 n<=10). HiGHS can
    overrun the time limit during presolve on large models.

    Args:
        cost_matrices: Pairwise cost matrices in combinations order.
        maximize: Maximize the score instead of minimizing it.
        time_limit: Finite positive seconds, or None for no limit.
        return_free: Include unmatched vertices in the result.
        max_variables: Maximum permitted number of tuple variables, n**k.

    Returns:
        AssignmentResult: Cliques, their costs, the total cost, and optionally free vertices.

    Raises:
        ExactTimeoutError: HiGHS stopped at its limit before proving optimality.
    """
    matrices, k = _validate_cost_matrices(cost_matrices)
    promoted = _promote_cost_matrices(matrices)
    integer = all(m.dtype.kind in "biu" for m in matrices)
    sizes = [matrices[0].shape[0]] + [m.shape[1] for m in matrices[:k - 1]]
    if len(set(sizes)) != 1:
        raise ValueError("exact requires equal part sizes")
    n = sizes[0]
    if integer:
        # Convert extrema before abs/sum so neither int64 abs(MIN) nor the
        # bound itself can overflow. This also bounds every partial tuple sum.
        upper = n * sum(max(abs(int(m.min())), abs(int(m.max())))
                        for m in matrices if m.size)
        if upper > 2**53:
            raise OverflowError("integer costs are too large to be solved exactly through float64; "
                                "rescale the costs")
    if any(not np.isfinite(m).all() for m in promoted):
        raise ValueError("exact requires finite costs; +inf forbidden entries are not supported")
    if isinstance(max_variables, (bool, np.bool_)) or not isinstance(max_variables, (int, np.integer)):
        raise TypeError("max_variables must be a nonnegative integer")
    if max_variables < 0:
        raise ValueError("max_variables must be a nonnegative integer")
    nv = n ** k
    if nv > max_variables:
        raise ValueError(f"n**k = {nv} exceeds max_variables = {max_variables}")
    if time_limit is not None:
        try:
            valid = bool(np.isfinite(time_limit)) and time_limit > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("time_limit must be finite and positive")
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csr_matrix
    except ImportError as exc:
        raise ImportError("kap.exact requires SciPy; install it with 'pip install scipy'") from exc
    if n == 0:
        return AssignmentResult([], [], [] if return_free else None)

    pairs = list(itertools.combinations(range(k), 2))
    indices = np.indices((n,) * k).reshape(k, -1)
    objective = np.zeros(nv, dtype=np.float64)
    for matrix, (p, q) in zip(promoted, pairs):
        objective += np.asarray(matrix[indices[p], indices[q]], dtype=np.float64)
    if not np.isfinite(objective).all():
        raise ValueError("exact requires finite tuple costs; rescale the costs")
    scale_exponent = 0
    if not integer:
        largest = float(np.max(np.abs(objective)))
        if largest:
            # ldexp avoids materializing an overflowing scale for subnormals.
            scale_exponent = 10 - math.frexp(largest)[1]
            objective = np.ldexp(objective, scale_exponent)
    if maximize:
        objective = -objective
    rows = np.concatenate([p * n + indices[p] for p in range(k)])
    cols = np.tile(np.arange(nv), k)
    constraints = csr_matrix((np.ones(k * nv), (rows, cols)), shape=(k * n, nv))
    options = {"disp": False, "mip_rel_gap": 0}
    if time_limit is not None:
        options["time_limit"] = time_limit
    result = milp(objective, integrality=np.ones(nv), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(constraints, 1, 1), options=options)
    if result.status == 1:
        def restore(value):
            if value is None:
                return None
            original = math.ldexp(value, -scale_exponent)
            return -original if maximize else original
        raise ExactTimeoutError(result.message, bound=restore(getattr(result, "mip_dual_bound", None)),
                                best_cost=restore(getattr(result, "fun", None)))
    if result.status != 0:
        raise RuntimeError(result.message)
    x = getattr(result, "x", None)
    if x is None or np.shape(x) != (nv,):
        raise RuntimeError("HiGHS returned an invalid optimal solution")
    selected = indices[:, np.asarray(x) > 0.5].T
    if len(selected) != n or any(not np.array_equal(np.sort(selected[:, p]), np.arange(n))
                                 for p in range(k)):
        raise RuntimeError("HiGHS returned a solution that is not a perfect partition")
    matches, costs = [], []
    for vertices in selected:
        matches.append([(p, int(vertex)) for p, vertex in enumerate(vertices)])
        costs.append(sum(matrix[vertices[p], vertices[q]].item()
                         for matrix, (p, q) in zip(matrices, pairs)))
    return AssignmentResult(matches, costs, [] if return_free else None)
