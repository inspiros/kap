"""Native bounds on the optimal equal-size k-assignment cost."""

from ._assignment import _integer, _promote_cost_matrices
from ._reference.assignment import _validate_cost_matrices

__all__ = ["lower_bound"]


def lower_bound(cost_matrices, *, maximize=False, max_iter=300) -> int | float:
    """Bound the optimal total cost of an equal-size k-assignment problem.

    Returns a lower bound when minimising, or an upper bound when maximising.
    For k=2 this is the exact LAP optimum. For k=3 it is an axial Lagrangian
    bound, which equals at best the LP relaxation bound. For k>=4 it averages
    bounds over triples (scaled by 1/(k-2)); this is loose for non-metric costs.
    Integer costs give a rigorous bound. Float bounds are computed in floating
    point and can exceed the true minimum (or fall below the true maximum) by
    rounding error on the order of machine epsilon times the sum of the cost
    magnitudes involved. This error can be large when costs of opposite sign
    nearly cancel. All part sizes must be equal. The native core is required.

    Args:
        cost_matrices: Pairwise cost matrices in combinations order, as for ils.
            Integer and boolean matrices promote to int64; otherwise float64.
        maximize: Maximise the score instead of minimising it.
        max_iter: Nonnegative integer subgradient updates per triple (default 300).
            Zero still evaluates the initial multiplier seeds.

    Returns:
        int or float: A bound on the optimum, matching the promoted cost dtype.

    Typical use for a heuristic solution with positive total cost::

        cost = sum(kap.ils(cost_matrices).matching_costs)
        bound = kap.lower_bound(cost_matrices)
        gap = (cost - bound) / cost
    """
    iterations = _integer(max_iter, "max_iter", 0)
    try:
        from ._core import lower_bound as native
    except ImportError as exc:
        raise RuntimeError("the native core is required") from exc
    matrices, _ = _validate_cost_matrices(cost_matrices)
    return native(_promote_cost_matrices(matrices), maximize=bool(maximize), max_iter=iterations)
