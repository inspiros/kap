"""Solve a rectangular linear assignment, minimizing and maximizing."""
import kap
import numpy as np

rng = np.random.default_rng(2)
costs = rng.integers(1, 20, size=(3, 5))
print("Cost matrix:\n", costs)
rows, cols = kap.linear_sum_assignment(costs)
print("Minimum:", list(zip(rows.tolist(), cols.tolist())), "total:", costs[rows, cols].sum())
try:
    from scipy.optimize import linear_sum_assignment as scipy_lap
except ImportError:
    print("SciPy comparison unavailable")
else:
    scipy_rows, scipy_cols = scipy_lap(costs)
    print("SciPy minimum total:", costs[scipy_rows, scipy_cols].sum())
rows, cols = kap.linear_sum_assignment(costs, maximize=True)
print("Maximum:", list(zip(rows.tolist(), cols.tolist())), "total:", costs[rows, cols].sum())
