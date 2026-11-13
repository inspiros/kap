"""Solve a small three-part assignment."""
import kap
import numpy as np

rng = np.random.default_rng(1)
# Condensed combinations order: (0, 1), (0, 2), (1, 2).
costs = [rng.integers(1, 20, size=(4, 4)) for _ in range(3)]
result = kap.k_assignment(costs, random_state=1)
print("Cliques:", result.cliques)
print("Clique costs:", result.costs)
print("Total cost:", result.cost)
