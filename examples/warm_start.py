"""Improve a multiple Hungarian solution with a warm-started ILS."""
import kap
import numpy as np

rng = np.random.default_rng(5)
costs = [rng.integers(1, 100, size=(8, 8)) for _ in range(3)]
initial = kap.multiple_hungarian(costs, method="Am")
result = kap.ils(costs, initial=initial, max_iter=20, random_state=5)
print("Multiple Hungarian total:", initial.cost)
print("Warm-started ILS total:", result.cost)
