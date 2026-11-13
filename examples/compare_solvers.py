"""Compare heuristic costs with a lower bound and, optionally, the optimum."""
import kap
import numpy as np

# Four parts of 7 vertices: 6 condensed matrices (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
rng = np.random.default_rng(3)
costs = [rng.integers(1, 100, size=(7, 7)) for _ in range(6)]
bound = kap.lower_bound(costs)
results = [
    ("Am", kap.multiple_hungarian(costs, method="Am")),
    ("Cm", kap.multiple_hungarian(costs, method="Cm")),
    ("Local search", kap.local_search(costs)),
    ("ILS", kap.ils(costs, max_iter=20, random_state=3)),
    ("Memetic", kap.memetic(costs, generations=10, population=8, random_state=3)),
]
try:
    import scipy.optimize  # noqa: F401
except ImportError:
    print("SciPy unavailable; skipping exact solver")
else:
    results.append(("Exact (optimum)", kap.exact(costs)))

print(f"Lower bound: {bound}")
print(f"{'Solver':<18} {'Cost':>8} {'Gap to bound':>14}")
for name, result in results:
    gap = (result.cost - bound) / bound
    print(f"{name:<18} {result.cost:>8} {gap:>14.1%}")
print("The bound is not tight: a gap to it overstates the distance from the optimum.")
