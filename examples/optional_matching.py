"""Leave vertices free when their pair costs exceed the threshold."""
import kap
import numpy as np

# At threshold 5, only the two diagonal pairs improve the shifted objective.
costs = [np.array([[1, 9, 9], [9, 2, 9], [9, 9, 8]])]
result = kap.ils(costs, threshold=5, return_free=True, random_state=4)
print("Cost matrix:\n", costs[0])
print("Threshold: 5")
print("Cliques:", result.cliques)
print("Free vertices:", result.free)
