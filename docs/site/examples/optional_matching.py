import numpy as np
import kap

costs = [np.array([[1.0, np.inf], [9.0, 2.0], [8.0, 7.0]])]
result = kap.ils(costs, threshold=5, max_iter=10, random_state=0, return_free=True)
print("matches =", result.matches)
print("matching_costs =", result.matching_costs)
print("free =", result.free)
