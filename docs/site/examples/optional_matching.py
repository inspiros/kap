import numpy as np
import kap

costs = [np.array([[1.0, np.inf], [9.0, 2.0], [8.0, 7.0]])]
result = kap.ils(costs, threshold=5, max_iter=10, random_state=0, return_free=True)
print("cliques =", result.cliques)
print("costs =", result.costs)
print("free =", result.free)
