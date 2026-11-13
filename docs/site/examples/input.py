import numpy as np
import kap

C01 = np.array([[9, 2, 7], [6, 4, 3], [5, 8, 1]], dtype=np.int64)
C02 = np.array([[3, 8], [7, 2], [6, 4]], dtype=np.int64)
C12 = np.array([[4, 9], [1, 5], [8, 2]], dtype=np.int64)
costs = [C01, C02, C12]

print("shapes =", [matrix.shape for matrix in costs])
result = kap.multiple_hungarian(costs, method="Am", return_free=True)
print("cliques =", result.cliques)
print("costs =", result.costs)
print("free =", result.free)
result = kap.multiple_hungarian(costs, method="Am", maximize=True, return_free=True)
print("maximize:")
print("cliques =", result.cliques)
print("costs =", result.costs)
print("free =", result.free)
