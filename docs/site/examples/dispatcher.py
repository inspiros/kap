import numpy as np
import kap

C01 = np.array([[6, 10, 12], [1, 29, 2], [28, 29, 28]], dtype=np.int64)
C02 = np.array([[26, 29, 16], [7, 29, 26], [29, 2, 12]], dtype=np.int64)
C12 = np.array([[15, 10, 28], [20, 17, 1], [13, 17, 13]], dtype=np.int64)
costs = [C01, C02, C12]

result = kap.k_assignment(costs, algo="Shift", refine="1dv", maximize=True, return_free=True)
print("cliques =", result.cliques)
print("costs =", result.costs)
print("free =", result.free)
