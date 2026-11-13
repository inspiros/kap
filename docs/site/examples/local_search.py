import numpy as np
import kap

C01 = np.array([[12, 6, 13], [12, 13, 26], [28, 12, 27]], dtype=np.int64)
C02 = np.array([[20, 14, 15], [27, 16, 14], [26, 7, 8]], dtype=np.int64)
C12 = np.array([[3, 5, 22], [10, 5, 12], [1, 20, 26]], dtype=np.int64)
costs = [C01, C02, C12]

result = kap.local_search(costs, start="Dm", neighborhood="sdv", max_sweeps=10, return_free=True)
print("cliques =", result.cliques)
print("costs =", result.costs)
print("free =", result.free)
