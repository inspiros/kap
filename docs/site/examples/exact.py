import numpy as np
import kap

C01 = np.array([[24, 22, 24], [22, 6, 16], [28, 15, 27]], dtype=np.int64)
C02 = np.array([[16, 7, 20], [1, 3, 9], [28, 14, 27]], dtype=np.int64)
C12 = np.array([[16, 7, 8], [24, 18, 6], [8, 4, 29]], dtype=np.int64)
costs = [C01, C02, C12]

result = kap.exact(costs, time_limit=10, return_free=True)
print("matches =", result.matches)
print("matching_costs =", result.matching_costs)
print("free =", result.free)
