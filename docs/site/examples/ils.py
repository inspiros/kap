import numpy as np
import kap

C01 = np.array([[18, 8, 29], [28, 2, 6], [6, 6, 17]], dtype=np.int64)
C02 = np.array([[11, 14, 7], [28, 20, 20], [4, 5, 26]], dtype=np.int64)
C12 = np.array([[10, 25, 23], [1, 14, 16], [14, 4, 21]], dtype=np.int64)
costs = [C01, C02, C12]

result = kap.ils(costs, max_iter=20, n_chains=2, random_state=0, return_free=True)
print("matches =", result.matches)
print("matching_costs =", result.matching_costs)
print("free =", result.free)
