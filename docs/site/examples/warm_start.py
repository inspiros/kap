import numpy as np
import kap

C01 = np.array([[28, 25, 18], [1, 12, 7], [3, 4, 14]], dtype=np.int64)
C02 = np.array([[21, 18, 10], [12, 17, 24], [2, 17, 9]], dtype=np.int64)
C12 = np.array([[8, 28, 25], [27, 8, 11], [2, 20, 3]], dtype=np.int64)
costs = [C01, C02, C12]

initial = [[(0, i), (1, i), (2, i)] for i in range(3)]
result = kap.local_search(costs, initial=initial, return_free=True)
print("matches =", result.matches)
print("matching_costs =", result.matching_costs)
print("free =", result.free)
