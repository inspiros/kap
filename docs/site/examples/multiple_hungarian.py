import numpy as np
import kap

C01 = np.array([[4, 12, 15], [18, 20, 15], [2, 21, 28]], dtype=np.int64)
C02 = np.array([[19, 25, 26], [18, 18, 4], [16, 23, 23]], dtype=np.int64)
C12 = np.array([[24, 7, 24], [4, 11, 3], [11, 14, 2]], dtype=np.int64)
costs = [C01, C02, C12]

result = kap.multiple_hungarian(costs, method="Bm", return_free=True)
print("matches =", result.matches)
print("matching_costs =", result.matching_costs)
print("free =", result.free)
