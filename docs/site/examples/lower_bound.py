import numpy as np
import kap

C01 = np.array([[19, 22, 6], [27, 16, 22], [16, 26, 10]], dtype=np.int64)
C02 = np.array([[7, 10, 22], [15, 19, 26], [9, 27, 9]], dtype=np.int64)
C12 = np.array([[6, 25, 5], [7, 3, 12], [19, 13, 2]], dtype=np.int64)
costs = [C01, C02, C12]

result = kap.local_search(costs)
cost = result.cost
bound = kap.lower_bound(costs, max_iter=30)
print("cost =", cost)
print("bound =", bound)
print(f"gap = {(cost - bound) / cost:.3f}")
