import numpy as np
import kap

C01 = np.array([[12, 10, 22], [12, 23, 17], [4, 15, 12]], dtype=np.int64)
C02 = np.array([[17, 17, 17], [17, 26, 10], [3, 20, 22]], dtype=np.int64)
C12 = np.array([[14, 24, 9], [21, 20, 12], [23, 28, 8]], dtype=np.int64)
costs = [C01, C02, C12]

result = kap.memetic(costs, generations=4, population=4, random_state=0, return_free=True)
print("cliques =", result.cliques)
print("costs =", result.costs)
print("free =", result.free)
