import numpy as np
import kap

cost = np.array([[8, 2, 7, 6], [6, 9, 3, 5], [4, 7, 8, 1]])
rows, cols = kap.linear_sum_assignment(cost)
print("rows =", rows.tolist())
print("columns =", cols.tolist())
print("cost =", int(cost[rows, cols].sum()))
result = kap.linear_assignment(cost, return_free=True)
print("free =", result.free)
