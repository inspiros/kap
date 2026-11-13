import numpy as np
import kap

costs = [np.array([[4, 1], [2, 3]]) for _ in range(3)]
result = kap.ils(costs, max_iter=10, random_state=0)
print(result.cliques, result.costs)
