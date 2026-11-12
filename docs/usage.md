# Usage

## Linear assignment

`kap.linear_assignment` gives every backend the same interface.
`kap.linear_sum_assignment` calls the native solver directly, with the same
signature and results as `scipy.optimize.linear_sum_assignment`.

```python
import numpy as np
from kap import linear_assignment

cost = np.array([[9, 6, 4],
                 [4, 4, 6],
                 [3, 9, 2]])
res = linear_assignment(cost, return_free=True)  # backend="kap" by default
res.matches         # array of (row, col) pairs
res.matching_costs  # cost of each matched pair
res.free            # unmatched vertices as (0, row) or (1, col)
```

Arguments:

| Argument | Meaning |
|---|---|
| `cost_matrix` | 2D array or nested list (rectangular allowed, except with `lapjv`) |
| `maximize` | maximize total weight instead of minimizing cost |
| `return_free` | also return the unmatched vertices |
| `backend` | `"kap"` (default, native), `"scipy"`, `"lap"`, `"lapjv"` or `"munkres"` |

## k-assignment

`kap.k_assignment` takes the k(k-1)/2 pairwise cost matrices, ordered as
`itertools.combinations(range(k), 2)`. For k = 4 that is
`[C01, C02, C03, C12, C13, C23]`, where `Cij` has shape `(n_i, n_j)`.
Partite sizes may differ.

```python
from kap import k_assignment

res = k_assignment(cost_matrices, algo="Cm")
res.matches         # list of matches, each a list of (partite, node) tuples
res.matching_costs  # cost of each match: sum over all pairs of its members
res.free            # with return_free=True, the unmatched vertices
```

A match need not contain a vertex from every partite: when partites have
different sizes, some matches are incomplete.

| Argument | Meaning |
|---|---|
| `algo` | `"Am"` to `"Fm"`, see [Algorithms](algorithms.md). Default `"Cm"` |
| `max_iter` | iteration cap for the local-search methods (Cm, Em, Fm) |
| `n_trials` | number of restarts for Em |
| `maximize` | maximize instead of minimize |
| `return_free` | also return unmatched vertices |
| `backend` | LAP backend, as above |
| `random_state` | seed or `numpy.random.RandomState` for Em and Fm |
