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

The functions accept k(k-1)/2 pairwise cost matrices, ordered as
`itertools.combinations(range(k), 2)`. For k = 4 that is
`[C01, C02, C03, C12, C13, C23]`, with `Cij` shaped `(n_i, n_j)`.
Partite sizes may differ; matches can be incomplete.

| Function | When to use it |
|---|---|
| `kap.ils` | Default search; for the best quality in a time budget, set `time_limit` and `n_chains` to the number of CPU cores. |
| `kap.memetic` | Evolve a population of locally optimized solutions. |
| `kap.local_search` | Fast deterministic DV polishing from Am, Bm, Dm or Shift. |
| `kap.multiple_hungarian` | Use the paper's Am–Fm methods or Shift, with alternative LAP backends for Am–Fm. |

See [Algorithms](algorithms.md) and the [benchmarks](benchmarks.md#solution-quality-of-the-search-methods).
ILS, memetic and local search require the native core. Integer seeds make
iteration-limited randomized runs reproducible; time limits depend on machine speed.

`kap.k_assignment` dispatches to these methods and retains the original interface:

```python
from kap import k_assignment

res = k_assignment(cost_matrices)  # algo="ils" by default with the native core
res.matches         # list of matches, each a list of (partite, node) tuples
res.matching_costs  # cost of each match: sum over all pairs of its members
res.free            # with return_free=True, the unmatched vertices
```

| Argument | Meaning |
|---|---|
| `algo` | `"Am"` to `"Fm"`, `"Shift"`, `"1DV"`, `"sDV"`, `"ils"` or `"memetic"`; default `"ils"` with the native core and `backend="kap"`, else `"Cm"` |
| `refine` | Apply `"1DV"` or `"sDV"` after the method, or select the neighborhood inside ILS or memetic; native-only |
| `max_iter` | Move, sweep, perturbation or generation limit depending on the method. Default 100, or unlimited for ILS and memetic when `time_limit` is set; an explicit limit applies alongside the time limit |
| `n_trials` | Independent Em trials or ILS chains |
| `maximize` | Maximize instead of minimize |
| `return_free` | Also return unmatched vertices |
| `backend` | LAP backend, as above |
| `random_state` | Integer seed, `numpy.random.RandomState`, or `Generator` |
| `time_limit` | Finite positive seconds for native ILS or memetic; applies alongside an explicit `max_iter` |
| `population` | Memetic population size, at least 2; native-only |

## Exact k-assignment

`kap.exact(cost_matrices, time_limit=60, return_free=True)` solves equal-size
instances using SciPy's HiGHS MILP solver (`pip install scipy`). It proves
optimality up to HiGHS floating-point tolerances. The default `max_variables`
guard limits the model to 150,000 tuples; practical limits are about 100,000.
A time limit may be exceeded during presolve. If optimality is not proven,
`kap.ExactTimeoutError` exposes `bound` and `best_cost` when available.
`maximize=True` maximizes the original costs.
