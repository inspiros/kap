# Exact solver

`exact` solves equal-size instances with SciPy's HiGHS MILP solver and proves optimality up to its floating-point tolerances. Use it for small instances or to check heuristic costs against a proven optimum.

## Signature

The Python declarations show public defaults. The Rust declaration shows the entry point; its types are imported from `kap`.

```python
def exact(cost_matrices, *, maximize=False, time_limit=None, return_free=False,
          max_variables=150_000) -> AssignmentResult:
    ...
```

## Parameters

| Python | Rust | Default | Meaning |
|---|---|---|---|
| `cost_matrices` | - | Required | Pairwise cost matrices with equal part sizes. |
| `maximize` | - | `False` | Maximize the sum instead of minimizing it. |
| `time_limit` | - | `None` | Finite positive seconds; None sets no limit. |
| `return_free` | - | `False` | Return an empty free list on these equal-size complete assignments; otherwise None. |
| `max_variables` | - | `150_000` | Nonnegative integer guard on n**k tuple variables; booleans are rejected. |

## Returns

Python returns `AssignmentResult` with n full k-vertex cliques and their original pairwise cost sums. `free` is `[]` when requested and `None` otherwise. The Rust crate has no exact solver.

## Errors

| Exception | Condition |
|---|---|
| `ValueError` | Empty or inconsistent matrices, unequal part sizes, negative max_variables, n**k above the guard, or an invalid time limit. |
| `TypeError` | Unsupported matrix dtype or a non-integer/boolean max_variables. |
| `OverflowError` | All matrices are integer or boolean and a uint64 value exceeds int64. |
| `ImportError` | SciPy is unavailable. |
| `ExactTimeoutError` | HiGHS stops at its limit before proving optimality. |
| `RuntimeError` | HiGHS reports another failure, or its returned solution is not a complete partition. |

`ExactTimeoutError` subclasses `TimeoutError`. Its `bound` and `best_cost` attributes use the original objective sense and can be `None` if HiGHS supplied no value.


## Example

This example is Python only; the Rust crate has no exact solver.

```python
{{#include ../examples/exact.py}}
```

Python output:

```text
{{#include ../examples/exact.python.txt}}
```

## Notes and limits

- The MILP contains n**k binary tuple variables and k*n equality constraints. Building the objective and constraints allocates arrays proportional to that model.
- The practical limit documented by the API is about 100,000 tuples: roughly k=3,n≤45; k=4,n≤17; or k=5,n≤10. Solve time depends on the costs.
- The [proven-optimum measurements](benchmarks.md#ils-and-the-lower-bound-against-the-proven-optimum) include k=3,n=40 (16.6 s maximum), k=4,n=16 (28.1 s), and k=6,n=6 (29.1 s).
- HiGHS can exceed the time budget during presolve. Python model construction also occurs before the solver receives the budget.
- The objective uses float64 even for integer input. Optimality follows HiGHS tolerances; large integer magnitudes can lose precision.
- SciPy is imported when the function is called. No random seed parameter is exposed.

## See also

[Installation](installation.md#optional-python-backends) · [ILS](ils.md) · [Lower bound](lower-bound.md)
