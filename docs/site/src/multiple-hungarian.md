# Multiple Hungarian methods

These methods build cliques by solving pairwise LAPs and merging parts. Use Am–Fm to reproduce the original paper's methods, or use Am, Dm, or Shift for bounded construction work.

## Signature

The Python declarations show public defaults. The Rust declaration shows the entry point; its types are imported from `kap`.

<div class="lang-tabs">

```python
def multiple_hungarian(cost_matrices, method="Cm", *, max_iter=100, n_trials=1,
                       maximize=False, return_free=False, backend="kap", random_state=None,
                       threshold=None):
    ...
```

```rust
pub fn k_assignment<T: Cost>(
    mats: &[MatrixView<'_, T>],
    algo: Algo,
    opts: &Options,
) -> Result<KSolution<T::Acc>, KapError>
where
    T::Acc: Cost<Acc = T::Acc>;
```

</div>

## Parameters

| Python | Rust | Default | Meaning |
|---|---|---|---|
| `cost_matrices` | `mats` | Required | Pairwise matrices in [combinations order](input.md#matrix-order). |
| `method` | `algo` | `"Cm"` / explicit `Algo` | Am, Bm, Cm, Dm, Em, Fm, or Shift; Python names are case-insensitive. |
| `max_iter` | `opts.max_iter` | `100` | Cm/Em/Fm move limit; positive for Em/Fm. Ignored by Am/Bm/Dm/Shift. |
| `n_trials` | `opts.n_trials` | `1` | Positive independent Em trial count; ignored by other methods. |
| `maximize` | `opts.sense` | `False` / `Sense::Minimize` | Maximize the objective when true / `Sense::Maximize`; with a threshold, use the shifted objective. |
| `threshold` | `opts.threshold` | `None` | Finite real pair-cost threshold; native core only. Optimizes sums of (cost − threshold), allowing singletons; costs remain in original units. See [optional matching](optional-matching.md). |
| `return_free` | - | `False` | Include singleton vertices in Python `free`; Rust always returns them. |
| `backend` | - | `"kap"` | LAP backend: kap, scipy, lap, lapjv, or munkres. Rust uses its native LAP solver. |
| `random_state` | `opts.seed` | `None` / `0` | Python accepts a nonnegative integer, `RandomState`, or `Generator`; Rust uses `u64`. |

## Returns

Python returns `AssignmentResult(matches, matching_costs, free)`. Each match is a clique of `(part, vertex)` pairs. Its cost is the sum of its pairwise edges. `free` is `None` unless requested; it then lists singleton vertices.

Rust returns `Result<KSolution<T::Acc>, KapError>`. `cliques` and `costs` correspond to the Python fields. `cost` is their sum, and `free` always lists singletons. Singleton vertices are excluded from both languages' clique lists.

## Errors

| Python | Rust | Condition |
|---|---|---|
| `ValueError` | `KapError::BadCount` | The matrix count does not describe 2 through 64 parts in the native core. |
| `ValueError` | `KapError::ShapeMismatch` | Matrix dimensions disagree about a part's size. |
| `TypeError` | Rejected by the `Cost` trait at compilation | A matrix has an unsupported element type. |
| `OverflowError` | - | Python uses int64 arithmetic and a `uint64` value exceeds `int64`. |
| `ValueError` | `KapError::TooManyVertices` | A native part exceeds `u32` vertex capacity. |
| `ValueError` | `KapError::Lap` | A constituent LAP encounters an invalid numeric entry or an infeasible assignment. |

Python raises `ValueError` for a negative integer seed and `TypeError` for an unsupported seed object.

Python raises `ValueError` for an invalid method or backend, nonpositive Em/Fm move limits, or nonpositive Em trials. Shift raises `ValueError` without the native core and `backend="kap"`. Missing optional backends raise `ModuleNotFoundError`. Rust returns `KapError::BadOption("max_iter")` for zero Em/Fm moves and `KapError::BadOption("n_trials")` for zero Em trials.


## Example

<div class="lang-tabs">

```python
{{#include ../examples/multiple_hungarian.py}}
```

```rust
{{#include ../../../crates/kap-solver/examples/doc_multiple_hungarian.rs}}
```

</div>

Python output:

```text
{{#include ../examples/multiple_hungarian.python.txt}}
```

The Rust example prints the same cliques and costs, with the same free vertices.

## Notes and limits

- `threshold=None` preserves unchanged behaviour. All-integer Python matrices require an integral threshold and keep exact int64 arithmetic. For a fractional threshold, explicitly convert the matrices to float64; otherwise the call raises `ValueError`. See [optional matching](optional-matching.md) for the shifted objective, forbidden pairs, and validation.

- Am uses k−1 LAPs, and Shift uses k(k−1). Dm chooses the cheapest immediate merge. Bm explores merge trees and memoizes commuting merges.
- Native Bm uses 6, 39, 310, and 3,000 LAPs for k=3, 4, 5, and 6. Cm, Em, and Fm invoke Bm for neighbors, so their cost grows further.
- Cm follows steepest descent. Em runs randomized first-improvement trials. Fm permits sideways or worse moves and returns its best solution.
- Without the native core, Am–Fm use the Python implementation. Other LAP backends also select it; its Cm and Fm behavior differs from the native implementation.
- Integer seeds reproduce Em/Fm iteration-limited runs. The random sequences differ from kap 0.2.
- Part sizes may differ. Prefer Am or Dm over Bm-based methods when k exceeds about 6.

## See also

[Algorithm strategies and LAP counts](algorithms.md) · [Backend measurements](benchmarks.md#lap-backends) · [ILS](ils.md)
