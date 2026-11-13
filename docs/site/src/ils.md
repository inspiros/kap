# Iterated local search

ILS perturbs a solution and applies dimensionwise variation (DV) after each perturbation. Use a time budget and several chains when solution quality matters; see the [measured comparisons](benchmarks.md#solution-quality-of-the-search-methods).

## Signature

The Python declarations show public defaults. The Rust declaration shows the entry point; its types are imported from `kap`.

<div class="lang-tabs">

```python
def ils(cost_matrices, *, max_iter=None, n_chains=1, time_limit=None, neighborhood="1dv",
        maximize=False, return_free=False, random_state=None, initial=None, threshold=None):
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
| `max_iter` | `opts.max_iter` | `None` / `100` | Nonnegative perturbations per chain. Python resolves `None` to 100, or `sys.maxsize` with a time limit. |
| `n_chains` | `opts.n_trials` | `1` | Positive number of independent chains. |
| `time_limit` | `opts.time_limit` | `None` | Python: finite positive seconds. Rust: `Option<Duration>`. Both time and iteration limits apply. |
| `neighborhood` | `opts.dv` | `"1dv"` / `Dv::One` | Use `"1dv"` or `"sdv"` / `Dv::Subsets`; aliases `"dv1"` and `"dvs"` are accepted in Python, case-insensitively. |
| `maximize` | `opts.sense` | `False` / `Sense::Minimize` | Maximize the objective when true / `Sense::Maximize`; with a threshold, use the shifted objective. |
| `threshold` | `opts.threshold` | `None` | Finite real pair-cost threshold; native core only. Optimizes sums of (cost − threshold), allowing singletons; costs remain in original units. See [optional matching](optional-matching.md). |
| `return_free` | - | `False` | Include singleton vertices in Python `free`; Rust always returns them. |
| `random_state` | `opts.seed` | `None` / `0` | Python accepts a nonnegative integer, `RandomState`, or `Generator`; Rust uses `u64`. |
| `initial` | `cliques` in `k_assignment_from` | `None` / required by `k_assignment_from` | A complete solution without a threshold; partial or empty solutions with one. See [warm starts](warm-start.md). |
| - | `opts.refine` | `None` | Optional DV pass after the search; separate from the internal `opts.dv`. |

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

Python raises `RuntimeError` if the required native core is unavailable without a threshold; with a threshold it raises `ValueError`.

Python raises `ValueError` for a negative integer seed and `TypeError` for an unsupported seed object.

Python raises `TypeError` for non-integer or boolean iteration and chain counts, and `ValueError` for counts below their permitted minimum, invalid neighborhoods, or an invalid time budget. Rust returns `KapError::BadOption("n_trials")` for zero chains and `KapError::Unsupported` for sDV above 16 parts. Invalid starts use the errors listed under [warm starts](warm-start.md#errors).


## Example

<div class="lang-tabs">

```python
{{#include ../examples/ils.py}}
```

```rust
{{#include ../../../crates/kap-solver/examples/doc_ils.rs}}
```

</div>

Python output:

```text
{{#include ../examples/ils.python.txt}}
```

The Rust example prints the same cliques and costs, with the same free vertices.

## Notes and limits

- `threshold=None` preserves unchanged behaviour. A fractional threshold promotes all-integer Python matrices to float64 for that call; integral thresholds keep exact int64 arithmetic. See [optional matching](optional-matching.md) for the shifted objective, forbidden pairs, and validation.

- Select `Algo::Ils` in Rust. The initial Am solution, or a supplied start, passes through DV before the chains begin.
- The deadline starts after the full start solution has been computed. It is checked before each perturbation; an ongoing DV descent can finish after it.
- Rust retains its default 100 perturbations even with `time_limit`; set `max_iter` explicitly, for example to `usize::MAX`, for a time budget alone.
- An integer seed and iteration limits make runs reproducible. Timed runs depend on machine speed. Chains run in parallel in the Python native core, and in Rust with `rayon`.
- After 50 unsuccessful perturbations, the current chain moves to its latest candidate. The function still returns the best solution seen.
- Each 1DV sweep uses k LAPs (one for k=2). sDV visits 2^(k−1)−1 subsets and supports at most 16 parts.

## See also

[Warm starts](warm-start.md) · [Local search](local-search.md) · [Algorithm details](algorithms.md#iterated-local-search-ils)
