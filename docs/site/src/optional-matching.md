# Optional matching

A threshold lets vertices stay unmatched when joining a clique does not improve the shifted objective. It is available on all native heuristic methods, including DV refinement and supported warm starts.

## Signature

The Python declaration shows public defaults. Rust selects a method with `Algo` and sets `Options::threshold`.

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
| `cost_matrices` | `mats` | Required | Pairwise matrices in [combinations order](input.md#matrix-order). Part sizes may differ. |
| `threshold` | `opts.threshold` | `None` | Finite real pair-cost threshold; Python rejects booleans and non-numbers. |
| `maximize` | `opts.sense` | `False` / `Sense::Minimize` | Maximise the shifted objective when true / `Sense::Maximize`. |
| `initial` | `cliques` in `k_assignment_from` | `None` / required by `k_assignment_from` | For ILS and local search, partial and empty solutions are accepted with a threshold. |
| `return_free` | - | `False` | Include singleton vertices in Python `free`; Rust always returns them. |
| `max_iter` | `opts.max_iter` | `None` / `100` | ILS perturbations per chain; Python resolves None to 100, or sys.maxsize with a time limit. |
| `n_chains` | `opts.n_trials` | `1` | Positive number of ILS chains. |
| `time_limit` | `opts.time_limit` | `None` | Finite positive seconds in Python; `Option<Duration>` in Rust. Both iteration and time limits apply. |
| `neighborhood` | `opts.dv` | `"1dv"` / `Dv::One` | ILS DV neighborhood: 1DV or sDV. |
| `random_state` | `opts.seed` | `None` / `0` | Integer seeds reproduce runs with iteration limits. |

`threshold` is keyword-only on Python `ils`, `memetic`, `local_search`, `multiple_hungarian`, and `k_assignment`.

## Semantics

With `threshold=tau`, the objective is the sum of (cost − tau) over all pairs sharing a clique, over all partitions into cliques with at most one vertex per part, singletons allowed; minimised (maximised with `maximize=True`). A clique forms only if it improves that objective, so vertices can stay unmatched. `threshold=None` leaves the behaviour unchanged.

This is a soft gate: a pair costlier than tau can share a clique when the clique's other pairs pay for it. For example, costs 1, 1, and 12 with tau=10 give a triangle a shifted cost of −16, even though one pair costs more than tau. These are heuristics for k>2; they do not guarantee the globally optimal partition. For k=2 they solve optional bipartite matching optimally.

A float cost of `+inf` (minimise) / `-inf` (maximise) forbids a pair. To forbid every pair above a limit, set those entries to `inf` before calling. For maximisation, use `-inf` to forbid pairs, for example weights below a limit. NaN and the other infinity are errors.

Returned costs are in original units, not shifted. Each reported clique cost is the sum of its original pair costs. Singletons are omitted from `matches` and appear in `free` when `return_free=True`.

For all-integer Python matrices, a fractional threshold promotes the matrices to float64 for that call; integral thresholds keep exact int64 arithmetic. This includes integral float thresholds such as `2.0`. Float64 promotion can round large integer costs. Thresholds pass to Rust as float64; Python rejects integral thresholds that cannot be represented exactly as float64. Rust integer matrices require an integral threshold representable in the accumulator type.

With a threshold, `initial` on `ils` or `local_search` may be partial or empty. Unlisted vertices become singletons. Index, uniqueness, and one-vertex-per-part checks still apply, and an initial clique containing a forbidden pair is rejected. Improvement of a supplied start refers to the shifted objective, up to rounding for float costs. Without a threshold, the initial solution must remain complete; see [warm starts](warm-start.md).

Dm ranks intermediate merges by the shifted objective, so a large threshold does not reproduce ungated Dm.

## Returns

Python returns `AssignmentResult(matches, matching_costs, free)`. Rust returns `KSolution` with `cliques`, `costs`, total `cost`, and `free`. Costs use original units in both languages.

## Errors

| Python | Rust | Condition |
|---|---|---|
| `TypeError` | Rejected by the `Option<f64>` type | Boolean or non-real Python threshold. |
| `ValueError` | `KapError::BadOption("threshold")` | Non-finite threshold, or an unrepresentable threshold for integer arithmetic. |
| `ValueError` | `KapError::Lap` | NaN entry or wrong-sign infinity. |
| `ValueError` | `KapError::ForbiddenInitial` | `initial` places a forbidden pair in one clique. |
| `ValueError` | `KapError::Unsupported("cost overflow")` | Checked cost arithmetic overflows. |
| `ValueError` | - | Threshold supplied to the reference backend, another LAP backend, or without the native core. |

Matrix shape, dtype, and other method-specific checks still apply.

## Example

The third vertex of part 0 stays unmatched. The infinite entry forbids vertices `(0, 0)` and `(1, 1)` from sharing a clique.

<div class="lang-tabs">

```python
{{#include ../examples/optional_matching.py}}
```

```rust
{{#include ../../../crates/kap-solver/examples/doc_optional_matching.rs}}
```

</div>

Python output:

```text
{{#include ../examples/optional_matching.python.txt}}
```

The Rust example prints identical output.

## Notes and limits

- `exact` and `lower_bound` do not take a threshold.
- Thresholds require the native core. Python reference and other LAP backends reject them with `ValueError`.
- Shifted integer costs must lie strictly between −2⁶¹ and 2⁶¹, as for costs without a threshold; larger values are rejected as invalid entries.
- Float costs near the float64 limit (about 1e308) can overflow when summed; such sums are not detected in every path.
- Infinite thresholds are rejected. Use `None` for the unchanged objective.
- Thresholding and forbidden pairs also apply during refinement and search.

## See also

[Warm starts](warm-start.md) · [ILS](ils.md) · [Local search](local-search.md) · [Multiple Hungarian methods](multiple-hungarian.md)
