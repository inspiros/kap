# Dispatcher

`k_assignment` selects a k-assignment method and optionally applies DV refinement. Use the explicit Python functions when their named parameters describe your task more directly.

## Signature

The Python declarations show public defaults. The Rust declaration shows the entry point; its types are imported from `kap`.

<div class="lang-tabs">

```python
def k_assignment(cost_matrices: Sequence[Array2DLike], algo: Optional[str] = None,
                 max_iter: Optional[int] = None, n_trials: int = 1, maximize: bool = False,
                 return_free: bool = False, backend: str = "kap",
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
                 refine: Optional[str] = None, time_limit: Optional[float] = None,
                 population: Optional[int] = None, *, threshold=None) -> AssignmentResult:
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
| `algo` | `algo` | `None` / required | Python: ILS with native kap, otherwise Cm. Rust always requires an `Algo`. |
| `max_iter` | `opts.max_iter` | `None` / `100` | Moves, sweeps, perturbations per chain, or generations; Python resolves `None` by algorithm and time mode. |
| `n_trials` | `opts.n_trials` | `1` | Em trials or ILS chains; positive for those methods. |
| `maximize` | `opts.sense` | `False` / `Sense::Minimize` | Maximize the objective when true / `Sense::Maximize`; with a threshold, use the shifted objective. |
| `threshold` | `opts.threshold` | `None` | Finite real pair-cost threshold; native core only. Optimizes sums of (cost − threshold), allowing singletons; costs remain in original units. See [optional matching](optional-matching.md). |
| `return_free` | - | `False` | Include singleton vertices in Python `free`; Rust always returns them. |
| `backend` | - | `"kap"` | Native kap or optional scipy, lap, lapjv, munkres; newer methods require native kap. |
| `random_state` | `opts.seed` | `None` / `0` | Python accepts a nonnegative integer, `RandomState`, or `Generator`; Rust uses `u64`. |
| `refine` | `opts.refine` or `opts.dv` | `None` / `None` or `Dv::One` | 1DV or sDV after construction; inside ILS/memetic it selects the search neighborhood. |
| `time_limit` | `opts.time_limit` | `None` | Python: finite positive seconds. Rust: `Option<Duration>`. Both time and iteration limits apply. |
| `population` | `opts.population` | `None` | Memetic population size; supplied values must be integers at least 2. |

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

Python raises `ValueError` for an unknown algorithm or neighborhood, `time_limit` outside ILS/memetic, `population` outside memetic, or a newer method/refinement without native kap. Selected methods also apply their own [ILS](ils.md#errors), [memetic](memetic.md#errors), [local-search](local-search.md#errors), and [multiple-Hungarian](multiple-hungarian.md#errors) checks. Rust uses `KapError::BadOption` for zero Em/Fm moves, zero Em/ILS trials, or a population below 2; sDV above 16 parts produces `KapError::Unsupported`.


## Example

<div class="lang-tabs">

```python
{{#include ../examples/dispatcher.py}}
```

```rust
{{#include ../../../crates/kap-solver/examples/doc_dispatcher.rs}}
```

</div>

Python output:

```text
{{#include ../examples/dispatcher.python.txt}}
```

The Rust example prints the same cliques and costs, with the same free vertices.

## Notes and limits

- `threshold=None` preserves unchanged behaviour. A fractional threshold promotes all-integer Python matrices to float64 for that call; integral thresholds keep exact int64 arithmetic. See [optional matching](optional-matching.md) for the shifted objective, forbidden pairs, and validation.

- Python accepts Am–Fm, Shift, 1DV, sDV, ILS, and memetic, case-insensitively. `initial` is available through `ils` and `local_search`, not this dispatcher.
- Python defaults to 100 iterations, or an effectively unlimited count for timed ILS/memetic. Explicit iteration and time limits both apply; Rust always retains the supplied `max_iter`.
- Negative dispatcher limits act as zero, except that Em and Fm require positive limits. Explicit Python method functions reject negative search limits.
- Seeded iteration-limited randomized methods are reproducible. Time-limited runs depend on machine speed.
- Choose `Algo::Ils` and `Options::dv` for ILS in Rust. `Options::refine` applies a separate pass after the main Rust algorithm.

## See also

[ILS](ils.md) · [Memetic](memetic.md) · [Local search](local-search.md) · [Multiple Hungarian methods](multiple-hungarian.md)
