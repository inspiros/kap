# Memetic search

Memetic search evolves a population of DV local optima by mutation and common-clique crossover. Use it to compare population search with ILS on your data.

## Signature

The Python declarations show public defaults. The Rust declaration shows the entry point; its types are imported from `kap`.

<div class="lang-tabs">

```python
def memetic(cost_matrices, *, generations=None, population=None, time_limit=None,
            neighborhood="1dv", maximize=False, return_free=False, random_state=None):
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
| `generations` | `opts.max_iter` | `None` / `100` | Nonnegative generation limit. Python resolves `None` to 100, or `sys.maxsize` with a time limit. |
| `population` | `opts.population` | `None` | At least 2 if supplied; otherwise 12 without a time limit, or sized from measured local-search time with one. |
| `time_limit` | `opts.time_limit` | `None` | Python: finite positive seconds. Rust: `Option<Duration>`. Both time and iteration limits apply. |
| `neighborhood` | `opts.dv` | `"1dv"` / `Dv::One` | Use `"1dv"` or `"sdv"` / `Dv::Subsets`; aliases `"dv1"` and `"dvs"` are accepted in Python, case-insensitively. |
| `maximize` | `opts.sense` | `False` / `Sense::Minimize` | Maximize the original costs when true / `Sense::Maximize`. |
| `return_free` | - | `False` | Include singleton vertices in Python `free`; Rust always returns them. |
| `random_state` | `opts.seed` | `None` / `0` | Python accepts a nonnegative integer, `RandomState`, or `Generator`; Rust uses `u64`. |
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
| `OverflowError` | - | All Python matrices are integer or boolean and a `uint64` value exceeds `int64`. |
| `ValueError` | `KapError::TooManyVertices` | A native part exceeds `u32` vertex capacity. |
| `ValueError` | `KapError::Lap` | A constituent LAP encounters an invalid numeric entry or an infeasible assignment. |

Python raises `RuntimeError` if the required native core is unavailable.

Python raises `ValueError` for a negative integer seed and `TypeError` for an unsupported seed object.

Python raises `TypeError` for non-integer or boolean generation counts. It raises `ValueError` for negative generations, a population that is not an integer at least 2, an invalid neighborhood, or an invalid time budget. Rust returns `KapError::BadOption("population")` for a population below 2 and `KapError::Unsupported` for sDV above 16 parts.


## Example

<div class="lang-tabs">

```python
{{#include ../examples/memetic.py}}
```

```rust
{{#include ../../../crates/kap-solver/examples/doc_memetic.rs}}
```

</div>

Python output:

```text
{{#include ../examples/memetic.python.txt}}
```

The Rust example prints the same cliques and costs, with the same free vertices.

## Notes and limits

- Select `Algo::Memetic` in Rust. `opts.n_trials` is ignored. Warm starts are not supported.
- The deadline counts from entry. An in-progress batch or generation finishes after it. Rust still applies its default 100-generation limit unless changed.
- Zero generations returns the best member of the initial population. Each candidate undergoes DV; the initialization starts from Shift.
- Integer seeds reproduce iteration-limited runs, including parallel execution. Time limits depend on machine speed.
- The measured multi-chain ILS runs had lower or equal costs at the 60 s budget. See the benchmark table for each size.

## See also

[ILS](ils.md) · [DV neighborhoods](local-search.md) · [Algorithm details](algorithms.md#memetic-algorithm)
