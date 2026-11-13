# Local search and neighborhoods

Local search improves a constructed or supplied solution using DV sweeps. Use it for deterministic refinement without perturbations or a population.

## Signature

The Python declarations show public defaults. The Rust declaration shows the entry point; its types are imported from `kap`.

<div class="lang-tabs">

```python
def local_search(cost_matrices, *, neighborhood="1dv", start=None, initial=None, max_sweeps=100,
                 maximize=False, return_free=False):
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
| `neighborhood` | `Algo::Dv1`, `Algo::DvS`, or `opts.refine` | `"1dv"` / explicit `Algo` | Select 1DV or sDV; Python accepts case-insensitive `dv1` / `dvs` aliases. |
| `start` | `algo` before `opts.refine` | `None` (Am) / explicit `Algo` | Am, Bm, Dm, or Shift. Cannot combine with Python `initial`. |
| `initial` | `cliques` in `k_assignment_from` | `None` / required by `k_assignment_from` | A complete solution of cliques; see [warm starts](warm-start.md). |
| `max_sweeps` | `opts.max_iter` | `100` | Nonnegative maximum number of DV sweeps. |
| `maximize` | `opts.sense` | `False` / `Sense::Minimize` | Maximize the original costs when true / `Sense::Maximize`. |
| `return_free` | - | `False` | Include singleton vertices in Python `free`; Rust always returns them. |

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

Python raises `ValueError` for invalid start or neighborhood names, `start` together with `initial`, or negative `max_sweeps`. Non-integer and boolean sweep limits raise `TypeError`. Rust returns `KapError::Unsupported` for sDV above 16 parts. Supplied starts can also produce the [warm-start errors](warm-start.md#errors).


## Example

<div class="lang-tabs">

```python
{{#include ../examples/local_search.py}}
```

```rust
{{#include ../../../crates/kap-solver/examples/doc_local_search.rs}}
```

</div>

Python output:

```text
{{#include ../examples/local_search.python.txt}}
```

The Rust example prints the same cliques and costs, with the same free vertices.

## Notes and limits

- 1DV reassigns one part against its complement. Each sweep uses k LAPs, or one for k=2.
- sDV visits one member of each subset/complement pair: 2^(k−1)−1 subsets per full sweep. It supports at most 16 parts.
- Each accepted move strictly improves the objective. A sweep without an improvement ends the search; zero sweeps returns the start.
- For an Am start in Rust, select `Algo::Dv1` or `Algo::DvS`. For Bm, Dm, or Shift, select that `Algo` and set `opts.refine`.
- The method is deterministic. Part sizes may differ.

## See also

[Warm starts](warm-start.md) · [ILS](ils.md) · [DV algorithm](algorithms.md#dimensionwise-variation-dv)
