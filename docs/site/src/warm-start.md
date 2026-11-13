# Warm starts

A warm start supplies a solution for DV refinement or ILS. Without a threshold it must be complete; with a threshold it may be partial or empty. Use it to continue from an earlier result or improve a solution produced outside kap.

## Signature

The Python declarations show public defaults. The Rust declaration shows the entry point; its types are imported from `kap`.

<div class="lang-tabs">

```python
def local_search(cost_matrices, *, neighborhood="1dv", start=None, initial=None, max_sweeps=100,
                 maximize=False, return_free=False, threshold=None):
    ...

def ils(cost_matrices, *, max_iter=None, n_chains=1, time_limit=None, neighborhood="1dv",
        maximize=False, return_free=False, random_state=None, initial=None, threshold=None):
    ...
```

```rust
pub fn k_assignment_from<T: Cost>(
    mats: &[MatrixView<'_, T>],
    cliques: &[Vec<(usize, usize)>],
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
| `neighborhood` | `algo` / `opts.dv` / `opts.refine` | `"1dv"` / `Dv::One` internally | Select the DV neighborhood for refinement or ILS; sDV needs k≤16. |
| `start` | - | `None` | Python local_search rejects a non-None start together with initial. |
| `initial` | `cliques` in `k_assignment_from` | `None` / required by `k_assignment_from` | A complete solution without a threshold; partial or empty solutions with one. See [warm starts](warm-start.md). |
| `max_sweeps` | `opts.max_iter` for DV | `100` | Local-search sweep limit; nonnegative. |
| `max_iter` | `opts.max_iter` for ILS | `None` / `100` | ILS perturbations per chain; Python resolves None to 100 or sys.maxsize with a time limit. |
| `n_chains` | `opts.n_trials` | `1` | Positive ILS chain count. |
| `time_limit` | `opts.time_limit` | `None` | Python: finite positive seconds. Rust: `Option<Duration>`. Both time and iteration limits apply. |
| `maximize` | `opts.sense` | `False` / `Sense::Minimize` | Maximize the objective when true / `Sense::Maximize`; with a threshold, use the shifted objective. |
| `threshold` | `opts.threshold` | `None` | Finite real pair-cost threshold; native core only. Optimizes sums of (cost − threshold), allowing singletons; costs remain in original units. See [optional matching](optional-matching.md). |
| `return_free` | - | `False` | Include singleton vertices in Python `free`; Rust always returns them. |
| `random_state` | `opts.seed` | `None` / `0` | Python accepts a nonnegative integer, `RandomState`, or `Generator`; Rust uses `u64`. |
| - | `algo` | Required | Am, Bm, Dm, Shift, Dv1, DvS, or Ils; the supplied start replaces construction. |

## Returns

Python returns `AssignmentResult(matches, matching_costs, free)`. Each match is a clique of `(part, vertex)` pairs. Its cost is the sum of its pairwise edges. `free` is `None` unless requested; it then lists singleton vertices.

Rust returns `Result<KSolution<T::Acc>, KapError>`. `cliques` and `costs` correspond to the Python fields. `cost` is their sum, and `free` always lists singletons. Singleton vertices are excluded from both languages' clique lists.

DV and ILS return a solution whose objective is no worse than the supplied start. With a threshold, this refers to the shifted objective. This guarantee is exact for integer costs; for float costs it holds up to floating-point rounding of the cost sums. Passing Am, Bm, Dm, or Shift to Rust `k_assignment_from` returns the supplied start unless `opts.refine` is set.

## Errors

| Python | Rust | Condition |
|---|---|---|
| `ValueError` | `KapError::BadCount` | The matrix count does not describe 2 through 64 parts in the native core. |
| `ValueError` | `KapError::ShapeMismatch` | Matrix dimensions disagree about a part's size. |
| `TypeError` | Rejected by the `Cost` trait at compilation | A matrix has an unsupported element type. |
| `OverflowError` | - | Python uses int64 arithmetic and a `uint64` value exceeds `int64`. |
| `ValueError` | `KapError::TooManyVertices` | A native part exceeds `u32` vertex capacity. |
| `ValueError` | `KapError::Lap` | A constituent LAP encounters an invalid numeric entry or an infeasible assignment. |
| `ValueError` | `KapError::Unsupported("cost overflow")` | An integer clique cost or total overflows while evaluating the supplied start. |

Python raises `RuntimeError` if the required native core is unavailable without a threshold; with a threshold it raises `ValueError`.

| Python | Rust | Condition |
|---|---|---|
| `ValueError` | `KapError::BadMatching` | Empty clique, invalid index, repeated vertex, or two vertices from one part in a clique. |
| `ValueError` | `KapError::IncompleteInitial` | Without a threshold, restoring unlisted vertices as singletons does not give exactly `max(part sizes)` nodes. |
| `ValueError` | `KapError::ForbiddenInitial` | With a threshold, the initial solution places a forbidden pair in one clique. |
| `ValueError` | - | Python `local_search` receives both `start` and `initial`. |
| - | `KapError::BadOption("initial")` | Rust selects Cm, Em, Fm, or Memetic with supplied cliques. |

Python also rejects non-integer or boolean indices with `ValueError`. ILS and local-search option errors remain as documented on their guide pages.
All matrix entries are checked against the LAP's numeric rules when a start is supplied, even with zero sweeps.


## Example

<div class="lang-tabs">

```python
{{#include ../examples/warm_start.py}}
```

```rust
{{#include ../../../crates/kap-solver/examples/doc_warm_start.rs}}
```

</div>

Python output:

```text
{{#include ../examples/warm_start.python.txt}}
```

The Rust example prints the same cliques and costs, with the same free vertices.

## Notes and limits

- `threshold=None` preserves unchanged behaviour. All-integer Python matrices require an integral threshold and keep exact int64 arithmetic. For a fractional threshold, explicitly convert the matrices to float64; otherwise the call raises `ValueError`. See [optional matching](optional-matching.md) for the shifted objective, forbidden pairs, and validation.

- Without a threshold, a complete solution has exactly N=max(part sizes) nodes after singleton restoration. It need not contain a vertex from every part in each node.
- Python accepts an `AssignmentResult` or an iterable of cliques. Only the result's `matches` field is used; costs are recomputed from the supplied matrices.
- Unlisted vertices become singletons. Without a threshold this permits omitted free vertices but requires a complete assignment. With a threshold, arbitrary partial and empty assignments are accepted, subject to the index, uniqueness, and forbidden-pair checks.
- ILS refines the supplied solution before starting all chains. Its time budget starts after that initial refinement.
- DV uses k LAPs per 1DV sweep (one for k=2), or up to 2^(k−1)−1 per sDV sweep.

## See also

[Input format](input.md) · [Local search](local-search.md) · [ILS](ils.md)
