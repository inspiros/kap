# Input and results

The k-assignment functions accept pairwise cost matrices and return cliques with their summed costs. Use unequal part sizes when some cliques may omit parts.

## Signature

The Python declarations show public defaults. The Rust declaration shows the entry point; its types are imported from `kap`.

<div class="lang-tabs">

```python
def k_assignment(cost_matrices: Sequence[Array2DLike], algo: Optional[str] = None,
                 max_iter: Optional[int] = None, n_trials: int = 1, maximize: bool = False,
                 return_free: bool = False, backend: str = "kap",
                 random_state: Optional[Union[np.random.RandomState, int]] = None,
                 refine: Optional[str] = None, time_limit: Optional[float] = None,
                 population: Optional[int] = None) -> AssignmentResult:
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
| `maximize` | `opts.sense` | `False` / `Sense::Minimize` | Maximize the original costs when true / `Sense::Maximize`. |
| `return_free` | - | `False` | Include singleton vertices in Python `free`; Rust always returns them. |
| `backend` | - | `"kap"` | Native kap or optional scipy, lap, lapjv, munkres; newer methods require native kap. |
| `random_state` | `opts.seed` | `None` / `0` | Python accepts a nonnegative integer, `RandomState`, or `Generator`; Rust uses `u64`. |
| `refine` | `opts.refine` or `opts.dv` | `None` / `None` or `Dv::One` | 1DV or sDV after construction; inside ILS/memetic it selects the search neighborhood. |
| `time_limit` | `opts.time_limit` | `None` | Python: finite positive seconds. Rust: `Option<Duration>`. Both time and iteration limits apply. |
| `population` | `opts.population` | `None` | Memetic population size; supplied values must be integers at least 2. |

## Returns

Python returns `AssignmentResult(cliques, costs, free)`. Each clique contains `(part, vertex)` pairs, and its cost is the sum of its pairwise edges. `result.cost` gives the total cost. `free` is `None` unless requested; it then lists singleton vertices.

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

## Matrix order

For k=4, supply the matrices in this order. Rows index the first part; columns index the second part.

| Position | Pair | Matrix shape |
|---|---|---|
| 0 | (0, 1) | C01: n0 × n1 |
| 1 | (0, 2) | C02: n0 × n2 |
| 2 | (0, 3) | C03: n0 × n3 |
| 3 | (1, 2) | C12: n1 × n2 |
| 4 | (1, 3) | C13: n1 × n3 |
| 5 | (2, 3) | C23: n2 × n3 |

## Example

<div class="lang-tabs">

```python
{{#include ../examples/input.py}}
```

```rust
{{#include ../../../crates/kap-solver/examples/doc_input.rs}}
```

</div>

Python output:

```text
{{#include ../examples/input.python.txt}}
```

The Rust example prints the same cliques and costs, with the same free vertices.

Here k=3, with part sizes 3, 3, and 2. C01 has shape (3, 3); C02 and C12 each have shape (3, 2). Each `cliques` entry lists the vertices in one clique. `costs` gives the cost of that clique, summed over its pairs. `free` is empty because every vertex belongs to a clique of size at least two; a part missing from a clique does not make the other vertices free.

The minimizing result contains these cliques:

| Vertices (part, vertex) | Pairwise cost calculation | Clique cost |
|---|---|---|
| (0, 0), (1, 1), (2, 0) | C01[0,1] + C02[0,0] + C12[1,0] = 2 + 3 + 1 | 6 |
| (0, 2), (1, 2), (2, 1) | C01[2,2] + C02[2,1] + C12[2,1] = 1 + 4 + 2 | 7 |
| (0, 1), (1, 0) | C01[1,0] = 6 | 6 |

The second solve maximizes the same pairwise costs. The costs printed in that result retain their original signs.

## Dtype promotion

The native Python k-assignment methods and lower bound promote all integer/boolean matrices to int64. If any matrix is floating-point, every matrix becomes float64. Complex, object, and string dtypes are rejected.

A uint64 value above the signed 64-bit maximum raises OverflowError on the all-integer path. Conversion to float64 permits larger magnitudes but can lose integer precision. The [linear-assignment function](lap.md#notes-and-limits) has its own dtype conversion rules; the exact solver forms its MILP objective in float64.

## Rust matrix views

`MatrixView::from_row_major(data, rows, cols)` borrows a row-major slice. The slice must contain at least rows*cols elements; extra elements are allowed. `MatrixView::from_strided(data, offset, rows, cols, row_stride, col_stride)` borrows a general strided view, with strides in elements and negative strides permitted when every addressed element stays inside the slice. Both constructors return `Result<MatrixView, LapError>`.

`t()` returns a transposed view; `rows()`, `cols()`, and `get(row, col)` inspect it. There is no `MatrixView::new` constructor. All views passed to a solve share an element type supported by `Cost`.

## Notes and limits

- The native core accepts 2 through 64 parts. A valid matrix count is k(k−1)/2, with consistent dimensions across every occurrence of a part.
- Part sizes may differ for the heuristic methods; exact and lower_bound require equal sizes.
- Free vertices are singleton nodes. They contribute zero pairwise cost and are excluded from the clique and cost lists.
- Select maximize=True in Python, or Sense::Maximize in Rust. The returned costs use the original objective sense.

## See also

[Dispatcher options](dispatcher.md) · [Warm starts](warm-start.md) · [Linear assignment](lap.md)
