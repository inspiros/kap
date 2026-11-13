# Linear assignment

`linear_sum_assignment` computes an optimal assignment with min(rows, columns) pairs from one dense cost matrix. Use it for a two-part problem; use Python `linear_assignment` when you also need costs, free vertices, or an alternative LAP backend.

## Signature

The Python declarations show public defaults. The Rust declaration shows the entry point; its types are imported from `kap`.

<div class="lang-tabs">

```python
def linear_sum_assignment(cost_matrix, maximize: bool = False):
    ...

def linear_assignment(cost_matrix: Array2DLike,
                      maximize: bool = False,
                      return_free: bool = False,
                      backend: str = "kap") -> AssignmentResult:
    ...
```

```rust
pub fn linear_sum_assignment<T: Cost>(
    m: MatrixView<'_, T>,
    sense: Sense,
) -> Result<LapSolution<T::Acc>, LapError>;
```

</div>

## Parameters

| Python | Rust | Default | Meaning |
|---|---|---|---|
| `cost_matrix` | `m` | Required | A dense 2-D matrix; rectangular inputs are supported. |
| `maximize` | `sense` | `False` / explicit Sense | Maximize with True / Sense::Maximize. |
| `return_free` (wrapper only) | - | `False` | Include singleton vertices in the linear_assignment result. |
| `backend` (wrapper only) | - | `"kap"` | kap, scipy, lap, lapjv, or munkres; lapjv accepts square matrices only. |

## Returns

Python `linear_sum_assignment` returns `(row_ind, col_ind)` as NumPy index arrays. Rows are increasing. The cost is `cost_matrix[row_ind, col_ind].sum()`.

Python `linear_assignment` returns `AssignmentResult`: `cliques` is an array of `(row, column)` pairs, `costs` contains the selected entries, and optional `free` contains `(0, row)` or `(1, column)` singleton vertices.

Rust returns `Result<LapSolution<T::Acc>, LapError>` with `row_ind`, `col_ind`, and total `cost`. It has no backend-selector or free-vertex field.

## Errors

| Python | Rust | Condition |
|---|---|---|
| `ValueError` | `LapError::BadShape` from MatrixView construction | Non-2-D Python input or Rust storage/strides outside the slice. |
| `ValueError` | `LapError::InvalidCost(row, col)` | NaN, wrong-sign infinity, or a magnitude outside the solver's arithmetic headroom. |
| `OverflowError` (native LAP) | `LapError::Overflow` | An integer intermediate value, dual, or total cost does not fit int64. |
| `ValueError` | `LapError::Infeasible` | No assignment of min(rows, columns) pairs avoids forbidden entries. |
| `TypeError` | Rejected by Cost at compilation | Unsupported matrix dtype. |
| `ValueError` | - | The Python wrapper receives an unknown backend. |
| `ModuleNotFoundError` | - | A selected optional backend is unavailable. |

Optional backends can also raise their own input errors, including square-input requirements for lapjv.


## Example

<div class="lang-tabs">

```python
{{#include ../examples/lap.py}}
```

```rust
{{#include ../../../crates/kap-solver/examples/doc_lap.rs}}
```

</div>

Python output:

```text
{{#include ../examples/lap.python.txt}}
```

The Rust example prints the same row indices, column indices, and total cost. It does not print the Python wrapper's `free` field.

## Notes and limits

- Positive infinity forbids a pair for minimization; negative infinity does so for maximization. These rules describe the native LAP solver.
- Square finite matrices use Jonker–Volgenant initialization and augmentation. Rectangular matrices or matrices with forbidden entries use shortest augmenting paths.
- Integer LAP arithmetic is checked. If JV dual updates overflow, the solver restarts with shortest augmenting paths. If that path or the total cost also overflows, the call returns an error; integer costs are not converted to float64.
- With tied costs, square inputs can choose a different optimum from SciPy. Rectangular inputs use SciPy's tie-breaking order.
- Worst-case square LAP work is O(n³). Reuse Rust LapSolver when solving many matrices to retain its buffers.
- Native Python supports float32/64 and signed integer types through int64 plus uint8/16/32. Booleans become uint8; uint64 fitting int64 becomes int64. Float16, longdouble, and larger uint64 values convert to float64, with possible loss of precision.
- Rust Cost implementations support f32/f64, i8/i16/i32/i64, and u8/u16/u32; they accumulate in f64 or i64. Sparse matrices are not supported.

## See also

[MatrixView constructors](input.md#rust-matrix-views) · [Backend measurements](benchmarks.md#lap-backends) · [Multiple Hungarian methods](multiple-hungarian.md)
