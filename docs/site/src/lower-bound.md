# Lower bound

`lower_bound` bounds the optimum for equal-size parts: below for minimization and above for maximization. Use it to assess the remaining gap of a heuristic solution.

## Signature

The Python declarations show public defaults. The Rust declaration shows the entry point; its types are imported from `kap`.

<div class="lang-tabs">

```python
def lower_bound(cost_matrices, *, maximize=False, max_iter=300) -> int | float:
    ...
```

```rust
pub fn lower_bound<T: Cost>(
    mats: &[MatrixView<'_, T>],
    opts: &LowerBoundOptions<T::Acc>,
) -> Result<T::Acc, KapError>
where
    T::Acc: LowerBoundAcc;
```

</div>

## Parameters

| Python | Rust | Default | Meaning |
|---|---|---|---|
| `cost_matrices` | `mats` | Required | Pairwise matrices in [combinations order](input.md#matrix-order). |
| `maximize` | `opts.sense` | `False` / `Sense::Minimize` | Maximize the original costs when true / `Sense::Maximize`. |
| `max_iter` | `opts.max_iter` | `300` | Nonnegative subgradient updates per triple; zero still evaluates the initial seeds. |
| - | `opts.upper` | `None` | Optional finite numerical upper bound on the whole optimum, in the original objective sense. The caller must establish its validity. |

## Returns

Python returns an `int` for integer/boolean inputs and a `float` otherwise. Rust returns `Result<T::Acc, KapError>` with i64 or f64 accumulation. Integer costs give a rigorous bound. Float bounds are computed in floating point and can exceed the true minimum (or fall below the true maximum) by rounding error on the order of machine epsilon times the sum of the cost magnitudes involved. This error can be large when costs of opposite sign nearly cancel.

For a positive minimization solution cost c and bound b, `(c-b)/c` bounds the relative improvement still possible. The example uses this definition; division by zero or a negative c needs a different normalization.

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

Python raises `TypeError` for non-integer or boolean `max_iter`, and `ValueError` for negative values or unequal part sizes. Rust returns `KapError::Unsupported` for unequal sizes, a non-finite `upper`, or exhausted arithmetic headroom. Invalid numeric entries and infeasible pairwise LAPs produce `KapError::Lap`. If a multiplier update exhausts headroom after valid bounds were evaluated, that triple retains its best evaluated bound.


## Example

<div class="lang-tabs">

```python
{{#include ../examples/lower_bound.py}}
```

```rust
{{#include ../../../crates/kap-solver/examples/doc_lower_bound.rs}}
```

</div>

Python output:

```text
{{#include ../examples/lower_bound.python.txt}}
```

The Rust example prints the same cost, bound, and gap.

## Notes and limits

- For k=2, the bound is the exact LAP optimum. For k=3, it is an axial Lagrangian bound, at best equal to the LP relaxation bound.
- For k≥4, the function averages all triple bounds with scale 1/(k−2). Random non-metric costs can leave a large gap; measured metric instances gave smaller gaps.
- Each active triple uses O(n²) scratch space. The number of triples is k(k−1)(k−2)/6.
- The initial evaluations include a pairwise-LAP sum and zero/dual multiplier seeds. Increasing max_iter retains all earlier evaluations; the returned bound does not weaken.
- Rust `upper` clamps the final numerical bound in either objective sense. It is not a feasible-solution argument and is not used as the per-triple Polyak target.
- The method is deterministic and has no time-limit parameter. Equal zero-sized parts return zero.

## See also

[Proven-optimum benchmarks](benchmarks.md#ils-and-the-lower-bound-against-the-proven-optimum) · [Exact solver](exact.md) · [Input and dtypes](input.md#dtype-promotion)
