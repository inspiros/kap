# kap-solver

Rust core of [kap](https://github.com/inspiros/kap): solvers for the k-assignment
(k-partite matching) problem and a rectangular linear assignment solver.

```toml
[dependencies]
kap-solver = "0.3"
```

```rust
use kap::{MatrixView, Sense, linear_sum_assignment};

let cost = [4i64, 1, 3, 2, 0, 5, 3, 2, 2];
let sol = linear_sum_assignment(MatrixView::from_row_major(&cost, 3, 3)?, Sense::Minimize)?;
assert_eq!(sol.col_ind, vec![1, 0, 2]);
assert_eq!(sol.cost, 5);
# Ok::<(), kap::LapError>(())
```

Status: early development. The linear assignment solver is in place; the
k-assignment algorithms (Am–Fm) are being ported from the Python package.
