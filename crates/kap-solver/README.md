# kap-solver

`kap-solver` provides the Rust core of [kap](https://github.com/inspiros/kap). Add `kap-solver = "0.3"` to `[dependencies]` and import the library as `kap`.

```rust
use kap::{k_assignment, Algo, MatrixView, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = [4i64, 1, 2, 3];
    let matrix = MatrixView::from_row_major(&data, 2, 2)?;
    let costs = [matrix, matrix, matrix];
    let result = k_assignment(&costs, Algo::Ils, &Options::default())?;
    println!("{:?} {:?}", result.cliques, result.costs);
    Ok(())
}
```

`k_assignment` runs a selected heuristic; `k_assignment_from` accepts a complete starting solution. `lower_bound` bounds equal-size problems. `linear_sum_assignment` solves rectangular linear assignments. See the [documentation](https://inspiros.github.io/kap/) for inputs and limits.

`Options::threshold` enables optional matching with singleton vertices and forbidden pairs; returned costs remain in original units.
