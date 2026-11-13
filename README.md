<h1><img src="https://raw.githubusercontent.com/inspiros/kap/master/assets/kap-icon.svg" alt="" width="40" height="40" align="absmiddle" /> kap</h1>

[![Build wheels](https://img.shields.io/github/actions/workflow/status/inspiros/kap/build_wheels.yml)](https://github.com/inspiros/kap/actions/workflows/build_wheels.yml)
[![Documentation](https://img.shields.io/badge/docs-website-16776d?logo=mdbook&logoColor=white)](https://inspiros.github.io/kap/)
[![PyPI](https://img.shields.io/pypi/v/kap)](https://pypi.org/project/kap/)
[![Downloads](https://static.pepy.tech/badge/kap)](https://pepy.tech/project/kap)
[![License](https://img.shields.io/github/license/inspiros/kap)](LICENSE.txt)

![kap: three parts partitioned into three disjoint cliques](https://raw.githubusercontent.com/inspiros/kap/master/assets/kpartite_hero.svg)

`kap` solves k-partite assignment problems from pairwise cost matrices. It partitions vertices into cliques with at most one vertex per part and minimizes the total pairwise cost. The Python package and Rust crate provide heuristics, lower bounds, and linear assignment; Python also provides an exact solver through SciPy and HiGHS.

<p align="center">
  <a href="https://inspiros.github.io/kap/"><img src="https://img.shields.io/badge/Read_the_docs-16776d?style=for-the-badge&amp;logo=mdbook&amp;logoColor=white" alt="Read the documentation" /></a>
  <a href="https://pypi.org/project/kap/"><img src="https://img.shields.io/badge/Install_from_PyPI-3775a9?style=for-the-badge&amp;logo=pypi&amp;logoColor=white" alt="Install from PyPI" /></a>
</p>

## Install

```sh
pip install kap
```

Python 3.10 or newer and NumPy are required. Rust users can add `kap-solver = "0.3"` to `[dependencies]` and import `kap`.

## Examples

Python:

```python
import numpy as np
import kap

costs = [np.array([[4, 1], [2, 3]]) for _ in range(3)]
result = kap.ils(costs, max_iter=10, random_state=0)
print(result.matches, result.matching_costs)
```

Rust:

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

## Functions

| Python function | Purpose |
|---|---|
| `ils` | Run iterated local search with independent chains. |
| `memetic` | Evolve a population of locally optimized solutions. |
| `local_search` | Improve a starting solution with dimensionwise variation. |
| `multiple_hungarian` | Run Am–Fm or Shift. |
| `k_assignment` | Dispatch to a selected k-assignment method. |
| `exact` | Solve equal-size instances with SciPy and HiGHS. |
| `lower_bound` | Bound the optimum for equal-size parts. |
| `linear_sum_assignment` | Solve a rectangular linear assignment. |

See the [documentation](https://inspiros.github.io/kap/) for inputs, Rust APIs, algorithm limits, and benchmarks.

## Changes in 0.3

- The default backend uses a Rust core. ILS is the default k-assignment method with this backend.
- `ils`, `memetic`, `local_search`, and `multiple_hungarian` expose individual methods. Shift and dimensionwise variation are available in the native core.
- `exact` uses SciPy and HiGHS; `lower_bound` bounds equal-size instances. `initial` supplies a complete warm start to ILS or local search.
- Native Cm, Em, and Fm follow the paper's search rules and can return different results from 0.2.3. Fm runs all `max_iter` steps.
- The `lapsolver` backend was removed. Python 3.10 or newer is required.

## License

[MIT](LICENSE.txt).
