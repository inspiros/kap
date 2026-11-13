<h1><img src="https://raw.githubusercontent.com/inspiros/kap/v0.3.0/assets/kap-icon.svg" alt="" width="40" height="40" align="absmiddle" /> kap</h1>

[![Build wheels](https://img.shields.io/github/actions/workflow/status/inspiros/kap/build_wheels.yml)](https://github.com/inspiros/kap/actions/workflows/build_wheels.yml)
[![Documentation](https://img.shields.io/badge/docs-website-16776d?logo=mdbook&logoColor=white)](https://inspiros.github.io/kap/)
[![PyPI](https://img.shields.io/pypi/v/kap)](https://pypi.org/project/kap/)
[![Downloads](https://static.pepy.tech/badge/kap)](https://pepy.tech/project/kap)
[![crates.io](https://img.shields.io/crates/v/kap-solver)](https://crates.io/crates/kap-solver)
[![crates.io downloads](https://img.shields.io/crates/d/kap-solver)](https://crates.io/crates/kap-solver)
[![docs.rs](https://img.shields.io/docsrs/kap-solver)](https://docs.rs/kap-solver)
[![DOI](https://zenodo.org/badge/731651337.svg)](https://doi.org/10.5281/zenodo.10449790)
[![License](https://img.shields.io/github/license/inspiros/kap)](https://github.com/inspiros/kap/blob/v0.3.0/LICENSE.txt)

![kap: three parts partitioned into three disjoint cliques](https://raw.githubusercontent.com/inspiros/kap/v0.3.0/assets/kpartite_hero.svg)

`kap` solves k-partite assignment problems from pairwise cost matrices. It partitions vertices into cliques with at most one vertex per part and minimizes the total pairwise cost. It provides heuristics, lower bounds, and linear assignment in Python and Rust; Python also provides an exact solver through SciPy and HiGHS. For k > 2, the heuristics do not guarantee an optimal solution.

<p align="center">
  <a href="https://inspiros.github.io/kap/"><img src="https://img.shields.io/badge/Read_the_docs-16776d?style=for-the-badge&amp;logo=mdbook&amp;logoColor=white" alt="Read the documentation" /></a>
  <a href="https://pypi.org/project/kap/"><img src="https://img.shields.io/badge/Install_from_PyPI-3775a9?style=for-the-badge&amp;logo=pypi&amp;logoColor=white" alt="Install from PyPI" /></a>
</p>

## Install

Install the Python package from PyPI (Python 3.10 or newer; wheels include the Rust core):

```sh
pip install kap
```

Install `scipy` too (`pip install "kap[scipy]"`) to use `exact`. Rust users can add the [`kap-solver`](https://crates.io/crates/kap-solver) crate (`cargo add kap-solver`) and import it as `kap`. For 0.2.3 and its API, see the [0.2.3 README](https://github.com/inspiros/kap/tree/v0.2.3#readme).

## Examples

Python:

```python
import numpy as np
import kap

costs = [np.array([[4, 1], [2, 3]]) for _ in range(3)]
result = kap.ils(costs, max_iter=10, random_state=0)
print(result.cliques, result.costs)
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

The native dispatcher defaults to ILS; select `algo="memetic"` explicitly to use memetic search. Native heuristics support optional matching via `threshold=`, including forbidden pairs; returned costs remain in original units. ILS and local search accept warm starts, which may be partial or empty when a threshold is used. See [optional matching](https://inspiros.github.io/kap/optional-matching.html) and [warm starts](https://inspiros.github.io/kap/warm-start.html) for details.

## References and citation

`kap` combines several assignment methods. The main algorithm references are:

- **Local search and ILS:** D. Karapetyan and G. Gutin, “Local search heuristics for the multidimensional assignment problem,” *Journal of Heuristics* 17(3), 201–249, 2011. [arXiv:0806.3258](https://arxiv.org/abs/0806.3258).
- **Shift construction:** D. Karapetyan, G. Gutin, and B. Goldengorin, “Empirical evaluation of construction heuristics for the multidimensional assignment problem,” *Text in Algorithmics* 11, 107–122, 2009. [arXiv:0906.2960](https://arxiv.org/abs/0906.2960).
- **Memetic search and population sizing:** D. Karapetyan and G. Gutin, “A New Approach to Population Sizing for Memetic Algorithms: A Case Study for the Multidimensional Assignment Problem,” *Evolutionary Computation* 19(3), 345–371, 2011. [arXiv:1003.4314](https://arxiv.org/abs/1003.4314).
- **Linear assignment:** R. Jonker and A. Volgenant, “A shortest augmenting path algorithm for dense and sparse linear assignment problems,” 1987; D. F. Crouse, “On implementing 2D rectangular assignment algorithms,” 2016.
- **Multiple Hungarian methods (Am–Fm):** B. Gabrovšek et al., “Multiple Hungarian Method for k-Assignment Problem,” *Mathematics* 8(11), 2050, 2020. [doi:10.3390/math8112050](https://doi.org/10.3390/math8112050).

The exact solver uses [SciPy's MILP interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html) to [HiGHS](https://highs.dev/). See the [algorithm reference](https://inspiros.github.io/kap/algorithms.html#references) for additional papers, implementation details, and differences from the published methods, and the [lower-bound guide](https://inspiros.github.io/kap/lower-bound.html) for the bound implementation.

To cite the software, use the [Zenodo DOI](https://doi.org/10.5281/zenodo.10449790) and [CITATION.cff](https://github.com/inspiros/kap/blob/v0.3.0/CITATION.cff). Cite the relevant algorithm papers when discussing a particular method. Release history up to this version is in [CHANGELOG.md](https://github.com/inspiros/kap/blob/v0.3.0/CHANGELOG.md).

## License

[MIT](https://github.com/inspiros/kap/blob/v0.3.0/LICENSE.txt).
