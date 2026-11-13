# Overview

<figure class="kap-hero">
  <img src="assets/kpartite_hero.svg" alt="kap: three parts partitioned into three disjoint cliques" width="1440" height="560" />
</figure>

`kap` provides Python and Rust solvers for k-partite assignment from pairwise cost matrices. It includes heuristic searches, an optimum bound for equal-size parts, and a linear assignment solver; Python also provides an exact MILP solver.

<nav class="kap-quicklinks" aria-label="Documentation links">
  <a href="installation.md">Installation &rarr;</a>
  <a href="choosing-a-solver.md">Solver selection</a>
  <a href="problem.md">Problem definition</a>
</nav>

## Documentation

<div class="kap-guide-grid">
  <a class="kap-guide" href="input.md"><span class="kap-eyebrow">INPUT</span><strong>Cost matrices and results <span aria-hidden="true">&rarr;</span></strong><span>Matrix order, part sizes, numeric types, and the returned cliques.</span></a>
  <a class="kap-guide" href="choosing-a-solver.md"><span class="kap-eyebrow">SOLVERS</span><strong>Solver selection <span aria-hidden="true">&rarr;</span></strong><span>Heuristic search, linear assignment, exact solving, and bounds.</span></a>
  <a class="kap-guide" href="benchmarks.md"><span class="kap-eyebrow">MEASUREMENTS</span><strong>Benchmarks <span aria-hidden="true">&rarr;</span></strong><span>Measured solution quality, execution times, and algorithm limits.</span></a>
</div>

## Example

<div class="lang-tabs">

```python
{{#include ../examples/overview.py}}
```

```rust
{{#include ../../../crates/kap-solver/examples/doc_overview.rs}}
```

</div>

Python output:

```text
{{#include ../examples/overview.python.txt}}
```

The Rust example prints the same cliques and costs in the same format.

## Initial solutions

Use a [warm start](warm-start.md) to supply a complete assignment to ILS or local search.
For implementation details and the original algorithms, see the [algorithms reference](algorithms.md).

## Reference

B. Gabrovšek, T. Novak, J. Povh, D. Rupnik Poklukar, and J. Žerovnik, “Multiple Hungarian Method for k-Assignment Problem,” *Mathematics* 8(11), 2050, 2020. [doi:10.3390/math8112050](https://doi.org/10.3390/math8112050).

The papers behind the other methods are listed in the [algorithms reference](algorithms.md#references).
