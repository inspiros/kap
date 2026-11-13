# Solver selection

Use [iterated local search](ils.md) for heuristic k-assignment.
Use [linear assignment](lap.md) for a two-part problem.

## By task


| Need | Function | When |
|---|---|---|
| Default heuristic search | `ils` / `Algo::Ils` | Use `time_limit` and several chains for quality; measured 1 s runs with eight chains reached the proven optimum on all 33 small benchmark instances. |
| Original multiple Hungarian methods | `multiple_hungarian` / Am–Fm | Reproduce the paper's methods; Am, Dm, and Shift limit construction work, while Bm-based methods grow rapidly with k. |
| Refine a complete solution | `local_search` / DV | Apply deterministic 1DV or sDV sweeps, with a constructed start or a supplied complete solution. |
| Population search | `memetic` / `Algo::Memetic` | Compare mutation and crossover with ILS on your instances; measured multi-chain ILS costs were lower or equal at 60 s. |
| Choose a method by name | `k_assignment` | Retain a common Python dispatcher or select an explicit Rust `Algo`. |
| Allow vertices to stay unmatched | `threshold=` / `Options::threshold` | Use [optional matching](optional-matching.md) with the native heuristics; forbidden infinities exclude individual pairs. |
| Prove an optimum | `exact` (Python only) | Use small equal-size inputs, with about 10⁵ tuple variables as a practical limit. |
| Bound the remaining improvement | `lower_bound` | Measured k=3 and metric-cost gaps were small; random costs at k≥4 gave looser bounds. |
| Solve a two-part LAP | `linear_sum_assignment` | Compute an optimal rectangular linear assignment. |

See the [benchmarks](benchmarks.md) for instance definitions, hardware, timings, and gaps.


## Usage

1. Define your parts and [pairwise cost matrices](input.md#matrix-order).
2. Run [ILS](ils.md) with an iteration limit or time budget.
3. Refine an existing complete assignment with a [warm start](warm-start.md).
4. For equal-size parts, compare against a [lower bound](lower-bound.md), or use the Python [exact solver](exact.md) on a small instance.

For the mathematical objective and unequal-size parts, see [problem definition](problem.md).
