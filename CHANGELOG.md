# Changelog

## [Unreleased]

## [0.3.0] - 2026-11-13

First release of the Rust crate `kap-solver` on crates.io.

### Added

- Rust core and Rust API, with a native linear assignment solver.
- ILS, memetic search, dimensionwise local search, and Shift, alongside the multiple Hungarian methods Am–Fm. Python exposes these through individual functions and `k_assignment`.
- An exact solver for equal-size instances using SciPy and HiGHS, and a lower bound for equal-size instances.
- Warm starts for ILS and local search. Without a threshold the start must be complete; with a threshold it may be partial or empty.
- Optional matching in native heuristics using a threshold, with singletons and forbidden pairs. The optimized objective uses shifted pair costs; returned costs remain in original units.

### Changed

- Python `AssignmentResult` fields renamed to `cliques`, `costs`, `free` to match Rust `KSolution`, with a new `cost` total; `matches` and `matching_costs` remain as deprecated aliases until 0.4.
- Fractional thresholds with all-integer Python matrices require explicit float64 conversion and otherwise raise `ValueError`.
- The default backend uses the Rust core, and ILS is the default k-assignment method.
- Native Cm, Em, and Fm follow the paper's search rules and can return different results from 0.2.3. Fm runs all `max_iter` steps.
- Python 3.10 or newer is required. The `lapsolver` backend was removed.

### Fixed

- Checked integer LAP arithmetic and total-cost accumulation. If JV dual updates overflow, solving restarts with shortest augmenting paths; unrepresentable arithmetic returns an overflow error instead of panicking or wrapping.
- Rejected the most negative int64 entry safely when maximizing, before negation can overflow.

## [0.2.3]

- Fixed incorrect quotient-graph cost matrix indices for k ≥ 4, which could produce suboptimal assignments and reported costs below the true optimum. Recompute results obtained with k ≥ 4.
- Fixed Em so `n_trials` runs independent trials from the Bm solution.
- Changed Fm to stop on a plateau; sideways moves must reach a different solution and cannot return to the previous one.
- Fixed an internal argument passed to the wrong `linear_assignment` parameter in Am/Bm; this had no effect on results.
- Fixed a crash when a fixed matching is empty.

[Unreleased]: https://github.com/inspiros/kap/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/inspiros/kap/compare/v0.2.3...v0.3.0
[0.2.3]: https://github.com/inspiros/kap/releases/tag/v0.2.3
