# Python examples

Run an example from the repository root, for example: `python examples/quickstart.py`.

- `quickstart.py` — solve a three-part assignment.
- `linear_assignment.py` — solve a rectangular linear assignment, including maximization.
- `compare_solvers.py` — compare heuristics, a bound, and optionally the exact solver.
- `optional_matching.py` — leave vertices free using a threshold.
- `warm_start.py` — improve a solution using a warm start.

SciPy is optional; install it with `pip install "kap[scipy]"` for the comparisons and exact solver.

Rust examples are in `crates/kap-solver/examples`; run one with `cargo run -p kap-solver --example doc_overview`. See the [documentation](https://inspiros.github.io/kap/).
