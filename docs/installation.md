# Installation

```bash
pip install kap
```

`kap` needs only `numpy`; its native LAP solver is the default backend.
Other backends are optional extras:

```bash
pip install "kap[scipy]"     # scipy.optimize.linear_sum_assignment
pip install "kap[lap]"       # lap.lapjv
pip install "kap[lapjv]"     # lapjv.lapjv (square matrices only)
pip install "kap[munkres]"   # munkres (pure Python, slow)
```

Python 3.10 or newer is required.

## From source

Install a Rust toolchain (1.85 or newer), then run:

```bash
git clone https://github.com/inspiros/kap
cd kap
pip install .
# Or, in an activated virtual environment with maturin installed:
maturin develop
```
