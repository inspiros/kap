# Installation

```bash
pip install kap
```

`kap` needs only `numpy`, plus at least one LAP backend. Install a backend
with its extra:

```bash
pip install "kap[scipy]"     # scipy.optimize.linear_sum_assignment (default)
pip install "kap[lap]"       # lap.lapjv
pip install "kap[lapjv]"     # lapjv.lapjv (square matrices only)
pip install "kap[munkres]"   # munkres (pure Python, slow)
```

Python 3.9 or newer is required.

## From source

Install a Rust toolchain (1.85 or newer), then run:

```bash
git clone https://github.com/inspiros/kap
cd kap
pip install .
# Or, in an activated virtual environment with maturin installed:
maturin develop
```
