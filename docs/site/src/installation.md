# Installation

## PyPI wheels

The Python package requires Python 3.10 or newer and NumPy. The wheel workflow builds CPython abi3 wheels with a Python 3.10 minimum for these targets:

| System | Architecture |
|---|---|
| Linux | x86-64, AArch64 |
| macOS | x86-64, Apple Silicon (AArch64) |
| Windows | x86-64 |

The workflow tests Python 3.10 and 3.14. Each abi3 wheel serves compatible CPython versions from 3.10 onward; it does not require a separate wheel for every Python minor version.

```sh
python -m pip install kap
```

The wheel includes the native Rust core and its LAP solver. It requires no Rust compiler at installation time.

## Optional Python backends

SciPy supplies both the exact solver's HiGHS MILP implementation and the optional SciPy LAP backend:

```sh
python -m pip install "kap[scipy]"
```

The other LAP extras are `kap[lap]`, `kap[lapjv]`, and `kap[munkres]`. `lapjv` requires square matrices. See [linear assignment](lap.md) for the interfaces and [benchmarks](benchmarks.md#lap-backends) for timings.

## Build from source

The workspace declares Rust 1.85 as its minimum toolchain version and uses Rust edition 2024. Create a Python virtual environment and install maturin before building:

```sh
git clone https://github.com/inspiros/kap
cd kap
python -m venv .venv
. .venv/bin/activate
python -m pip install "maturin>=1.9,<2"
maturin develop --release
```

`maturin develop` installs the extension into the active virtual environment. A source installation through `python -m pip install .` also builds with maturin and requires Rust.

## Rust crate

The package is `kap-solver`; the library import is `kap`. For the version documented here, use:

```toml
[dependencies]
kap-solver = "0.3.0-alpha.1"
```

The crate's default feature set is empty. Enable `rayon` for parallel chains, trials, or neighborhood work:

```toml
[dependencies]
kap-solver = { version = "0.3.0-alpha.1", features = ["rayon"] }
```

`MatrixView::from_row_major` and `MatrixView::from_strided` require no optional features.
