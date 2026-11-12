use pyo3::prelude::*;

/// Native core of the `kap` Python package.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", kap::VERSION)?;
    Ok(())
}
