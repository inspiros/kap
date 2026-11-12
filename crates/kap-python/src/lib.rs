use std::cell::RefCell;

use kap::{Cost, LapError, LapSolver, MatrixView, Sense};
use numpy::ndarray::ArrayView2;
use numpy::{Element, PyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

thread_local! {
    static SOLVER_F64: RefCell<LapSolver<f64>> = RefCell::new(LapSolver::new());
    static SOLVER_I64: RefCell<LapSolver<i64>> = RefCell::new(LapSolver::new());
}

/// Accumulator types with a per-thread reusable solver.
trait WithSolver: kap::Acc {
    fn with_solver<R>(f: impl FnOnce(&mut LapSolver<Self>) -> R) -> R;
}

impl WithSolver for f64 {
    fn with_solver<R>(f: impl FnOnce(&mut LapSolver<Self>) -> R) -> R {
        SOLVER_F64.with_borrow_mut(f)
    }
}

impl WithSolver for i64 {
    fn with_solver<R>(f: impl FnOnce(&mut LapSolver<Self>) -> R) -> R {
        SOLVER_I64.with_borrow_mut(f)
    }
}

fn lap_error(e: LapError) -> PyErr {
    match e {
        LapError::InvalidCost(..) => {
            PyValueError::new_err("matrix contains invalid numeric entries")
        }
        LapError::Infeasible => PyValueError::new_err("cost matrix is infeasible"),
        LapError::BadShape => PyValueError::new_err("cost matrix has an invalid shape"),
    }
}

/// Solve on a C- or F-contiguous view without copying; copy anything else.
fn solve_view<T>(a: ArrayView2<'_, T>, sense: Sense) -> Result<(Vec<usize>, Vec<usize>), LapError>
where
    T: Cost,
    T::Acc: WithSolver,
{
    let (r, c) = a.dim();
    let run = |m: MatrixView<'_, T>| {
        T::Acc::with_solver(|s| s.solve(m, sense)).map(|sol| (sol.row_ind, sol.col_ind))
    };
    if let Some(s) = a.to_slice() {
        run(MatrixView::from_row_major(s, r, c)?)
    } else if let Some(s) = a.t().to_slice() {
        run(MatrixView::from_row_major(s, c, r)?.t())
    } else {
        let owned: Vec<T> = a.iter().copied().collect();
        run(MatrixView::from_row_major(&owned, r, c)?)
    }
}

fn try_dtype<'py, T>(
    py: Python<'py>,
    cost: &Bound<'py, PyAny>,
    sense: Sense,
) -> Option<PyResult<(Vec<usize>, Vec<usize>)>>
where
    T: Cost + Element,
    T::Acc: WithSolver,
{
    let arr = cost.extract::<PyReadonlyArray2<'py, T>>().ok()?;
    let view = arr.as_array();
    Some(py.detach(|| solve_view(view, sense)).map_err(lap_error))
}

type IndexPair<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

/// Solve the linear sum assignment problem natively.
///
/// `cost` must be a 2-D numpy array of dtype float32/64, int8/16/32/64 or
/// uint8/16/32 in native byte order (any strides). Returns `(row_ind, col_ind)`
/// as int64 arrays, rows increasing, like `scipy.optimize.linear_sum_assignment`.
#[pyfunction]
#[pyo3(signature = (cost, maximize = false))]
fn linear_sum_assignment<'py>(
    py: Python<'py>,
    cost: &Bound<'py, PyAny>,
    maximize: bool,
) -> PyResult<IndexPair<'py>> {
    let sense = if maximize {
        Sense::Maximize
    } else {
        Sense::Minimize
    };
    let result = try_dtype::<f64>(py, cost, sense)
        .or_else(|| try_dtype::<i64>(py, cost, sense))
        .or_else(|| try_dtype::<f32>(py, cost, sense))
        .or_else(|| try_dtype::<i32>(py, cost, sense))
        .or_else(|| try_dtype::<i16>(py, cost, sense))
        .or_else(|| try_dtype::<i8>(py, cost, sense))
        .or_else(|| try_dtype::<u32>(py, cost, sense))
        .or_else(|| try_dtype::<u16>(py, cost, sense))
        .or_else(|| try_dtype::<u8>(py, cost, sense))
        .ok_or_else(|| {
            PyTypeError::new_err(
                "cost must be a 2-D numpy array of a supported numeric dtype in native byte order",
            )
        })?;
    let (rows, cols) = result?;
    let to_py = |v: Vec<usize>| PyArray1::from_vec(py, v.into_iter().map(|x| x as i64).collect());
    Ok((to_py(rows), to_py(cols)))
}

/// Native core of the `kap` Python package.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", kap::VERSION)?;
    m.add_function(wrap_pyfunction!(linear_sum_assignment, m)?)?;
    Ok(())
}
