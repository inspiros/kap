use std::cell::RefCell;

use kap::{Algo, Cost, Dv, KSolution, LapError, LapSolver, MatrixView, Options, Sense};
use numpy::ndarray::ArrayView2;
use numpy::{Element, PyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Matrices with fewer entries are solved without releasing the GIL.
const DETACH_MIN_ELEMS: usize = 4096;

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
    // Releasing the GIL costs more than solving a small matrix.
    let r = if view.len() <= DETACH_MIN_ELEMS {
        solve_view(view, sense)
    } else {
        py.detach(|| solve_view(view, sense))
    };
    Some(r.map_err(lap_error))
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

type KResult<A> = (Vec<Vec<(usize, usize)>>, Vec<A>, Vec<(usize, usize)>);

fn try_k_dtype<'py, T>(
    py: Python<'py>,
    costs: &[Bound<'py, PyAny>],
    algo: Algo,
    opts: &Options,
) -> Option<PyResult<KResult<T::Acc>>>
where
    T: Cost + Element,
    T::Acc: Cost<Acc = T::Acc>,
{
    let arrays: Vec<PyReadonlyArray2<'py, T>> = costs
        .iter()
        .map(|cost| cost.extract().ok())
        .collect::<Option<_>>()?;
    let arrays: Vec<_> = arrays.iter().map(|a| a.as_array()).collect();
    // Borrow contiguous inputs; only irregularly strided inputs need a staging copy.
    let owned: Vec<Option<Vec<T>>> = arrays
        .iter()
        .map(|a| {
            if a.to_slice().is_some() || a.t().to_slice().is_some() {
                None
            } else {
                Some(a.iter().copied().collect())
            }
        })
        .collect();
    let views = arrays
        .iter()
        .zip(&owned)
        .map(|(a, owned)| {
            let (r, c) = a.dim();
            if let Some(data) = a.to_slice() {
                MatrixView::from_row_major(data, r, c)
            } else if let Some(data) = a.t().to_slice() {
                MatrixView::from_row_major(data, c, r).map(MatrixView::t)
            } else {
                MatrixView::from_row_major(owned.as_ref().expect("staging copy"), r, c)
            }
        })
        .collect::<Result<Vec<_>, _>>();
    Some(views.map_err(lap_error).and_then(|views| {
        py.detach(|| kap::k_assignment(&views, algo, opts))
            .map(
                |KSolution {
                     cliques,
                     costs,
                     free,
                     ..
                 }| (cliques, costs, free),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }))
}

/// Solve a k-partite problem natively; return Python lists.
#[pyfunction]
#[pyo3(signature = (cost_matrices, algo, maximize = false, max_iter = 100, n_trials = 1, seed = 0, refine = None))]
#[allow(clippy::too_many_arguments)]
fn k_assignment<'py>(
    py: Python<'py>,
    cost_matrices: Vec<Bound<'py, PyAny>>,
    algo: &str,
    maximize: bool,
    max_iter: usize,
    n_trials: usize,
    seed: u64,
    refine: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let algo = match algo.to_ascii_lowercase().as_str() {
        "a" | "am" => Algo::Am,
        "b" | "bm" => Algo::Bm,
        "c" | "cm" => Algo::Cm,
        "d" | "dm" => Algo::Dm,
        "e" | "em" => Algo::Em,
        "f" | "fm" => Algo::Fm,
        "shift" | "shiftam" => Algo::Shift,
        "1dv" | "dv1" => Algo::Dv1,
        "sdv" | "dvs" => Algo::DvS,
        _ => {
            return Err(PyValueError::new_err(
                "algo must be Am | Bm | Cm | Dm | Em | Fm | Shift | 1DV | sDV",
            ));
        }
    };
    let refine = refine
        .map(|name| match name.to_ascii_lowercase().as_str() {
            "1dv" | "dv1" => Ok(Dv::One),
            "sdv" | "dvs" => Ok(Dv::Subsets),
            _ => Err(PyValueError::new_err("refine must be 1DV | sDV")),
        })
        .transpose()?;
    let sense = if maximize {
        Sense::Maximize
    } else {
        Sense::Minimize
    };
    let opts = Options {
        sense,
        max_iter,
        n_trials,
        seed,
        refine,
    };
    if let Some(result) = try_k_dtype::<f64>(py, &cost_matrices, algo, &opts) {
        return Ok(result?.into_pyobject(py)?.unbind().into_any());
    }
    if let Some(result) = try_k_dtype::<i64>(py, &cost_matrices, algo, &opts) {
        return Ok(result?.into_pyobject(py)?.unbind().into_any());
    }
    Err(PyTypeError::new_err(
        "cost_matrices must contain 2-D numpy arrays, all float64 or all int64 in native byte order",
    ))
}

/// Native core of the `kap` Python package.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", kap::VERSION)?;
    m.add_function(wrap_pyfunction!(linear_sum_assignment, m)?)?;
    m.add_function(wrap_pyfunction!(k_assignment, m)?)?;
    Ok(())
}
