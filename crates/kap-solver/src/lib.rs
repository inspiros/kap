//! k-assignment (k-partite matching) solvers.
//!
//! The Rust package is published as `kap-solver`; the library is imported as `kap`.

/// Version of this crate, shared with the `kap` Python package.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod lap;

pub use lap::{LapError, LapSolution, LapSolver, MatrixView, Sense, linear_sum_assignment};
