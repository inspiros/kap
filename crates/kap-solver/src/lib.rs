//! k-assignment (k-partite matching) solvers.
//!
//! The Rust package is published as `kap-solver`; the library is imported as `kap`.

/// Version of this crate, shared with the `kap` Python package.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod algo;
pub mod kpartite;
pub mod lap;

pub use algo::{Algo, Dv, Options, k_assignment, k_assignment_from};
pub use kpartite::{KPartite, KSolution, KapError, Quotient};

pub use lap::{
    Acc, Cost, Duals, LapError, LapSolution, LapSolver, MatrixView, Sense, linear_sum_assignment,
};
