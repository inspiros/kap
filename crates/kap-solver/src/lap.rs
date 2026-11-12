//! Rectangular linear assignment problem (LAP).
//!
//! Shortest augmenting path method of Jonker–Volgenant, in the rectangular form
//! described by Crouse (2016), "On implementing 2D rectangular assignment
//! algorithms", IEEE TAES 52(4). This is the same algorithm as
//! `scipy.optimize.linear_sum_assignment`.
//!
//! For an `nr x nc` matrix, `min(nr, nc)` pairs are returned, so every row (or
//! every column, whichever side is smaller) is matched. Costs are widened to an
//! accumulator type ([`Cost::Acc`]): `f64` for floating-point input, `i64` for
//! integer input. Integer solving is exact as long as sums of `2 * min(nr, nc)`
//! entries fit in `i64`.
//!
//! For floating-point input, `+inf` marks a forbidden pair; `NaN` and `-inf`
//! are rejected (`+inf` when maximizing is rejected too, `-inf` marks a
//! forbidden pair instead).

use std::fmt;
use std::ops::{Add, Neg, Sub};

/// Accumulator type in which the solver works.
pub trait Acc:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + fmt::Debug
    + Send
    + Sync
    + 'static
{
    const ZERO: Self;
    /// Value larger than any finite path length; `+inf` for floats.
    const INF: Self;
    /// Whether `self` is a permitted entry of a minimization problem
    /// (finite or `+INF` for floats).
    fn is_valid(self) -> bool;
}

impl Acc for f64 {
    const ZERO: Self = 0.0;
    const INF: Self = f64::INFINITY;
    #[inline]
    fn is_valid(self) -> bool {
        !self.is_nan() && self != f64::NEG_INFINITY
    }
}

impl Acc for i64 {
    const ZERO: Self = 0;
    const INF: Self = i64::MAX;
    #[inline]
    fn is_valid(self) -> bool {
        // Leave headroom so that `-x` and the dual updates cannot overflow
        // for any single entry.
        self > i64::MIN / 4 && self < i64::MAX / 4
    }
}

/// Element type of a cost matrix.
pub trait Cost: Copy + fmt::Debug + Send + Sync + 'static {
    type Acc: Acc;
    fn widen(self) -> Self::Acc;
}

macro_rules! impl_cost {
    ($acc:ty: $($t:ty),*) => {$(
        impl Cost for $t {
            type Acc = $acc;
            #[inline]
            fn widen(self) -> $acc {
                self as $acc
            }
        }
    )*};
}
impl_cost!(f64: f32, f64);
impl_cost!(i64: i8, i16, i32, i64, u8, u16, u32);

/// Optimization direction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Sense {
    #[default]
    Minimize,
    Maximize,
}

/// Errors returned by the solver and by [`MatrixView`] constructors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LapError {
    /// The data slice is too short or a stride points outside it.
    BadShape,
    /// An entry is NaN, has the wrong sign of infinity, or is too large in
    /// magnitude for exact integer arithmetic. Holds `(row, col)`.
    InvalidCost(usize, usize),
    /// No complete assignment avoids the forbidden (infinite) entries.
    Infeasible,
}

impl fmt::Display for LapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LapError::BadShape => write!(f, "matrix shape or strides do not fit the data"),
            LapError::InvalidCost(i, j) => write!(f, "invalid cost entry at ({i}, {j})"),
            LapError::Infeasible => write!(f, "cost matrix is infeasible"),
        }
    }
}

impl std::error::Error for LapError {}

/// Read-only 2-D view with arbitrary (possibly negative) element strides.
#[derive(Clone, Copy, Debug)]
pub struct MatrixView<'a, T> {
    data: &'a [T],
    offset: usize,
    rows: usize,
    cols: usize,
    row_stride: isize,
    col_stride: isize,
}

impl<'a, T: Copy> MatrixView<'a, T> {
    /// C-contiguous `rows x cols` matrix; `data` may be longer than needed.
    pub fn from_row_major(data: &'a [T], rows: usize, cols: usize) -> Result<Self, LapError> {
        let need = rows.checked_mul(cols).ok_or(LapError::BadShape)?;
        if data.len() < need {
            return Err(LapError::BadShape);
        }
        Ok(MatrixView {
            data,
            offset: 0,
            rows,
            cols,
            row_stride: cols as isize,
            col_stride: 1,
        })
    }

    /// General strided view: element `(i, j)` is `data[offset + i*row_stride + j*col_stride]`.
    /// Strides are in elements, not bytes.
    pub fn from_strided(
        data: &'a [T],
        offset: usize,
        rows: usize,
        cols: usize,
        row_stride: isize,
        col_stride: isize,
    ) -> Result<Self, LapError> {
        if rows > 0 && cols > 0 {
            let last = |n: usize, s: isize| (n as isize - 1).checked_mul(s);
            let (ri, ci) = (
                last(rows, row_stride).ok_or(LapError::BadShape)?,
                last(cols, col_stride).ok_or(LapError::BadShape)?,
            );
            let base = offset as isize;
            for corner in [base, base + ri, base + ci, base + ri + ci] {
                if corner < 0 || corner as usize >= data.len() {
                    return Err(LapError::BadShape);
                }
            }
        }
        Ok(MatrixView {
            data,
            offset,
            rows,
            cols,
            row_stride,
            col_stride,
        })
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Transposed view (no copy).
    pub fn t(self) -> Self {
        MatrixView {
            rows: self.cols,
            cols: self.rows,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
            ..self
        }
    }

    /// Element `(i, j)`. Panics if out of range.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> T {
        assert!(i < self.rows && j < self.cols, "index out of range");
        let k = self.offset as isize + i as isize * self.row_stride + j as isize * self.col_stride;
        self.data[k as usize]
    }
}

/// An optimal assignment.
#[derive(Clone, Debug, PartialEq)]
pub struct LapSolution<A> {
    /// Matched rows, increasing.
    pub row_ind: Vec<usize>,
    /// `col_ind[k]` is the column matched to `row_ind[k]`.
    pub col_ind: Vec<usize>,
    /// Total cost of the matched pairs, in the original sign.
    pub cost: A,
}

/// Dual potentials of the last solve, in the original orientation and sign.
///
/// For a minimization, `c[i][j] - row[i] - col[j] >= 0` for all pairs, with
/// equality on matched pairs, and potentials of unmatched rows/columns are 0
/// (on the larger side). For a maximization the inequality is reversed.
#[derive(Clone, Debug, PartialEq)]
pub struct Duals<A> {
    pub row: Vec<A>,
    pub col: Vec<A>,
}

/// Reusable solver: keeps its buffers between calls, which matters for the many
/// small LAPs of the k-assignment heuristics.
#[derive(Clone, Debug, Default)]
pub struct LapSolver<A> {
    // Internal problem: nr <= nc, minimization, row-major.
    cost: Vec<A>,
    nr: usize,
    nc: usize,
    transposed: bool,
    sense: Sense,
    u: Vec<A>,
    v: Vec<A>,
    shortest: Vec<A>,
    path: Vec<usize>,
    col4row: Vec<usize>,
    row4col: Vec<usize>,
    sr: Vec<bool>,
    sc: Vec<bool>,
    remaining: Vec<usize>,
}

const NONE: usize = usize::MAX;

impl<A: Acc> LapSolver<A> {
    pub fn new() -> Self {
        LapSolver {
            cost: Vec::new(),
            nr: 0,
            nc: 0,
            transposed: false,
            sense: Sense::Minimize,
            u: Vec::new(),
            v: Vec::new(),
            shortest: Vec::new(),
            path: Vec::new(),
            col4row: Vec::new(),
            row4col: Vec::new(),
            sr: Vec::new(),
            sc: Vec::new(),
            remaining: Vec::new(),
        }
    }

    /// Solve the LAP on `m`.
    pub fn solve<T>(
        &mut self,
        m: MatrixView<'_, T>,
        sense: Sense,
    ) -> Result<LapSolution<A>, LapError>
    where
        T: Cost<Acc = A>,
    {
        self.load(m, sense)?;
        let (nr, nc) = (self.nr, self.nc);
        self.u.clear();
        self.u.resize(nr, A::ZERO);
        self.v.clear();
        self.v.resize(nc, A::ZERO);
        self.col4row.clear();
        self.col4row.resize(nr, NONE);
        self.row4col.clear();
        self.row4col.resize(nc, NONE);
        self.path.clear();
        self.path.resize(nc, NONE);
        self.shortest.resize(nc, A::INF);
        self.sr.resize(nr, false);
        self.sc.resize(nc, false);
        self.remaining.resize(nc, 0);

        for cur_row in 0..nr {
            let (sink, min_val) = self.augmenting_path(cur_row).ok_or(LapError::Infeasible)?;
            // Update duals.
            self.u[cur_row] = self.u[cur_row] + min_val;
            for i in 0..nr {
                if self.sr[i] && i != cur_row {
                    let j = self.col4row[i];
                    self.u[i] = self.u[i] + (min_val - self.shortest[j]);
                }
            }
            for j in 0..nc {
                if self.sc[j] {
                    self.v[j] = self.v[j] - (min_val - self.shortest[j]);
                }
            }
            // Augment along the path back to cur_row.
            let mut j = sink;
            loop {
                let i = self.path[j];
                self.row4col[j] = i;
                std::mem::swap(&mut self.col4row[i], &mut j);
                if i == cur_row {
                    break;
                }
            }
        }
        Ok(self.solution(&m))
    }

    /// Dual potentials of the last successful [`solve`](Self::solve).
    pub fn duals(&self) -> Duals<A> {
        let flip = |x: &A| match self.sense {
            Sense::Minimize => *x,
            Sense::Maximize => -*x,
        };
        let u: Vec<A> = self.u.iter().map(flip).collect();
        let v: Vec<A> = self.v.iter().map(flip).collect();
        if self.transposed {
            Duals { row: v, col: u }
        } else {
            Duals { row: u, col: v }
        }
    }

    fn load<T: Cost<Acc = A>>(
        &mut self,
        m: MatrixView<'_, T>,
        sense: Sense,
    ) -> Result<(), LapError> {
        self.transposed = m.rows() > m.cols();
        self.sense = sense;
        let mm = if self.transposed { m.t() } else { m };
        self.nr = mm.rows();
        self.nc = mm.cols();
        self.cost.clear();
        self.cost.reserve(self.nr * self.nc);
        for i in 0..self.nr {
            for j in 0..self.nc {
                let x = mm.get(i, j).widen();
                let x = match sense {
                    Sense::Minimize => x,
                    Sense::Maximize => -x,
                };
                if !x.is_valid() {
                    let (oi, oj) = if self.transposed { (j, i) } else { (i, j) };
                    return Err(LapError::InvalidCost(oi, oj));
                }
                self.cost.push(x);
            }
        }
        Ok(())
    }

    /// Dijkstra-like search for a shortest augmenting path from row `start`.
    /// Returns the sink column and the path length, or `None` if no finite path exists.
    fn augmenting_path(&mut self, start: usize) -> Option<(usize, A)> {
        let nc = self.nc;
        let mut i = start;
        let mut min_val = A::ZERO;
        let mut num_remaining = nc;
        // Reverse order, so that ties are broken towards lower column indices
        // the same way as scipy.
        for (it, r) in self.remaining.iter_mut().enumerate() {
            *r = nc - it - 1;
        }
        self.sr.fill(false);
        self.sc.fill(false);
        self.shortest.fill(A::INF);

        loop {
            let mut index = NONE;
            let mut lowest = A::INF;
            self.sr[i] = true;
            let row = &self.cost[i * nc..(i + 1) * nc];
            let ui = self.u[i];
            for it in 0..num_remaining {
                let j = self.remaining[it];
                let r = min_val + row[j] - ui - self.v[j];
                if r < self.shortest[j] {
                    self.path[j] = i;
                    self.shortest[j] = r;
                }
                let s = self.shortest[j];
                if s < lowest || (s == lowest && self.row4col[j] == NONE) {
                    lowest = s;
                    index = it;
                }
            }
            min_val = lowest;
            if index == NONE || min_val >= A::INF {
                return None;
            }
            let j = self.remaining[index];
            self.sc[j] = true;
            num_remaining -= 1;
            self.remaining[index] = self.remaining[num_remaining];
            if self.row4col[j] == NONE {
                return Some((j, min_val));
            }
            i = self.row4col[j];
        }
    }

    fn solution<T: Cost<Acc = A>>(&self, m: &MatrixView<'_, T>) -> LapSolution<A> {
        let mut pairs: Vec<(usize, usize)> = self
            .col4row
            .iter()
            .enumerate()
            .map(|(i, &j)| if self.transposed { (j, i) } else { (i, j) })
            .collect();
        if self.transposed {
            pairs.sort_unstable();
        }
        let mut cost = A::ZERO;
        for &(i, j) in &pairs {
            cost = cost + m.get(i, j).widen();
        }
        let (row_ind, col_ind) = pairs.into_iter().unzip();
        LapSolution {
            row_ind,
            col_ind,
            cost,
        }
    }
}

/// Solve one LAP. Use [`LapSolver`] to reuse buffers across many calls.
pub fn linear_sum_assignment<T: Cost>(
    m: MatrixView<'_, T>,
    sense: Sense,
) -> Result<LapSolution<T::Acc>, LapError> {
    LapSolver::new().solve(m, sense)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Optimal cost by enumerating injections from the smaller side.
    fn brute_force(c: &[i64], nr: usize, nc: usize, sense: Sense) -> i64 {
        fn rec(c: &[i64], nr: usize, nc: usize, i: usize, used: &mut [bool], sense: Sense) -> i64 {
            if i == nr {
                return 0;
            }
            let mut best: Option<i64> = None;
            for j in 0..nc {
                if !used[j] {
                    used[j] = true;
                    let x = c[i * nc + j] + rec(c, nr, nc, i + 1, used, sense);
                    used[j] = false;
                    best = Some(match (best, sense) {
                        (None, _) => x,
                        (Some(b), Sense::Minimize) => b.min(x),
                        (Some(b), Sense::Maximize) => b.max(x),
                    });
                }
            }
            best.unwrap()
        }
        if nr > nc {
            let t: Vec<i64> = (0..nc)
                .flat_map(|j| (0..nr).map(move |i| c[i * nc + j]))
                .collect();
            return brute_force(&t, nc, nr, sense);
        }
        rec(c, nr, nc, 0, &mut vec![false; nc], sense)
    }

    fn check_solution(sol: &LapSolution<i64>, c: &[i64], nr: usize, nc: usize) {
        assert_eq!(sol.row_ind.len(), nr.min(nc));
        assert!(sol.row_ind.windows(2).all(|w| w[0] < w[1]));
        let mut seen = vec![false; nc];
        for &j in &sol.col_ind {
            assert!(!seen[j]);
            seen[j] = true;
        }
        let total: i64 = sol
            .row_ind
            .iter()
            .zip(&sol.col_ind)
            .map(|(&i, &j)| c[i * nc + j])
            .sum();
        assert_eq!(total, sol.cost);
    }

    fn check_duals(
        d: &Duals<i64>,
        sol: &LapSolution<i64>,
        c: &[i64],
        nr: usize,
        nc: usize,
        sense: Sense,
    ) {
        let sgn = if sense == Sense::Minimize { 1 } else { -1 };
        for i in 0..nr {
            for j in 0..nc {
                assert!(
                    sgn * (c[i * nc + j] - d.row[i] - d.col[j]) >= 0,
                    "dual infeasible at ({i},{j})"
                );
            }
        }
        for (&i, &j) in sol.row_ind.iter().zip(&sol.col_ind) {
            assert_eq!(
                c[i * nc + j],
                d.row[i] + d.col[j],
                "slack on matched pair ({i},{j})"
            );
        }
        // Unmatched lines on the larger side carry zero potential.
        if nr <= nc {
            let matched: Vec<bool> = (0..nc).map(|j| sol.col_ind.contains(&j)).collect();
            (0..nc)
                .filter(|&j| !matched[j])
                .for_each(|j| assert_eq!(d.col[j], 0));
        } else {
            let matched: Vec<bool> = (0..nr).map(|i| sol.row_ind.contains(&i)).collect();
            (0..nr)
                .filter(|&i| !matched[i])
                .for_each(|i| assert_eq!(d.row[i], 0));
        }
    }

    fn matrix() -> impl Strategy<Value = (usize, usize, Vec<i64>)> {
        (0usize..=6, 0usize..=6).prop_flat_map(|(nr, nc)| {
            // Narrow value range to produce many ties.
            (
                Just(nr),
                Just(nc),
                prop::collection::vec(-5i64..=5, nr * nc),
            )
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        #[test]
        fn matches_brute_force((nr, nc, c) in matrix(), maximize in any::<bool>()) {
            let sense = if maximize { Sense::Maximize } else { Sense::Minimize };
            let mut solver = LapSolver::new();
            let sol = solver.solve(MatrixView::from_row_major(&c, nr, nc).unwrap(), sense).unwrap();
            check_solution(&sol, &c, nr, nc);
            prop_assert_eq!(sol.cost, brute_force(&c, nr, nc, sense));
            check_duals(&solver.duals(), &sol, &c, nr, nc, sense);
        }

        #[test]
        fn transposed_view_is_consistent((nr, nc, c) in matrix()) {
            let m = MatrixView::from_row_major(&c, nr, nc).unwrap();
            let a = linear_sum_assignment(m, Sense::Minimize).unwrap();
            let b = linear_sum_assignment(m.t(), Sense::Minimize).unwrap();
            prop_assert_eq!(a.cost, b.cost);
        }

        #[test]
        fn float_matches_int((nr, nc, c) in matrix()) {
            let f: Vec<f64> = c.iter().map(|&x| x as f64 * 0.5).collect();
            let a = linear_sum_assignment(MatrixView::from_row_major(&c, nr, nc).unwrap(), Sense::Minimize).unwrap();
            let b = linear_sum_assignment(MatrixView::from_row_major(&f, nr, nc).unwrap(), Sense::Minimize).unwrap();
            prop_assert_eq!(a.cost as f64 * 0.5, b.cost);
        }

        #[test]
        fn wide_random_values((nr, nc) in (1usize..=6, 1usize..=6), seed in any::<u64>()) {
            // Distinct-ish large values exercise the dual arithmetic.
            let mut s = seed;
            let c: Vec<i64> = (0..nr * nc).map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as i64 % 2_000_001) - 1_000_000
            }).collect();
            let sol = linear_sum_assignment(MatrixView::from_row_major(&c, nr, nc).unwrap(), Sense::Minimize).unwrap();
            prop_assert_eq!(sol.cost, brute_force(&c, nr, nc, Sense::Minimize));
        }
    }

    #[test]
    fn scipy_example() {
        let c = [4i64, 1, 3, 2, 0, 5, 3, 2, 2];
        let sol = linear_sum_assignment(
            MatrixView::from_row_major(&c, 3, 3).unwrap(),
            Sense::Minimize,
        )
        .unwrap();
        assert_eq!(sol.row_ind, vec![0, 1, 2]);
        assert_eq!(sol.col_ind, vec![1, 0, 2]);
        assert_eq!(sol.cost, 5);
    }

    #[test]
    fn empty() {
        let c: [f64; 0] = [];
        for (nr, nc) in [(0, 0), (0, 3), (3, 0)] {
            let sol = linear_sum_assignment(
                MatrixView::from_row_major(&c, nr, nc).unwrap(),
                Sense::Minimize,
            )
            .unwrap();
            assert!(sol.row_ind.is_empty() && sol.col_ind.is_empty());
            assert_eq!(sol.cost, 0.0);
        }
    }

    #[test]
    fn forbidden_entries() {
        let inf = f64::INFINITY;
        let c = [inf, 1.0, 2.0, inf];
        let sol = linear_sum_assignment(
            MatrixView::from_row_major(&c, 2, 2).unwrap(),
            Sense::Minimize,
        )
        .unwrap();
        assert_eq!(sol.col_ind, vec![1, 0]);
        assert_eq!(sol.cost, 3.0);

        let c = [inf, 1.0, inf, 2.0];
        let r = linear_sum_assignment(
            MatrixView::from_row_major(&c, 2, 2).unwrap(),
            Sense::Minimize,
        );
        assert_eq!(r, Err(LapError::Infeasible));

        let ninf = f64::NEG_INFINITY;
        let c = [ninf, 1.0, 2.0, ninf];
        let sol = linear_sum_assignment(
            MatrixView::from_row_major(&c, 2, 2).unwrap(),
            Sense::Maximize,
        )
        .unwrap();
        assert_eq!(sol.cost, 3.0);
    }

    #[test]
    fn invalid_entries() {
        let c = [1.0, f64::NAN, 0.0, 1.0, 2.0, 3.0];
        let m = MatrixView::from_row_major(&c, 3, 2).unwrap();
        assert_eq!(
            linear_sum_assignment(m, Sense::Minimize),
            Err(LapError::InvalidCost(0, 1))
        );
        let c = [f64::NEG_INFINITY, 0.0];
        let m = MatrixView::from_row_major(&c, 1, 2).unwrap();
        assert_eq!(
            linear_sum_assignment(m, Sense::Minimize),
            Err(LapError::InvalidCost(0, 0))
        );
        let c = [i64::MAX, 0];
        let m = MatrixView::from_row_major(&c, 1, 2).unwrap();
        assert_eq!(
            linear_sum_assignment(m, Sense::Minimize),
            Err(LapError::InvalidCost(0, 0))
        );
    }

    #[test]
    fn strided_views() {
        // 3x4 matrix stored column-major inside a larger buffer, read reversed.
        let rm = [7i64, 3, 9, 1, 4, 8, 2, 6, 5, 0, 7, 3];
        let (nr, nc) = (3, 4);
        let mut buf = vec![99i64; 2 + nr * nc];
        for i in 0..nr {
            for j in 0..nc {
                buf[2 + j * nr + i] = rm[i * nc + j];
            }
        }
        let cm = MatrixView::from_strided(&buf, 2, nr, nc, 1, nr as isize).unwrap();
        let a = linear_sum_assignment(cm, Sense::Minimize).unwrap();
        let b = linear_sum_assignment(
            MatrixView::from_row_major(&rm, nr, nc).unwrap(),
            Sense::Minimize,
        )
        .unwrap();
        assert_eq!(a, b);
        // Rows reversed via a negative stride.
        let rev = MatrixView::from_strided(&rm, 2 * nc, nr, nc, -(nc as isize), 1).unwrap();
        assert_eq!(rev.get(0, 0), 5);
        assert_eq!(
            linear_sum_assignment(rev, Sense::Minimize).unwrap().cost,
            b.cost
        );
        assert_eq!(
            MatrixView::from_strided(&rm, 0, nr, nc, -(nc as isize), 1).unwrap_err(),
            LapError::BadShape
        );
        assert_eq!(
            MatrixView::from_row_major(&rm, 4, 4).unwrap_err(),
            LapError::BadShape
        );
    }

    #[test]
    fn solver_reuse_across_shapes() {
        let mut s = LapSolver::new();
        let a = [1i64, 2, 3, 4, 5, 6];
        let b = [5i64, 1, 1, 5];
        assert_eq!(
            s.solve(
                MatrixView::from_row_major(&a, 2, 3).unwrap(),
                Sense::Minimize
            )
            .unwrap()
            .cost,
            6
        );
        assert_eq!(
            s.solve(
                MatrixView::from_row_major(&a, 3, 2).unwrap(),
                Sense::Maximize
            )
            .unwrap()
            .cost,
            9
        );
        assert_eq!(
            s.solve(
                MatrixView::from_row_major(&b, 2, 2).unwrap(),
                Sense::Minimize
            )
            .unwrap()
            .cost,
            2
        );
    }
}
