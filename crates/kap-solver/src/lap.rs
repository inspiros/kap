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
//!
//! Square matrices without forbidden entries are first reduced with the
//! Jonker–Volgenant (1987) initialisation (column reduction, reduction
//! transfer, augmenting row reduction); the remaining rows are augmented by
//! shortest paths. The optimum is the same either way, but ties may be broken
//! differently from scipy; use [`LapSolver::with_jv_init`]`(false)` to
//! reproduce scipy's assignment exactly.

// The hot loops index several parallel arrays by column; ranges whose bound is
// updated inside the loop follow LAPJV, where the bound is read once.
#![allow(clippy::needless_range_loop, clippy::mut_range_bound)]

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
    /// Convert a finite threshold without rounding or saturating integer casts.
    fn from_f64_exact(x: f64) -> Option<Self>;
    /// Whether the value is finite (always true for integer accumulators).
    fn is_finite(self) -> bool;
    /// Whether `self` is a permitted entry of a minimization problem
    /// (finite or `+INF` for floats).
    fn is_valid(self) -> bool;
    /// Add without integer overflow; floating-point sums use ordinary rounding.
    fn checked_add(self, rhs: Self) -> Option<Self> {
        Some(self + rhs)
    }
    /// Negate without integer overflow.
    fn checked_neg(self) -> Option<Self> {
        Some(-self)
    }
}

impl Acc for f64 {
    const ZERO: Self = 0.0;
    const INF: Self = f64::INFINITY;
    fn from_f64_exact(x: f64) -> Option<Self> {
        x.is_finite().then_some(x)
    }
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    #[inline]
    fn is_valid(self) -> bool {
        !self.is_nan() && self != f64::NEG_INFINITY
    }
}

impl Acc for i64 {
    const ZERO: Self = 0;
    const INF: Self = i64::MAX;
    fn from_f64_exact(x: f64) -> Option<Self> {
        // i64::MAX rounds to 2^63 as f64, so the upper bound is exclusive.
        (x.is_finite() && x.fract() == 0.0 && x >= i64::MIN as f64 && x < -(i64::MIN as f64))
            .then_some(x as i64)
    }
    fn is_finite(self) -> bool {
        true
    }
    #[inline]
    fn is_valid(self) -> bool {
        // Leave headroom so that `-x` and the dual updates cannot overflow
        // for any single entry.
        self > i64::MIN / 4 && self < i64::MAX / 4
    }
    fn checked_add(self, rhs: Self) -> Option<Self> {
        self.checked_add(rhs)
    }
    fn checked_neg(self) -> Option<Self> {
        self.checked_neg()
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
    // Jonker-Volgenant initialisation (square, no forbidden entries).
    jv_mode: JvMode,
    has_forbidden: bool,
    free: Vec<usize>,
    matches: Vec<u32>,
    collist: Vec<usize>,
}

const NONE: usize = usize::MAX;

/// How square matrices are solved (experimental knob).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum JvMode {
    Off,
    Lapjv,
    InitThenSap,
    /// JV initialisation, then LAPJV-style augmentation with relaxation and
    /// minimum search fused into one pass per scanned row.
    #[default]
    Fused,
}

/// Checked in debug builds only: guards the unchecked indexing in hot loops.
#[inline(always)]
fn assert_in_range(j: usize, n: usize) {
    debug_assert!(j < n, "index {j} out of range {n}");
}

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
            jv_mode: JvMode::default(),
            has_forbidden: false,
            free: Vec::new(),
            matches: Vec::new(),
            collist: Vec::new(),
        }
    }

    /// Enable or disable the Jonker-Volgenant initialisation (column reduction,
    /// reduction transfer, augmenting row reduction) used for square matrices
    /// without forbidden entries. Enabled by default. When disabled, or not
    /// applicable, every row is augmented by shortest path as in scipy, and
    /// ties are broken exactly like scipy.
    pub fn with_jv_init(mut self, on: bool) -> Self {
        self.jv_mode = if on { JvMode::default() } else { JvMode::Off };
        self
    }

    #[doc(hidden)]
    pub fn with_jv_mode(mut self, mode: JvMode) -> Self {
        self.jv_mode = mode;
        self
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

        let jv = nr == nc && nr >= 2 && !self.has_forbidden;
        match self.jv_mode {
            JvMode::Lapjv if jv => {
                self.jv_init();
                self.jv_augment();
                return Ok(self.solution(&m));
            }
            JvMode::Fused if jv => {
                self.jv_init();
                self.jv_augment_fused();
                return Ok(self.solution(&m));
            }
            JvMode::InitThenSap if jv => {
                self.jv_init();
                self.jv_row_duals();
                let free = std::mem::take(&mut self.free);
                let r = self.sap_rows(free.iter().copied());
                self.free = free;
                r?;
                return Ok(self.solution(&m));
            }
            _ => self.sap_rows(0..nr)?,
        }
        Ok(self.solution(&m))
    }

    /// Crouse's shortest augmenting path for each of `rows` (free rows with
    /// feasible duals `u`, `v`; matched pairs tight).
    fn sap_rows(&mut self, rows: impl Iterator<Item = usize>) -> Result<(), LapError> {
        let (nr, nc) = (self.nr, self.nc);
        for cur_row in rows {
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
        Ok(())
    }

    /// Row potentials after `jv_init`: tight for matched rows, the row minimum
    /// of the reduced costs for free rows.
    fn jv_row_duals(&mut self) {
        let n = self.nr;
        for i in 0..n {
            let row = &self.cost[i * n..(i + 1) * n];
            self.u[i] = if self.col4row[i] != NONE {
                row[self.col4row[i]] - self.v[self.col4row[i]]
            } else {
                let mut mn = A::INF;
                for j in 0..n {
                    let h = row[j] - self.v[j];
                    if h < mn {
                        mn = h;
                    }
                }
                mn
            };
        }
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
        let (nr, nc) = (self.nr, self.nc);
        let neg = sense == Sense::Maximize;
        let conv = |x: T| {
            let a = x.widen();
            if neg { -a } else { a }
        };
        self.cost.clear();
        self.cost.resize(nr * nc, A::ZERO);
        let dst = &mut self.cost[..];
        let (data, off) = (mm.data, mm.offset as isize);
        if nr > 0 && nc > 0 && mm.col_stride == 1 {
            // Rows contiguous in the source.
            for (i, out) in dst.chunks_exact_mut(nc).enumerate() {
                let start = (off + i as isize * mm.row_stride) as usize;
                for (o, &x) in out.iter_mut().zip(&data[start..start + nc]) {
                    *o = conv(x);
                }
            }
        } else if nr > 0 && nc > 0 && mm.row_stride == 1 {
            // Columns contiguous (e.g. a transposed C-contiguous matrix):
            // transpose in tiles to keep both sides in cache.
            const B: usize = 32;
            for jb in (0..nc).step_by(B) {
                for ib in (0..nr).step_by(B) {
                    let ie = (ib + B).min(nr);
                    for j in jb..(jb + B).min(nc) {
                        let start = (off + j as isize * mm.col_stride) as usize;
                        for (i, &x) in (ib..ie).zip(&data[start + ib..start + ie]) {
                            dst[i * nc + j] = conv(x);
                        }
                    }
                }
            }
        } else {
            for i in 0..nr {
                for j in 0..nc {
                    dst[i * nc + j] = conv(mm.get(i, j));
                }
            }
        }
        // Branch-free validity scan; locate the first bad entry only on failure.
        let (mut bad, mut forbidden) = (false, false);
        for &x in dst.iter() {
            bad |= !x.is_valid();
            forbidden |= x >= A::INF;
        }
        if bad {
            let k = dst.iter().position(|x| !x.is_valid()).unwrap();
            let (i, j) = (k / nc, k % nc);
            let (oi, oj) = if self.transposed { (j, i) } else { (i, j) };
            return Err(LapError::InvalidCost(oi, oj));
        }
        self.has_forbidden = forbidden;
        Ok(())
    }

    /// Jonker & Volgenant (1987) LAPJV for a square matrix with finite entries:
    /// column reduction, reduction transfer and two augmenting row reduction
    /// passes. Fills `col4row`, `row4col`, `v` and leaves the unassigned rows
    /// in `free`; matched rows are tight.
    fn jv_init(&mut self) {
        let LapSolver {
            cost: c,
            nr,
            v,
            path: pred,
            col4row: x,
            row4col: y,
            free,
            matches,
            collist,
            ..
        } = self;
        let n = *nr;
        matches.clear();
        matches.resize(n, 0);
        collist.clear();
        collist.resize(n, 0);

        // Column reduction: v[j] = min_i c[i][j], first minimal row wins.
        // Scanned row by row for cache locality; `pred` holds the argmin.
        v.fill(A::INF);
        for i in 0..n {
            let row = &c[i * n..(i + 1) * n];
            for j in 0..n {
                if row[j] < v[j] {
                    v[j] = row[j];
                    pred[j] = i;
                }
            }
        }
        for j in (0..n).rev() {
            let i = pred[j];
            matches[i] += 1;
            if matches[i] == 1 {
                x[i] = j;
                y[j] = i;
            } else if v[j] < v[x[i]] {
                let j1 = x[i];
                x[i] = j;
                y[j] = i;
                y[j1] = NONE;
            }
        }

        // Reduction transfer from rows assigned once.
        free.clear();
        for i in 0..n {
            if matches[i] == 0 {
                free.push(i);
            } else if matches[i] == 1 {
                let j1 = x[i];
                let row = &c[i * n..(i + 1) * n];
                let vv = &v[..n];
                let mut mn = A::INF;
                for j in 0..n {
                    let h = row[j] - vv[j];
                    if j != j1 && h < mn {
                        mn = h;
                    }
                }
                v[j1] = v[j1] - mn;
            }
        }

        // Augmenting row reduction, two passes. A row displaced after a strict
        // price decrease is retried at once; the budget bounds that (with
        // floats, near-ties can make it creep), and rows over budget are left
        // to the shortest path phase, which is always correct.
        for _ in 0..2 {
            let prev = free.len();
            if prev == 0 {
                break;
            }
            let mut budget = 8 * n;
            let (mut k, mut nf) = (0usize, 0usize);
            while k < prev {
                let i = free[k];
                k += 1;
                let row = &c[i * n..(i + 1) * n];
                let vv = &v[..n];
                let mut umin = row[0] - vv[0];
                let mut j1 = 0;
                let mut j2 = NONE;
                let mut usub = A::INF;
                for j in 1..n {
                    let h = row[j] - vv[j];
                    if h < usub {
                        if h >= umin {
                            usub = h;
                            j2 = j;
                        } else {
                            usub = umin;
                            umin = h;
                            j2 = j1;
                            j1 = j;
                        }
                    }
                }
                let mut i0 = y[j1];
                let strict = umin < usub;
                if strict {
                    v[j1] = v[j1] - (usub - umin);
                } else if i0 != NONE {
                    j1 = j2;
                    i0 = y[j2];
                }
                x[i] = j1;
                y[j1] = i;
                if i0 != NONE {
                    x[i0] = NONE;
                    if strict && budget > 0 {
                        budget -= 1;
                        k -= 1;
                        free[k] = i0;
                    } else {
                        free[nf] = i0;
                        nf += 1;
                    }
                }
            }
            free.truncate(nf);
        }
    }

    /// LAPJV's augmentation phase for the rows left in `free` by `jv_init`.
    fn jv_augment(&mut self) {
        let LapSolver {
            cost: c,
            nr,
            u,
            v,
            shortest: d,
            path: pred,
            col4row: x,
            row4col: y,
            free,
            collist,
            ..
        } = self;
        let n = *nr;
        // Shortest augmenting paths (Dijkstra on reduced costs, with the
        // row potentials implicit: u[i] = c[i][x[i]] - v[x[i]]).
        // collist: [0, low) ready, [low, up) at the current minimum, [up, n) todo.
        for &fr in free.iter() {
            let row = &c[fr * n..(fr + 1) * n];
            for j in 0..n {
                d[j] = row[j] - v[j];
                pred[j] = fr;
                collist[j] = j;
            }
            let (mut low, mut up, mut ready) = (0usize, 0usize, 0usize);
            let mut mn = A::ZERO;
            let endofpath = 'search: loop {
                if up == low {
                    ready = low;
                    mn = d[collist[up]];
                    up += 1;
                    for k in up..n {
                        let j = collist[k];
                        assert_in_range(j, n);
                        // SAFETY: collist is a permutation of 0..n; d has length n.
                        let h = unsafe { *d.get_unchecked(j) };
                        if h <= mn {
                            if h < mn {
                                up = low;
                                mn = h;
                            }
                            collist[k] = collist[up];
                            collist[up] = j;
                            up += 1;
                        }
                    }
                    for &j in &collist[low..up] {
                        if y[j] == NONE {
                            break 'search j;
                        }
                    }
                }
                let j1 = collist[low];
                low += 1;
                let i = y[j1];
                let rowi = &c[i * n..(i + 1) * n];
                let h = rowi[j1] - v[j1] - mn;
                let (vv, dd, pp, yy) = (&v[..n], &mut d[..n], &mut pred[..n], &y[..n]);
                for k in up..n {
                    let j = collist[k];
                    assert_in_range(j, n);
                    // SAFETY: collist is a permutation of 0..n; all slices have length n.
                    unsafe {
                        let v2 = *rowi.get_unchecked(j) - *vv.get_unchecked(j) - h;
                        if v2 < *dd.get_unchecked(j) {
                            *pp.get_unchecked_mut(j) = i;
                            if v2 == mn {
                                if *yy.get_unchecked(j) == NONE {
                                    break 'search j;
                                }
                                collist.swap(k, up);
                                up += 1;
                            }
                            *dd.get_unchecked_mut(j) = v2;
                        }
                    }
                }
            };
            for &j in &collist[..ready] {
                v[j] = v[j] + (d[j] - mn);
            }
            let mut j = endofpath;
            loop {
                let i = pred[j];
                y[j] = i;
                std::mem::swap(&mut x[i], &mut j);
                if i == fr {
                    break;
                }
            }
        }
        for i in 0..n {
            u[i] = c[i * n + x[i]] - v[x[i]];
        }
    }

    /// Augmentation phase for the rows left in `free` by `jv_init`: LAPJV's
    /// column list and implicit row potentials, with the search for the next
    /// minimum fused into the relaxation pass (as in Crouse's method) whenever
    /// the row being scanned is the last one at the current distance. A row
    /// that still has others waiting behind it is relaxed without tracking
    /// the minimum, as in LAPJV, since a later pass would overwrite it.
    fn jv_augment_fused(&mut self) {
        let LapSolver {
            cost: c,
            nr,
            u,
            v,
            shortest: d,
            path: pred,
            col4row: x,
            row4col: y,
            free,
            collist,
            ..
        } = self;
        let n = *nr;
        let (d, pred, collist) = (&mut d[..n], &mut pred[..n], &mut collist[..n]);
        // collist: [0, ready) scanned below the current minimum `mn`,
        // [ready, low) scanned at it, [low, up) to scan at it, [up, n) todo.
        for &fr in free.iter() {
            let row = &c[fr * n..(fr + 1) * n];
            // `lowest` is the least distance of a todo column and `idx` the
            // position in collist of one that attains it, a free column
            // being preferred among equals.
            let mut lowest = A::INF;
            let mut idx = 0;
            {
                let (vv, yy) = (&v[..n], &y[..n]);
                for j in 0..n {
                    d[j] = row[j] - vv[j];
                    pred[j] = fr;
                    collist[j] = j;
                }
                for j in 0..n {
                    let h = d[j];
                    if h < lowest || (h == lowest && yy[j] == NONE) {
                        lowest = h;
                        idx = j;
                    }
                }
            }
            let (mut low, mut up) = (0usize, 0usize);
            let (mut ready, mut mn);
            let endofpath = 'search: loop {
                // The scan set is empty and the last pass was a fused one, so
                // `lowest` and `idx` are current: next distance level.
                debug_assert!(low == up && idx >= up && idx < n);
                ready = low;
                mn = lowest;
                let j0 = collist[idx];
                if y[j0] == NONE {
                    break 'search j0;
                }
                collist[idx] = collist[up];
                collist[up] = j0;
                up += 1;
                while low < up {
                    let j1 = collist[low];
                    low += 1;
                    let i = y[j1];
                    let rowi = &c[i * n..(i + 1) * n];
                    let h = rowi[j1] - v[j1] - mn;
                    let (vv, yy) = (&v[..n], &y[..n]);
                    if low < up {
                        // Other rows wait at this distance: relax only.
                        for k in up..n {
                            // SAFETY: collist is a permutation of 0..n, so
                            // j < n and up <= k < n; all slices have length n.
                            unsafe {
                                let j = *collist.get_unchecked(k);
                                assert_in_range(j, n);
                                let v2 = *rowi.get_unchecked(j) - *vv.get_unchecked(j) - h;
                                if v2 < *d.get_unchecked(j) {
                                    *pred.get_unchecked_mut(j) = i;
                                    *d.get_unchecked_mut(j) = v2;
                                    if v2 <= mn {
                                        if *yy.get_unchecked(j) == NONE {
                                            break 'search j;
                                        }
                                        assert_in_range(up, n);
                                        *collist.get_unchecked_mut(k) = *collist.get_unchecked(up);
                                        *collist.get_unchecked_mut(up) = j;
                                        up += 1;
                                    }
                                }
                            }
                        }
                        continue;
                    }
                    // Last row at this distance: relax and find the minimum
                    // of the todo columns in the same pass.
                    lowest = A::INF;
                    idx = NONE;
                    for k in up..n {
                        // SAFETY: as above.
                        unsafe {
                            let j = *collist.get_unchecked(k);
                            assert_in_range(j, n);
                            let v2 = *rowi.get_unchecked(j) - *vv.get_unchecked(j) - h;
                            let mut dj = *d.get_unchecked(j);
                            if v2 < dj {
                                *pred.get_unchecked_mut(j) = i;
                                *d.get_unchecked_mut(j) = v2;
                                dj = v2;
                            }
                            if dj <= lowest {
                                if dj <= mn {
                                    // Ties with the current minimum: scan it
                                    // at this level too.
                                    if *yy.get_unchecked(j) == NONE {
                                        break 'search j;
                                    }
                                    // The column at `up` was already visited
                                    // and moves to k; follow it if it is the
                                    // todo minimum.
                                    if idx == up {
                                        idx = k;
                                    }
                                    assert_in_range(up, n);
                                    *collist.get_unchecked_mut(k) = *collist.get_unchecked(up);
                                    *collist.get_unchecked_mut(up) = j;
                                    up += 1;
                                } else if dj < lowest || *yy.get_unchecked(j) == NONE {
                                    lowest = dj;
                                    idx = k;
                                }
                            }
                        }
                    }
                }
            };
            for &j in &collist[..ready] {
                v[j] = v[j] + (d[j] - mn);
            }
            let mut j = endofpath;
            loop {
                let i = pred[j];
                y[j] = i;
                std::mem::swap(&mut x[i], &mut j);
                if i == fr {
                    break;
                }
            }
        }
        for i in 0..n {
            u[i] = c[i * n + x[i]] - v[x[i]];
        }
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
            let (v, shortest, path, row4col) = (
                &self.v[..nc],
                &mut self.shortest[..nc],
                &mut self.path[..nc],
                &self.row4col[..nc],
            );
            for (it, &j) in self.remaining[..num_remaining].iter().enumerate() {
                assert_in_range(j, nc);
                // SAFETY: `remaining` holds column indices < nc, and every
                // slice indexed here has length nc.
                unsafe {
                    let r = min_val + *row.get_unchecked(j) - ui - *v.get_unchecked(j);
                    let sj = shortest.get_unchecked_mut(j);
                    if r < *sj {
                        *path.get_unchecked_mut(j) = i;
                        *sj = r;
                    }
                    let s = *sj;
                    if s < lowest || (s == lowest && *row4col.get_unchecked(j) == NONE) {
                        lowest = s;
                        index = it;
                    }
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

    /// Values in `[-range, range]` from a fixed LCG (fast, reproducible).
    fn lcg_matrix(seed: u64, len: usize, range: i64) -> Vec<i64> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as i64 % (2 * range + 1)) - range
            })
            .collect()
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
        fn square_jv_matches_brute_force(n in 2usize..=7, seed in any::<u64>(), range in 1i64..=1000, maximize in any::<bool>()) {
            let sense = if maximize { Sense::Maximize } else { Sense::Minimize };
            let c = lcg_matrix(seed, n * n, range);
            let best = brute_force(&c, n, n, sense);
            for mode in [JvMode::Fused, JvMode::Lapjv, JvMode::InitThenSap] {
                let mut solver = LapSolver::new().with_jv_mode(mode);
                let sol = solver.solve(MatrixView::from_row_major(&c, n, n).unwrap(), sense).unwrap();
                check_solution(&sol, &c, n, n);
                prop_assert_eq!(sol.cost, best, "{:?}", mode);
                check_duals(&solver.duals(), &sol, &c, n, n, sense);
            }
        }

        #[test]
        fn square_jv_matches_sap(n in 2usize..=80, seed in any::<u64>(), range in 1i64..=100_000) {
            let c = lcg_matrix(seed, n * n, range);
            let m = MatrixView::from_row_major(&c, n, n).unwrap();
            let b = LapSolver::new().with_jv_init(false).solve(m, Sense::Minimize).unwrap();
            for mode in [JvMode::Fused, JvMode::Lapjv, JvMode::InitThenSap] {
                let mut jv = LapSolver::new().with_jv_mode(mode);
                let a = jv.solve(m, Sense::Minimize).unwrap();
                check_solution(&a, &c, n, n);
                prop_assert_eq!(a.cost, b.cost, "{:?}", mode);
                check_duals(&jv.duals(), &a, &c, n, n, Sense::Minimize);
            }
            // Floats: same optimum up to rounding.
            let f: Vec<f64> = c.iter().map(|&x| x as f64 / 7.0).collect();
            let fm = MatrixView::from_row_major(&f, n, n).unwrap();
            let fa = LapSolver::new().solve(fm, Sense::Minimize).unwrap();
            let fb = LapSolver::new().with_jv_init(false).solve(fm, Sense::Minimize).unwrap();
            prop_assert!((fa.cost - fb.cost).abs() <= 1e-9 * (1.0 + fb.cost.abs()));
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
