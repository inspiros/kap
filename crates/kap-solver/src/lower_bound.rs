//! Axial Lagrangian bounds, averaged over triples for four or more parts.
//!
//! Only equal part sizes are supported: restricting a solution with unequal
//! sizes need not give a maximum-cardinality matching on each pair.

use crate::{
    Acc, Algo, Cost, KPartite, KapError, LapError, LapSolver, MatrixView, Options, Sense,
    k_assignment,
};

/// Settings for [`lower_bound`], with `A` the cost accumulator (`i64` or `f64`).
#[derive(Clone, Copy, Debug)]
pub struct LowerBoundOptions<A = f64> {
    /// Optimization direction; maximization returns an upper bound.
    pub sense: Sense,
    /// Maximum subgradient updates per triple (default 300).
    /// Zero still evaluates the zero and pairwise-dual multiplier seeds.
    pub max_iter: usize,
    /// Optional upper bound on the *whole* optimum, in the original sign.
    /// This is a numerical upper bound in either sense, including maximization.
    /// It only clamps the final result; minimization can also exit when the
    /// pairwise lower bound reaches it. Never used as a triple's Polyak target.
    /// The caller is responsible for its validity.
    pub upper: Option<A>,
}

impl<A> Default for LowerBoundOptions<A> {
    fn default() -> Self {
        Self {
            sense: Sense::Minimize,
            max_iter: 300,
            upper: None,
        }
    }
}

fn overflow() -> KapError {
    KapError::Unsupported("lower_bound arithmetic exceeds accumulator headroom")
}

/// Arithmetic for the crate's two accumulator types.
///
/// This bound supplements [`Acc`] without changing the LAP API. Integer
/// evaluations and division never pass through floating point; only the
/// heuristic multiplier *updates* use `f64`.
#[doc(hidden)]
pub trait LowerBoundAcc: Acc + Cost<Acc = Self> {
    fn lb_add(self, rhs: Self) -> Result<Self, KapError>;
    fn lb_sub(self, rhs: Self) -> Result<Self, KapError>;
    fn lb_neg(self) -> Result<Self, KapError>;
    fn lb_ceil_div(self, divisor: usize) -> Self;
    fn lb_float(self) -> f64;
    fn lb_finite(self) -> bool;
    fn lb_lap_safe(self, n: usize, factor: usize) -> bool;
    fn lb_update(self, delta: f64) -> Option<Self>;
}

impl LowerBoundAcc for i64 {
    fn lb_add(self, rhs: Self) -> Result<Self, KapError> {
        self.checked_add(rhs).ok_or_else(overflow)
    }
    fn lb_sub(self, rhs: Self) -> Result<Self, KapError> {
        self.checked_sub(rhs).ok_or_else(overflow)
    }
    fn lb_neg(self) -> Result<Self, KapError> {
        self.checked_neg().ok_or_else(overflow)
    }
    fn lb_ceil_div(self, divisor: usize) -> Self {
        let d = divisor as i64;
        self / d + i64::from(self % d > 0)
    }
    fn lb_float(self) -> f64 {
        self as f64
    }
    fn lb_finite(self) -> bool {
        true
    }
    fn lb_lap_safe(self, n: usize, factor: usize) -> bool {
        // LAP and the internal Am + 1DV heuristic use unchecked accumulator
        // arithmetic. Reserve space for path lengths, potentials and merges.
        let limit = i64::MAX as u128 / (n.max(1) as u128 * factor as u128);
        self.unsigned_abs() as u128 <= limit
    }
    fn lb_update(self, delta: f64) -> Option<Self> {
        let d = delta.round();
        if d.is_finite() && d >= i64::MIN as f64 && d < i64::MAX as f64 {
            self.checked_add(d as i64)
        } else {
            None
        }
    }
}

impl LowerBoundAcc for f64 {
    fn lb_add(self, rhs: Self) -> Result<Self, KapError> {
        let value = self + rhs;
        if value.is_nan() || (self.is_finite() && rhs.is_finite() && !value.is_finite()) {
            Err(overflow())
        } else {
            Ok(value)
        }
    }
    fn lb_sub(self, rhs: Self) -> Result<Self, KapError> {
        self.lb_add(-rhs)
    }
    fn lb_neg(self) -> Result<Self, KapError> {
        Ok(-self)
    }
    fn lb_ceil_div(self, divisor: usize) -> Self {
        self / divisor as f64
    }
    fn lb_float(self) -> f64 {
        self
    }
    fn lb_finite(self) -> bool {
        self.is_finite()
    }
    fn lb_lap_safe(self, n: usize, factor: usize) -> bool {
        self == Self::INFINITY || self.abs() <= f64::MAX / (n.max(1) as f64 * factor as f64)
    }
    fn lb_update(self, delta: f64) -> Option<Self> {
        let value = self + delta;
        value.is_finite().then_some(value)
    }
}

/// Bound the optimum of a condensed k-partite problem (same order as
/// [`k_assignment`]), using O(n²) scratch space per active triple.
///
/// For two parts this is the exact LAP optimum. For three parts it returns the
/// best evaluated axial Lagrangian bound, with integral multipliers for integer
/// costs. For larger k it averages all triple bounds and rounds integer results
/// upward, *before* restoring the sign for maximization (giving a floor).
/// Floating-point bounds are exact up to rounding.
///
/// Each triple starts with the pairwise LAP sum and evaluates both `u = 0` and
/// a seed from the column duals of its second and third matrices. At `u = 0`,
/// the third part may be reused freely: that bound alone can be weaker than
/// the pairwise sum. The best bound is monotone in `max_iter` (decreasing for
/// maximization). Am + 1DV provides a deterministic, triple-local Polyak target;
/// validity never depends on that target being optimal. If forbidden entries
/// prevent the heuristic from finding a feasible target, the seeds still give
/// valid bounds and subgradient updates are skipped.
///
/// # Errors
/// Follows [`KPartite::new`] for invalid matrix counts/shapes: zero matrices
/// (including an unrepresentable one-part problem) return [`KapError::BadCount`].
/// Equal zero-sized parts return zero. Unequal sizes are unsupported. Invalid
/// costs follow LAP's infinity/sign rules. Excessive magnitudes or sums return
/// an error rather than overflowing. A multiplier step that exhausts arithmetic
/// headroom stops that triple and retains its best previously evaluated bound.
pub fn lower_bound<T: Cost>(
    mats: &[MatrixView<'_, T>],
    opts: &LowerBoundOptions<T::Acc>,
) -> Result<T::Acc, KapError>
where
    T::Acc: LowerBoundAcc,
{
    let problem = KPartite::new(mats)?;
    let sizes = problem.sizes();
    let (k, n) = (sizes.len(), sizes[0]);
    if sizes.iter().any(|&size| size != n) {
        return Err(KapError::Unsupported(
            "lower_bound requires equal part sizes",
        ));
    }
    drop(problem);
    if opts.upper.is_some_and(|u| !u.lb_finite()) {
        return Err(KapError::Unsupported("lower_bound upper must be finite"));
    }
    let maximize = opts.sense == Sense::Maximize;
    // Widen and normalize once. Checked negation protects i64::MIN.
    let data: Vec<Vec<T::Acc>> = mats
        .iter()
        .map(|m| {
            (0..n)
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .map(|(i, j)| {
                    let x = m.get(i, j).widen();
                    let x = if maximize { x.lb_neg()? } else { x };
                    // For k=2 use LAP's documented 2*n-entry headroom. Triple
                    // heuristics and multiplier arithmetic reserve more space.
                    if !x.is_valid() || !x.lb_lap_safe(n, if k == 2 { 2 } else { 128 }) {
                        return Err(KapError::Lap(LapError::InvalidCost(i, j)));
                    }
                    Ok(x)
                })
                .collect()
        })
        .collect::<Result<_, _>>()?;
    let views: Vec<_> = data.iter().map(|d| view(d, n)).collect();
    let mut solver = LapSolver::new();
    let mut pairs = Vec::with_capacity(mats.len());
    let mut columns = Vec::with_capacity(mats.len());
    let mut pairwise = T::Acc::ZERO;
    for &m in &views {
        let cost = solver.solve(m, Sense::Minimize)?.cost;
        pairwise = pairwise.lb_add(cost)?;
        pairs.push(cost);
        columns.push(solver.duals().col);
    }
    if k == 2 || n == 0 {
        return finish(pairwise, opts);
    }
    if !maximize && opts.upper.is_some_and(|upper| pairwise >= upper) {
        return Ok(opts.upper.unwrap());
    }
    let mut triples = Vec::new();
    for p in 0..k {
        for q in p + 1..k {
            for r in q + 1..k {
                triples.push([index(k, p, q), index(k, p, r), index(k, q, r)]);
            }
        }
    }
    let solve = |ids: &[usize; 3]| {
        let [a, b, c] = *ids;
        let floor = pairs[a].lb_add(pairs[b])?.lb_add(pairs[c])?;
        let u: Vec<_> = columns[b]
            .iter()
            .zip(&columns[c])
            .map(|(&b, &c)| b.lb_add(c))
            .collect::<Result<_, _>>()?;
        triple_bound([views[a], views[b], views[c]], floor, u, opts.max_iter)
    };
    // Indexed collection then serial summation: identical float reduction order
    // with/without rayon and for every rayon thread count.
    #[cfg(feature = "rayon")]
    let bounds: Vec<_> = {
        use rayon::prelude::*;
        triples.par_iter().map(solve).collect::<Result<_, _>>()?
    };
    #[cfg(not(feature = "rayon"))]
    let bounds: Vec<_> = triples.iter().map(solve).collect::<Result<_, _>>()?;
    let sum = bounds
        .into_iter()
        .try_fold(T::Acc::ZERO, |sum, b| sum.lb_add(b))?;
    let averaged = sum.lb_ceil_div(k - 2);
    // Floating addition/division can round the average just below the pairwise
    // sum. Both bounds are valid, so retain the stronger one.
    finish(
        if averaged > pairwise {
            averaged
        } else {
            pairwise
        },
        opts,
    )
}

fn view<A: Copy>(data: &[A], n: usize) -> MatrixView<'_, A> {
    MatrixView::from_row_major(data, n, n).expect("square buffer")
}

fn index(k: usize, p: usize, q: usize) -> usize {
    k * p + q - p * (p + 1) / 2 - p - 1
}

fn finish<A: LowerBoundAcc>(value: A, opts: &LowerBoundOptions<A>) -> Result<A, KapError> {
    let value = if opts.sense == Sense::Maximize {
        value.lb_neg()?
    } else {
        value
    };
    Ok(match opts.upper {
        Some(upper) if value > upper => upper,
        _ => value,
    })
}

/// Reusable min-plus product and argmin buffers; no cubic tensor.
struct Axial<A> {
    solver: LapSolver<A>,
    d: Vec<A>,
    chosen: Vec<usize>,
    counts: Vec<usize>,
}

impl<A: LowerBoundAcc> Axial<A> {
    fn new(n: usize) -> Self {
        Self {
            solver: LapSolver::new(),
            d: vec![A::ZERO; n * n],
            chosen: vec![0; n * n],
            counts: vec![0; n],
        }
    }

    fn evaluate(&mut self, mats: [MatrixView<'_, A>; 3], u: &[A]) -> Result<A, KapError> {
        let [a, b, c] = mats;
        let n = u.len();
        for i in 0..n {
            for j in 0..n {
                let mut best = A::INF;
                let mut chosen = 0;
                for (l, &ul) in u.iter().enumerate() {
                    let v = b.get(i, l).lb_add(c.get(j, l))?.lb_sub(ul)?;
                    if v < best {
                        best = v;
                        chosen = l;
                    }
                }
                let value = a.get(i, j).lb_add(best)?;
                if !value.lb_lap_safe(n, 16) {
                    return Err(overflow());
                }
                self.d[i * n + j] = value;
                self.chosen[i * n + j] = chosen;
            }
        }
        let lap = self.solver.solve(view(&self.d, n), Sense::Minimize)?;
        self.counts.fill(0);
        for (&i, &j) in lap.row_ind.iter().zip(&lap.col_ind) {
            self.counts[self.chosen[i * n + j]] += 1;
        }
        u.iter().try_fold(lap.cost, |value, &ul| value.lb_add(ul))
    }
}

fn triple_bound<A: LowerBoundAcc>(
    mats: [MatrixView<'_, A>; 3],
    mut best: A,
    mut u: Vec<A>,
    max_iter: usize,
) -> Result<A, KapError> {
    let n = u.len();
    let mut axial = Axial::new(n);
    let zero = axial.evaluate(mats, &vec![A::ZERO; n])?;
    if zero > best {
        best = zero;
    }
    if axial.counts.iter().all(|&c| c == 1) {
        return Ok(best);
    }
    // A LAP dual seed ensures L(u) >= the three pairwise LAP optima in exact
    // arithmetic: b[i,l]+c[j,l]-u[l] >= row_b[i]+row_c[j].
    let mut value = match axial.evaluate(mats, &u) {
        Ok(value) => value,
        Err(KapError::Unsupported(_)) => return Ok(best),
        Err(error) => return Err(error),
    };
    if value > best {
        best = value;
    }
    if max_iter == 0 || axial.counts.iter().all(|&c| c == 1) {
        return Ok(best);
    }
    // Fixed options, independent of the subgradient budget, preserve prefixes.
    let upper = match k_assignment(&mats, Algo::Dv1, &Options::default()) {
        Ok(solution) => solution.cost,
        Err(KapError::Lap(LapError::Infeasible)) => return Ok(best),
        Err(error) => return Err(error),
    };
    let (mut theta, mut stalled) = (1.0, 0);
    for _ in 0..max_iter {
        let norm: f64 = axial.counts.iter().map(|&c| (1.0 - c as f64).powi(2)).sum();
        if norm == 0.0 || best >= upper {
            break;
        }
        // Subtract in the accumulator before converting, avoiding catastrophic
        // cancellation for integer objectives above 2^53.
        let step = theta * upper.lb_sub(value)?.lb_float() / norm;
        let next: Option<Vec<_>> = u
            .iter()
            .zip(&axial.counts)
            .map(|(&ul, &c)| ul.lb_update(step * (1.0 - c as f64)))
            .collect();
        let Some(next) = next else { break };
        if next.iter().any(|&ul| !ul.lb_lap_safe(n, 64)) {
            break;
        }
        u = next;
        value = match axial.evaluate(mats, &u) {
            Ok(value) => value,
            Err(KapError::Unsupported(_)) => break,
            Err(error) => return Err(error),
        };
        if value > best {
            best = value;
            stalled = 0;
        } else {
            stalled += 1;
            if stalled == 10 {
                theta *= 0.5;
                stalled = 0;
            }
        }
    }
    Ok(best)
}

#[cfg(test)]
#[path = "lower_bound_tests.rs"]
mod tests;
