//! Multiple Hungarian heuristics with deterministic pair enumeration.

use crate::{Acc, Cost, KPartite, KSolution, KapError, LapSolver, MatrixView, Quotient, Sense};

pub(crate) mod dv;
mod merge;
mod pairfix;

#[cfg(test)]
pub(crate) use merge::bm_path;
pub(crate) use merge::{am, bm, dm, shift};
pub(crate) use pairfix::{cm, em, fm};

/// Dimensionwise-variation neighbourhood.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dv {
    /// Re-match one original part against the rest.
    One,
    /// Re-match all canonical nontrivial subsets (at most 16 parts).
    Subsets,
}

/// Multiple Hungarian algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algo {
    /// Repeatedly merge the first two parts.
    Am,
    /// Exhaustively choose the best merge tree, memoizing commuting merges.
    Bm,
    /// Steepest descent from Bm over the fix-one-pair neighbourhood.
    Cm,
    /// Choose the best immediate quotient at each level.
    Dm,
    /// Best of `n_trials` independent randomised first-improvement descents from Bm.
    Em,
    /// `max_iter` moves to a random best neighbour (sideways and worse moves
    /// allowed), the current and previous solutions tabu; returns the best seen.
    Fm,
    /// Best Am solution over all cyclic part orders.
    Shift,
    /// Am followed by single-part dimensionwise variation.
    Dv1,
    /// Am followed by subset dimensionwise variation.
    DvS,
}

/// Algorithm settings.
#[derive(Clone, Debug)]
pub struct Options {
    /// Optimization direction.
    pub sense: Sense,
    /// Move limit of Cm, Em (per trial) and Fm; sweep limit of DV.
    pub max_iter: usize,
    /// Number of independent Em trials.
    pub n_trials: usize,
    /// Seed of Em and Fm; equal seeds give equal results.
    pub seed: u64,
    /// Optional DV search after the selected algorithm.
    pub refine: Option<Dv>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            sense: Sense::Minimize,
            max_iter: 100,
            n_trials: 1,
            seed: 0,
            refine: None,
        }
    }
}

pub(crate) fn better<A: Acc>(candidate: A, best: A, sense: Sense) -> bool {
    match sense {
        Sense::Minimize => candidate < best,
        Sense::Maximize => candidate > best,
    }
}

/// Solve a condensed k-partite problem using the selected heuristic.
pub fn k_assignment<T: Cost>(
    mats: &[MatrixView<'_, T>],
    algo: Algo,
    opts: &Options,
) -> Result<KSolution<T::Acc>, KapError>
where
    T::Acc: Cost<Acc = T::Acc>,
{
    let problem = KPartite::new(mats)?;
    if matches!(algo, Algo::Em | Algo::Fm) && opts.max_iter == 0 {
        return Err(KapError::BadOption("max_iter"));
    }
    if algo == Algo::Em && opts.n_trials == 0 {
        return Err(KapError::BadOption("n_trials"));
    }
    if (algo == Algo::DvS || opts.refine == Some(Dv::Subsets)) && problem.sizes().len() > 16 {
        return Err(KapError::Unsupported("sDV requires k <= 16"));
    }
    let root = Quotient::root(&problem);
    let mut solver = LapSolver::new();
    let mut q = match algo {
        Algo::Am => am(&root, opts.sense, &mut solver)?,
        Algo::Bm => bm(&root, opts.sense, &mut solver)?,
        Algo::Dm => dm(&root, opts.sense, &mut solver)?,
        Algo::Cm => cm(&root, opts, &mut solver)?,
        Algo::Em => em(&root, opts, &mut solver)?,
        Algo::Fm => fm(&root, opts, &mut solver)?,
        Algo::Shift => shift(&root, opts.sense, &mut solver)?,
        Algo::Dv1 | Algo::DvS => {
            let start = am(&root, opts.sense, &mut solver)?;
            let kind = if algo == Algo::Dv1 {
                Dv::One
            } else {
                Dv::Subsets
            };
            dv::improve(&root, start, kind, opts, &mut solver)?
        }
    };
    if let Some(kind) = opts.refine {
        q = dv::improve(&root, q, kind, opts, &mut solver)?;
    }
    Ok(q.reconstruct())
}

/// SplitMix64; one independent stream per (seed, stream).
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64, stream: u64) -> Self {
        let mut r = Rng(seed ^ stream.wrapping_mul(0xD1B5_4A32_D192_ED03));
        r.next();
        r
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (n >= 1), by multiply-shift.
    pub(crate) fn below(&mut self, n: usize) -> usize {
        ((self.next() as u128 * n as u128) >> 64) as usize
    }

    pub(crate) fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}
