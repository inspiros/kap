//! Multiple Hungarian heuristics with deterministic pair enumeration.

use crate::{Acc, Cost, KPartite, KSolution, KapError, LapSolver, MatrixView, Quotient, Sense};

pub(crate) mod dv;
pub(crate) mod ils;
pub(crate) mod memetic;
mod merge;
mod pairfix;

#[cfg(test)]
mod gating_tests;
#[cfg(test)]
mod initial_tests;

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
    /// Iterated local search over DV, with independent seeded chains.
    Ils,
    /// Population of DV local optima with mutation and common-clique crossover.
    Memetic,
}

/// Algorithm settings.
#[derive(Clone, Debug)]
pub struct Options {
    /// Optimization direction.
    pub sense: Sense,
    /// Optional pair-cost threshold. Optimize sums of `d - threshold` over
    /// transversal partial cliques, with singletons allowed. `+inf` forbids a
    /// pair when minimizing, `-inf` when maximizing. All algorithms, DV
    /// refinement and supported initial solutions accept a threshold;
    /// None preserves complete matching.
    pub threshold: Option<f64>,
    /// Move limit of Cm, Em (per trial) and Fm; DV sweeps; ILS perturbations; memetic generations.
    pub max_iter: usize,
    /// Number of independent Em trials or ILS chains.
    pub n_trials: usize,
    /// Seed of Em, Fm, ILS and Memetic; iteration budgets are reproducible.
    pub seed: u64,
    /// Optional DV search after the selected algorithm.
    pub refine: Option<Dv>,
    /// Neighbourhood used within ILS and Memetic.
    pub dv: Dv,
    /// Optional time budget for ILS (after the start) or Memetic (from entry).
    pub time_limit: Option<std::time::Duration>,
    /// Memetic population size; None uses 12 or time-based population sizing.
    pub population: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            sense: Sense::Minimize,
            threshold: None,
            max_iter: 100,
            n_trials: 1,
            seed: 0,
            refine: None,
            dv: Dv::One,
            time_limit: None,
            population: None,
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
/// With `opts.threshold = Some(tau)`, optimize the sum of `d - tau` over
/// pairs within transversal partial cliques; singletons are allowed. `+inf`
/// forbids a pair when minimizing, `-inf` when maximizing. Returned costs use
/// the original matrices. Without a threshold, matching behavior is unchanged.
pub fn k_assignment<T: Cost>(
    mats: &[MatrixView<'_, T>],
    algo: Algo,
    opts: &Options,
) -> Result<KSolution<T::Acc>, KapError>
where
    T::Acc: Cost<Acc = T::Acc>,
{
    run(mats, None, algo, opts)
}

/// Improve supplied cliques instead of constructing a starting solution.
///
/// Each clique contains `(part, vertex)` pairs, with at most one vertex per
/// part. Unlisted vertices become singleton nodes. Supports deterministic
/// starts (Am, Bm, Dm, Shift, Dv1, DvS) and ILS; other algorithms reject
/// `initial`. Deterministic starts are replaced, then `opts.refine` is applied.
/// Without a threshold the initial solution must be complete: after adding
/// singleton nodes its node count must equal `max(part sizes)`, as in a previous
/// solver result. With a threshold, partial solutions (including an empty
/// clique list) are accepted, but a clique containing a forbidden pair is rejected.
/// The result is no worse than the start for integer costs; for floating-point
/// costs this holds up to rounding of the cost sums. With a threshold,
/// "no worse than the start" refers to the shifted objective.
pub fn k_assignment_from<T: Cost>(
    mats: &[MatrixView<'_, T>],
    cliques: &[Vec<(usize, usize)>],
    algo: Algo,
    opts: &Options,
) -> Result<KSolution<T::Acc>, KapError>
where
    T::Acc: Cost<Acc = T::Acc>,
{
    run(mats, Some(cliques), algo, opts)
}

fn from_cliques<A: Acc + Cost<Acc = A>>(
    root: &Quotient<A>,
    cliques: &[Vec<(usize, usize)>],
    sense: Sense,
) -> Result<Quotient<A>, KapError> {
    use crate::kpartite::NONE;
    let k = root.n_parts();
    // A zero-sweep start never reaches a LAP. Apply its entry rules here,
    // including entries not selected by the supplied cliques.
    for a in 0..k {
        for b in a + 1..k {
            let matrix = root.matrix(a, b);
            for r in 0..root.n_nodes(a) {
                for c in 0..root.n_nodes(b) {
                    let value = matrix.get(r, c);
                    let value = match sense {
                        Sense::Minimize => Some(value),
                        Sense::Maximize => value.checked_neg(),
                    };
                    if !value.is_some_and(Acc::is_valid) {
                        return Err(crate::LapError::InvalidCost(r, c).into());
                    }
                }
            }
        }
    }
    let mut used: Vec<_> = (0..k).map(|p| vec![false; root.n_nodes(p)]).collect();
    let mut nodes = Vec::new();
    for clique in cliques {
        if clique.is_empty() {
            return Err(KapError::BadMatching);
        }
        let mut row = vec![NONE; k];
        for &(p, v) in clique {
            let seen = used
                .get_mut(p)
                .and_then(|part| part.get_mut(v))
                .ok_or(KapError::BadMatching)?;
            if *seen || row[p] != NONE {
                return Err(KapError::BadMatching);
            }
            *seen = true;
            row[p] = v as u32;
        }
        nodes.extend(row);
    }
    for (p, part) in used.iter().enumerate() {
        for (v, &seen) in part.iter().enumerate() {
            if !seen {
                let mut row = vec![NONE; k];
                row[p] = v as u32;
                nodes.extend(row);
            }
        }
    }
    let expected = (0..k).map(|p| root.n_nodes(p)).max().unwrap();
    let actual = nodes.len() / k;
    if !root.is_gated() && actual != expected {
        return Err(KapError::IncompleteInitial { expected, actual });
    }
    let start = Quotient::from_nodes_checked(root, nodes)?;
    if root.is_gated() && !start.score.is_finite() {
        return Err(KapError::ForbiddenInitial);
    }
    Ok(start)
}

fn run<T: Cost>(
    mats: &[MatrixView<'_, T>],
    initial: Option<&[Vec<(usize, usize)>]>,
    algo: Algo,
    opts: &Options,
) -> Result<KSolution<T::Acc>, KapError>
where
    T::Acc: Cost<Acc = T::Acc>,
{
    let original = KPartite::new(mats)?;
    let problem = match opts.threshold {
        Some(tau) => original.gated(tau, opts.sense)?,
        None => original.clone(),
    };
    if initial.is_some() && matches!(algo, Algo::Cm | Algo::Em | Algo::Fm | Algo::Memetic) {
        return Err(KapError::BadOption("initial"));
    }
    if matches!(algo, Algo::Em | Algo::Fm) && opts.max_iter == 0 {
        return Err(KapError::BadOption("max_iter"));
    }
    if matches!(algo, Algo::Em | Algo::Ils) && opts.n_trials == 0 {
        return Err(KapError::BadOption("n_trials"));
    }
    if algo == Algo::Memetic && opts.population.is_some_and(|m| m < 2) {
        return Err(KapError::BadOption("population"));
    }
    if (algo == Algo::DvS
        || opts.refine == Some(Dv::Subsets)
        || (matches!(algo, Algo::Ils | Algo::Memetic) && opts.dv == Dv::Subsets))
        && problem.sizes().len() > 16
    {
        return Err(KapError::Unsupported("sDV requires k <= 16"));
    }
    let root = Quotient::root(&problem);
    let mut solver = LapSolver::new();
    let start = initial
        .map(|cliques| from_cliques(&root, cliques, opts.sense))
        .transpose()?;
    let mut q = match (algo, start) {
        (Algo::Ils, Some(start)) => ils::ils_from(&root, start, opts, &mut solver)?,
        (Algo::Dv1 | Algo::DvS, Some(start)) => {
            let kind = if algo == Algo::Dv1 {
                Dv::One
            } else {
                Dv::Subsets
            };
            dv::improve(&root, start, kind, opts.max_iter, opts.sense, &mut solver)?
        }
        (_, Some(start)) => start,
        (_, None) => match algo {
            Algo::Am => am(&root, opts.sense, &mut solver)?,
            Algo::Bm => bm(&root, opts.sense, &mut solver)?,
            Algo::Dm => dm(&root, opts.sense, &mut solver)?,
            Algo::Cm => cm(&root, opts, &mut solver)?,
            Algo::Em => em(&root, opts, &mut solver)?,
            Algo::Fm => fm(&root, opts, &mut solver)?,
            Algo::Ils => ils::ils(&root, opts, &mut solver)?,
            Algo::Memetic => memetic::memetic(&root, opts, &mut solver)?,
            Algo::Shift => shift(&root, opts.sense, &mut solver)?,
            Algo::Dv1 | Algo::DvS => {
                let start = am(&root, opts.sense, &mut solver)?;
                let kind = if algo == Algo::Dv1 {
                    Dv::One
                } else {
                    Dv::Subsets
                };
                dv::improve(&root, start, kind, opts.max_iter, opts.sense, &mut solver)?
            }
        },
    };
    if let Some(kind) = opts.refine {
        q = dv::improve(&root, q, kind, opts.max_iter, opts.sense, &mut solver)?;
    }
    let mut solution = q.reconstruct();
    if opts.threshold.is_some() {
        original.recost(&mut solution);
    }
    Ok(solution)
}

/// Column permutations can leave empty rows when no part is an anchor.
fn drop_empty_rows(nodes: Vec<u32>, k: usize) -> Vec<u32> {
    nodes
        .chunks_exact(k)
        .filter(|row| row.iter().any(|&v| v != crate::kpartite::NONE))
        .flatten()
        .copied()
        .collect()
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

    /// Uniform in [0, 1), using the high 53 bits.
    pub(crate) fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    pub(crate) fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}
