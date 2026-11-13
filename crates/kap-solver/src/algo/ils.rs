//! Karapetyan–Gutin Chain: perturb an incumbent and descend with DV.

use std::time::Instant;

use super::{Options, Rng, better, drop_empty_rows, dv, merge::am};
use crate::{Acc, Cost, KapError, LapSolver, Quotient};

/// Consecutive failed perturbations before the chain accepts a non-improving
/// step. Tuned on random-cost instances (k = 3..6): 50 beat 20, 30, 100 and 200
/// on balance across 1000-iteration and 2-second budgets.
const PATIENCE: usize = 50;

pub(super) fn perturb<A: Acc + Cost<Acc = A>>(
    root: &Quotient<A>,
    current: &Quotient<A>,
    anchor: usize,
    p: usize,
    rng: &mut Rng,
) -> Result<Quotient<A>, KapError> {
    let k = root.n_parts();
    let n = current.n_nodes(0);
    let mut chosen: Vec<_> = (0..n).collect();
    // Partial Fisher–Yates samples distinct nodes uniformly.
    for i in 0..p {
        let j = i + rng.below(n - i);
        chosen.swap(i, j);
    }
    chosen.truncate(p);
    let mut nodes = current.nodes().to_vec();
    let mut entries = Vec::with_capacity(p);
    for part in 0..k {
        if !root.is_gated() && part == anchor {
            continue;
        }
        entries.clear();
        entries.extend(chosen.iter().map(|&i| nodes[i * k + part]));
        rng.shuffle(&mut entries);
        for (&i, &v) in chosen.iter().zip(&entries) {
            nodes[i * k + part] = v;
        }
    }
    if root.is_gated() {
        nodes = drop_empty_rows(nodes, k);
    }
    Quotient::from_nodes(root, nodes)
}

pub(crate) fn ils<A: Acc + Cost<Acc = A>>(
    root: &Quotient<A>,
    opts: &Options,
    solver: &mut LapSolver<A>,
) -> Result<Quotient<A>, KapError> {
    let start = am(root, opts.sense, solver)?;
    ils_from(root, start, opts, solver)
}

pub(crate) fn ils_from<A: Acc + Cost<Acc = A>>(
    root: &Quotient<A>,
    start: Quotient<A>,
    opts: &Options,
    solver: &mut LapSolver<A>,
) -> Result<Quotient<A>, KapError> {
    let start = dv::improve(root, start, opts.dv, dv::CONVERGE, opts.sense, solver)?;
    if !root.is_gated() && start.n_nodes(0) < 2 {
        return Ok(start);
    }
    let mut anchor = 0;
    for part in 1..root.n_parts() {
        if root.n_nodes(part) > root.n_nodes(anchor) {
            anchor = part;
        }
    }
    // Without gating the first largest original part fills every node;
    // keeping it fixed preserves N for unequal sizes. Gated moves have no anchor.
    // A deadline beyond Instant's range cannot expire in a representable run.
    let deadline = opts
        .time_limit
        .and_then(|limit| Instant::now().checked_add(limit));
    let chain = |t: usize, solver: &mut LapSolver<A>| -> Result<Quotient<A>, KapError> {
        let mut rng = Rng::new(opts.seed, t as u64);
        let mut current = start.clone();
        let mut best = start.clone();
        let n = current.n_nodes(0);
        let p = if root.is_gated() {
            0 // Recomputed from the current row count below.
        } else {
            (n.div_ceil(25) + 1).clamp(2, n)
        };
        let mut fails = 0;
        for _ in 0..opts.max_iter {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            let p = if root.is_gated() {
                let n = current.n_nodes(0);
                if n < 2 {
                    fails += 1;
                    continue;
                }
                (n.div_ceil(25) + 1).clamp(2, n)
            } else {
                p
            };
            let candidate = perturb(root, &current, anchor, p, &mut rng)?;
            let candidate =
                dv::improve(root, candidate, opts.dv, dv::CONVERGE, opts.sense, solver)?;
            if candidate.score.is_finite() && better(candidate.score, current.score, opts.sense) {
                current = candidate;
                fails = 0;
                if better(current.score, best.score, opts.sense) {
                    best = current.clone();
                }
            } else {
                fails += 1;
                // Escape stagnation: after PATIENCE consecutive failures step
                // to the latest candidate even though it is not better. The
                // best solution seen is kept separately and returned.
                if fails >= PATIENCE && candidate.score.is_finite() {
                    current = candidate;
                    fails = 0;
                }
            }
        }
        let current = best;
        Ok(current)
    };
    #[cfg(feature = "rayon")]
    let results: Vec<Quotient<A>> = {
        use rayon::prelude::*;
        (0..opts.n_trials)
            .into_par_iter()
            .map_init(LapSolver::new, |s, t| chain(t, s))
            .collect::<Result<_, _>>()?
    };
    #[cfg(not(feature = "rayon"))]
    let results: Vec<Quotient<A>> = (0..opts.n_trials)
        .map(|t| chain(t, solver))
        .collect::<Result<_, _>>()?;
    let mut results = results.into_iter();
    let mut best = results.next().expect("n_trials >= 1");
    for q in results {
        if better(q.score, best.score, opts.sense) {
            best = q;
        }
    }
    Ok(best)
}
