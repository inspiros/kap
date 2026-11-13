//! Karapetyan–Gutin Chain: perturb an incumbent and descend with DV.

use std::time::Instant;

use super::{Options, Rng, better, dv, merge::am};
use crate::{Acc, Cost, KapError, LapSolver, Quotient};

fn perturb<A: Acc + Cost<Acc = A>>(
    root: &Quotient<A>,
    current: &Quotient<A>,
    anchor: usize,
    rng: &mut Rng,
) -> Result<Quotient<A>, KapError> {
    let k = root.n_parts();
    let n = current.n_nodes(0);
    let p = (n.div_ceil(25) + 1).clamp(2, n);
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
        if part == anchor {
            continue;
        }
        entries.clear();
        entries.extend(chosen.iter().map(|&i| nodes[i * k + part]));
        rng.shuffle(&mut entries);
        for (&i, &v) in chosen.iter().zip(&entries) {
            nodes[i * k + part] = v;
        }
    }
    Quotient::from_nodes(root, nodes)
}

pub(crate) fn ils<A: Acc + Cost<Acc = A>>(
    root: &Quotient<A>,
    opts: &Options,
    solver: &mut LapSolver<A>,
) -> Result<Quotient<A>, KapError> {
    let start = am(root, opts.sense, solver)?;
    let start = dv::improve(root, start, opts.dv, dv::CONVERGE, opts.sense, solver)?;
    if start.n_nodes(0) < 2 {
        return Ok(start);
    }
    let mut anchor = 0;
    for part in 1..root.n_parts() {
        if root.n_nodes(part) > root.n_nodes(anchor) {
            anchor = part;
        }
    }
    // The first largest original part has a vertex in every node. Keeping it
    // fixed ensures no node becomes empty and N is preserved for unequal sizes.
    // A deadline beyond Instant's range cannot expire in a representable run.
    let deadline = opts
        .time_limit
        .and_then(|limit| Instant::now().checked_add(limit));
    let chain = |t: usize, solver: &mut LapSolver<A>| -> Result<Quotient<A>, KapError> {
        let mut rng = Rng::new(opts.seed, t as u64);
        let mut current = start.clone();
        for _ in 0..opts.max_iter {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            let candidate = perturb(root, &current, anchor, &mut rng)?;
            let candidate =
                dv::improve(root, candidate, opts.dv, dv::CONVERGE, opts.sense, solver)?;
            if better(candidate.score, current.score, opts.sense) {
                current = candidate;
            }
        }
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
