//! Pair-fixing local search.

use super::{Options, Rng, better, merge::bm_path};
use crate::{Acc, Cost, KapError, LapSolver, Quotient, Sense};

// ---------------------------------------------------------------------------
// Local search (Cm, Em, Fm), after Gabrovšek et al. (2020), section 3.
//
// A solution is a final quotient of the root problem. Its neighbour for an
// original pair (p, q) keeps the matching it induces between V_p and V_q,
// contracts it and re-solves the remaining (k-1)-partite problem with Bm.
// Re-fixing the pair that produced the current solution gives that solution
// back, so the pair of the last move (for the start, Bm's first merge) is
// skipped.

type Pair = (usize, usize);

fn pairs_except(k: usize, skip: Pair) -> Vec<Pair> {
    (0..k)
        .flat_map(|a| (a + 1..k).map(move |b| (a, b)))
        .filter(|&p| p != skip)
        .collect()
}

fn neighbour<A: Acc + Cost<Acc = A>>(
    root: &Quotient<A>,
    current: &Quotient<A>,
    (a, b): Pair,
    sense: Sense,
    solver: &mut LapSolver<A>,
) -> Result<Quotient<A>, KapError> {
    let fixed = root.fix(a, b, &current.edges_between(a, b))?;
    Ok(bm_path(&fixed, sense, solver)?.0)
}

/// All neighbours for `pairs`, in order; in parallel with the `rayon` feature.
fn neighbours<A: Acc + Cost<Acc = A>>(
    root: &Quotient<A>,
    current: &Quotient<A>,
    pairs: &[Pair],
    sense: Sense,
    solver: &mut LapSolver<A>,
) -> Result<Vec<Quotient<A>>, KapError> {
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        let _ = solver;
        pairs
            .par_iter()
            .map_init(LapSolver::new, |s, &p| {
                neighbour(root, current, p, sense, s)
            })
            .collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        pairs
            .iter()
            .map(|&p| neighbour(root, current, p, sense, solver))
            .collect()
    }
}

/// Index of the first best score.
fn argbest<A: Acc>(qs: &[Quotient<A>], sense: Sense) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, q) in qs.iter().enumerate() {
        if best.is_none_or(|b| better(q.score, qs[b].score, sense)) {
            best = Some(i);
        }
    }
    best
}

pub(crate) fn cm<A: Acc + Cost<Acc = A>>(
    root: &Quotient<A>,
    opts: &Options,
    solver: &mut LapSolver<A>,
) -> Result<Quotient<A>, KapError> {
    let (mut current, mut skip) = bm_path(root, opts.sense, solver)?;
    // Pure steepest descent: a whole neighbourhood of the same solution is
    // evaluated before moving to its best member.
    for _ in 0..opts.max_iter {
        let pairs = pairs_except(root.n_parts(), skip);
        let mut nbs = neighbours(root, &current, &pairs, opts.sense, solver)?;
        match argbest(&nbs, opts.sense) {
            Some(i) if better(nbs[i].score, current.score, opts.sense) => {
                current = nbs.swap_remove(i);
                skip = pairs[i];
            }
            _ => break,
        }
    }
    Ok(current)
}

pub(crate) fn em<A: Acc + Cost<Acc = A>>(
    root: &Quotient<A>,
    opts: &Options,
    solver: &mut LapSolver<A>,
) -> Result<Quotient<A>, KapError> {
    let (start, first) = bm_path(root, opts.sense, solver)?;
    let trial = |t: usize, solver: &mut LapSolver<A>| -> Result<Quotient<A>, KapError> {
        let mut rng = Rng::new(opts.seed, t as u64);
        let (mut current, mut skip) = (start.clone(), first);
        for _ in 0..opts.max_iter {
            let mut pairs = pairs_except(root.n_parts(), skip);
            rng.shuffle(&mut pairs);
            let mut moved = false;
            for p in pairs {
                let nb = neighbour(root, &current, p, opts.sense, solver)?;
                if better(nb.score, current.score, opts.sense) {
                    current = nb;
                    skip = p;
                    moved = true;
                    break;
                }
            }
            if !moved {
                break;
            }
        }
        Ok(current)
    };
    #[cfg(feature = "rayon")]
    let results: Vec<Quotient<A>> = {
        use rayon::prelude::*;
        (0..opts.n_trials)
            .into_par_iter()
            .map_init(LapSolver::new, |s, t| trial(t, s))
            .collect::<Result<_, _>>()?
    };
    #[cfg(not(feature = "rayon"))]
    let results: Vec<Quotient<A>> = (0..opts.n_trials)
        .map(|t| trial(t, solver))
        .collect::<Result<_, _>>()?;
    let mut results = results;
    let i = argbest(&results, opts.sense).expect("n_trials >= 1");
    Ok(results.swap_remove(i))
}

pub(crate) fn fm<A: Acc + Cost<Acc = A>>(
    root: &Quotient<A>,
    opts: &Options,
    solver: &mut LapSolver<A>,
) -> Result<Quotient<A>, KapError> {
    let mut rng = Rng::new(opts.seed, 0);
    let (mut current, mut skip) = bm_path(root, opts.sense, solver)?;
    let mut labels = current.labels();
    let mut previous: Option<Vec<u32>> = None;
    let mut best = current.clone();
    for _ in 0..opts.max_iter {
        let pairs = pairs_except(root.n_parts(), skip);
        let nbs = neighbours(root, &current, &pairs, opts.sense, solver)?;
        // Tabu: staying put and stepping straight back.
        let mut allowed: Vec<(usize, Vec<u32>)> = Vec::new();
        for (i, nb) in nbs.iter().enumerate() {
            let l = nb.labels();
            if l != labels && previous.as_ref() != Some(&l) && !allowed.iter().any(|(_, x)| *x == l)
            {
                allowed.push((i, l));
            }
        }
        let Some(target) = allowed
            .iter()
            .map(|&(i, _)| nbs[i].score)
            .reduce(|x, y| if better(y, x, opts.sense) { y } else { x })
        else {
            break;
        };
        // Sideways and worse moves are allowed: pick uniformly among the best
        // distinct allowed neighbours.
        allowed.retain(|&(i, _)| !better(target, nbs[i].score, opts.sense));
        let (i, l) = allowed.swap_remove(rng.below(allowed.len()));
        previous = Some(std::mem::replace(&mut labels, l));
        current = nbs[i].clone();
        skip = pairs[i];
        if better(current.score, best.score, opts.sense) {
            best = current.clone();
        }
    }
    Ok(best)
}
