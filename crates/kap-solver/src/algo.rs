//! Multiple Hungarian heuristics with deterministic pair enumeration.

use std::collections::HashMap;

use crate::{Acc, Cost, KPartite, KSolution, KapError, LapSolver, MatrixView, Quotient, Sense};

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
}

/// Algorithm settings.
#[derive(Clone, Debug)]
pub struct Options {
    /// Optimization direction.
    pub sense: Sense,
    /// Move limit of Cm, Em (per trial) and Fm.
    pub max_iter: usize,
    /// Number of independent Em trials.
    pub n_trials: usize,
    /// Seed of Em and Fm; equal seeds give equal results.
    pub seed: u64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            sense: Sense::Minimize,
            max_iter: 100,
            n_trials: 1,
            seed: 0,
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
    let mut q = Quotient::root(&problem);
    let mut solver = LapSolver::new();
    match algo {
        Algo::Am => {
            while q.n_parts() > 1 {
                q = q.merge_lap(0, 1, &mut solver, opts.sense)?;
            }
        }
        Algo::Bm => q = bm(&q, opts.sense, &mut solver)?.0,
        Algo::Dm => q = dm(&q, opts.sense, &mut solver)?,
        Algo::Cm => q = cm(&q, opts, &mut solver)?,
        Algo::Em => q = em(&q, opts, &mut solver)?,
        Algo::Fm => q = fm(&q, opts, &mut solver)?,
    }
    Ok(q.reconstruct())
}

pub(crate) fn dm<A: Acc + Cost<Acc = A>>(
    start: &Quotient<A>,
    sense: Sense,
    solver: &mut LapSolver<A>,
) -> Result<Quotient<A>, KapError> {
    let mut q = start.clone();
    while q.n_parts() > 1 {
        let mut best: Option<Quotient<A>> = None;
        for a in 0..q.n_parts() {
            for b in a + 1..q.n_parts() {
                let candidate = q.merge_lap(a, b, solver, sense)?;
                if best
                    .as_ref()
                    .is_none_or(|best| better(candidate.score, best.score, sense))
                {
                    best = Some(candidate);
                }
            }
        }
        q = best.expect("at least two parts");
    }
    Ok(q)
}

// Prefix-coded binary trees: leaves 0..63 label the start state's parts;
// 64 introduces two children, ordered by their smallest leaf. Concatenating
// the trees in current part order uniquely encodes a forest. Disjoint merges
// have the same key, but different bracketings of a part never collide.
type Forest = Vec<Vec<u8>>;

fn merge_forest(forest: &Forest, a: usize, b: usize) -> Forest {
    let mut next = forest.clone();
    let mut tree = Vec::with_capacity(1 + forest[a].len() + forest[b].len());
    tree.push(64);
    tree.extend_from_slice(&forest[a]);
    tree.extend_from_slice(&forest[b]);
    next[a] = tree;
    next.remove(b);
    next
}

struct Best<A> {
    score: A,
    pair: (usize, usize),
    // Keep the chosen LAP result so replay requires no additional LAP calls.
    matching: Vec<(u32, u32)>,
    merge_cost: A,
}

type Memo<A> = HashMap<Vec<u8>, Best<A>>;

fn search<A: Acc + Cost<Acc = A>>(
    q: &Quotient<A>,
    forest: &Forest,
    sense: Sense,
    solver: &mut LapSolver<A>,
    memo: &mut Memo<A>,
) -> Result<A, KapError> {
    if q.n_parts() == 1 {
        return Ok(q.score);
    }
    let key: Vec<_> = forest.concat();
    if let Some(best) = memo.get(&key) {
        return Ok(best.score);
    }
    let mut best: Option<Best<A>> = None;
    for a in 0..q.n_parts() {
        for b in a + 1..q.n_parts() {
            let (matching, merge_cost) = q.solve_pair(a, b, solver, sense)?;
            let child = q.merge(a, b, &matching, merge_cost)?;
            let score = search(&child, &merge_forest(forest, a, b), sense, solver, memo)?;
            if best
                .as_ref()
                .is_none_or(|best| better(score, best.score, sense))
            {
                best = Some(Best {
                    score,
                    pair: (a, b),
                    matching,
                    merge_cost,
                });
            }
        }
    }
    let best = best.expect("at least two parts");
    let score = best.score;
    memo.insert(key, best);
    Ok(score)
}

/// Best final quotient and the first pair on the first optimal merge path.
/// A fresh memo labels the supplied start state's parts, including fixed merges.
pub(crate) fn bm<A: Acc + Cost<Acc = A>>(
    start: &Quotient<A>,
    sense: Sense,
    solver: &mut LapSolver<A>,
) -> Result<(Quotient<A>, (usize, usize)), KapError> {
    let mut forest: Forest = (0..start.n_parts()).map(|p| vec![p as u8]).collect();
    let mut memo = Memo::new();
    search(start, &forest, sense, solver, &mut memo)?;
    let mut q = start.clone();
    let mut first = (0, 0); // No choice when the supplied state is already final.
    while q.n_parts() > 1 {
        let best = &memo[&forest.concat()];
        let (a, b) = best.pair;
        if q.n_parts() == start.n_parts() {
            first = (a, b);
        }
        q = q.merge(a, b, &best.matching, best.merge_cost)?;
        forest = merge_forest(&forest, a, b);
    }
    Ok((q, first))
}

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
    Ok(bm(&fixed, sense, solver)?.0)
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
    let (mut current, mut skip) = bm(root, opts.sense, solver)?;
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
    let (start, first) = bm(root, opts.sense, solver)?;
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
    let (mut current, mut skip) = bm(root, opts.sense, solver)?;
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

/// SplitMix64; one independent stream per (seed, stream).
struct Rng(u64);

impl Rng {
    fn new(seed: u64, stream: u64) -> Self {
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
    fn below(&mut self, n: usize) -> usize {
        ((self.next() as u128 * n as u128) >> 64) as usize
    }

    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}
