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
    /// Reserved for steepest descent.
    Cm,
    /// Choose the best immediate quotient at each level.
    Dm,
    /// Reserved for randomized descent.
    Em,
    /// Reserved for tabu descent.
    Fm,
}

/// Algorithm settings.
#[derive(Clone, Debug)]
pub struct Options {
    /// Optimization direction.
    pub sense: Sense,
    /// Descent iteration limit.
    pub max_iter: usize,
    /// Number of independent trials.
    pub n_trials: usize,
    /// Random seed.
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
        _ => return Err(KapError::Unimplemented(algo)),
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
