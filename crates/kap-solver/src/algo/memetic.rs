//! Gutin–Karapetyan memetic search with DV local optima.

use std::{cmp::Ordering, collections::HashSet, time::Instant};

use super::{Options, Rng, better, dv, merge::shift};
use crate::{Acc, Cost, KapError, LapSolver, Quotient, Sense, kpartite::NONE};

const MU_F: f64 = 0.2;
const MU_M: f64 = 0.1;
const P_M: f64 = 0.5;
const L: usize = 3;
const KEEP: f64 = 0.8;
const A: f64 = 0.08;
const B: f64 = 0.35;
const C: f64 = 0.85;

/// Swap coordinates in non-anchor columns, preserving each column's multiset.
pub(crate) fn perturb(nodes: &[u32], k: usize, anchor: usize, mu: f64, rng: &mut Rng) -> Vec<u32> {
    let mut nodes = nodes.to_vec();
    let n = nodes.len() / k;
    if n < 2 {
        return nodes;
    }
    for _ in 0..(n as f64 * mu / 2.0).ceil() as usize {
        let i = rng.below(n);
        let j = rng.below(n - 1);
        let j = if j >= i { j + 1 } else { j };
        let part = rng.below(k - 1);
        let part = if part >= anchor { part + 1 } else { part };
        nodes.swap(i * k + part, j * k + part);
    }
    nodes
}

fn repair(nodes: &mut [u32], sizes: &[usize], common: usize, rng: &mut Rng) {
    let k = sizes.len();
    let n = nodes.len() / k;
    for (part, &size) in sizes.iter().enumerate() {
        let mut used = vec![false; size];
        let mut slots = Vec::new();
        let mut absent = Vec::new();
        let mut n_none = 0;
        for row in 0..n {
            let v = nodes[row * k + part];
            if v == NONE {
                n_none += 1;
                // Common rows are a prefix and must survive repair unchanged.
                if row >= common {
                    absent.push(row);
                }
            } else if std::mem::replace(&mut used[v as usize], true) {
                slots.push(row);
            }
        }
        let target_none = n - size;
        if n_none > target_none {
            rng.shuffle(&mut absent);
            slots.extend_from_slice(&absent[..n_none - target_none]);
        }
        let mut fill: Vec<_> = used
            .iter()
            .enumerate()
            .filter_map(|(v, &seen)| (!seen).then_some(v as u32))
            .collect();
        fill.extend(std::iter::repeat_n(
            NONE,
            target_none.saturating_sub(n_none),
        ));
        rng.shuffle(&mut fill);
        debug_assert_eq!(slots.len(), fill.len());
        for (row, v) in slots.into_iter().zip(fill) {
            nodes[row * k + part] = v;
        }
    }
}

/// Common rows are copied first; independently shuffled remainders are paired,
/// assigned with probability 0.8, then repaired in every column (anchor too).
pub(crate) fn crossover(
    x: &[u32],
    y: &[u32],
    sizes: &[usize],
    rng: &mut Rng,
) -> (Vec<u32>, Vec<u32>) {
    if x == y {
        return (x.to_vec(), y.to_vec());
    }
    let k = sizes.len();
    let y_rows: HashSet<_> = y.chunks_exact(k).collect();
    let common: HashSet<_> = x
        .chunks_exact(k)
        .filter(|row| y_rows.contains(row))
        .collect();
    let mut child1 = Vec::with_capacity(x.len());
    for row in x.chunks_exact(k).filter(|row| common.contains(row)) {
        child1.extend_from_slice(row);
    }
    let n_common = child1.len() / k;
    let mut child2 = child1.clone();
    let mut rest_x: Vec<_> = x
        .chunks_exact(k)
        .filter(|row| !common.contains(row))
        .collect();
    let mut rest_y: Vec<_> = y
        .chunks_exact(k)
        .filter(|row| !common.contains(row))
        .collect();
    rng.shuffle(&mut rest_x);
    rng.shuffle(&mut rest_y);
    for (x, y) in rest_x.into_iter().zip(rest_y) {
        let (x, y) = if rng.uniform() < KEEP { (x, y) } else { (y, x) };
        child1.extend_from_slice(x);
        child2.extend_from_slice(y);
    }
    repair(&mut child1, sizes, n_common, rng);
    repair(&mut child2, sizes, n_common, rng);
    (child1, child2)
}

enum Candidate<A> {
    Keep(Quotient<A>),
    Search(Vec<u32>),
}

/// All randomness is materialised before this deterministic, ordered LS pass.
fn improve<A: Acc + Cost<Acc = A>>(
    root: &Quotient<A>,
    candidates: &[Candidate<A>],
    opts: &Options,
    solver: &mut LapSolver<A>,
) -> Result<Vec<Quotient<A>>, KapError> {
    let run = |candidate: &Candidate<A>, solver: &mut LapSolver<A>| match candidate {
        Candidate::Keep(q) => Ok(q.clone()),
        Candidate::Search(nodes) => dv::improve(
            root,
            Quotient::from_nodes(root, nodes.clone())?,
            opts.dv,
            dv::CONVERGE,
            opts.sense,
            solver,
        ),
    };
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        let _ = solver;
        candidates
            .par_iter()
            .map_init(LapSolver::new, |s, q| run(q, s))
            .collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        candidates.iter().map(|q| run(q, solver)).collect()
    }
}

fn select<A: Acc + Cost<Acc = A>>(population: &mut Vec<Quotient<A>>, m: usize, sense: Sense) {
    population.sort_by(|x, y| {
        if better(x.score, y.score, sense) {
            Ordering::Less
        } else if better(y.score, x.score, sense) {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });
    let mut seen = HashSet::new();
    population.retain(|q| seen.insert(q.labels()));
    population.truncate(m);
}

fn m_opt(tau: f64, t: f64) -> f64 {
    A * tau.powf(B) / t.powf(C)
}

pub(crate) fn memetic<A: Acc + Cost<Acc = A>>(
    root: &Quotient<A>,
    opts: &Options,
    solver: &mut LapSolver<A>,
) -> Result<Quotient<A>, KapError> {
    let entered = Instant::now();
    // A deadline beyond Instant's range cannot expire in a representable run.
    let deadline = opts.time_limit.and_then(|limit| entered.checked_add(limit));
    let expired = || deadline.is_some_and(|deadline| Instant::now() >= deadline);
    let start = shift(root, opts.sense, solver)?;
    let start = dv::improve(root, start, opts.dv, dv::CONVERGE, opts.sense, solver)?;
    if start.n_nodes(0) < 2 {
        return Ok(start);
    }
    let sizes: Vec<_> = (0..root.n_parts()).map(|p| root.n_nodes(p)).collect();
    let mut anchor = 0;
    for part in 1..sizes.len() {
        if sizes[part] > sizes[anchor] {
            anchor = part;
        }
    }
    // The first largest original part has a vertex in every node. Keeping it
    // fixed during perturbation ensures no node becomes empty and N is preserved.
    let mut rng = Rng::new(opts.seed, 0);
    let mut population = vec![start.clone()];
    let mut produced = 1;
    let adaptive = opts.time_limit.is_some() && opts.population.is_none();
    let mut m = opts.population.unwrap_or(12);
    loop {
        if adaptive {
            let optimum = m_opt(
                opts.time_limit.expect("time mode").as_secs_f64(),
                entered.elapsed().as_secs_f64() / produced as f64,
            );
            m = optimum.round().clamp(2.0, 256.0) as usize;
            if produced as f64 >= optimum || expired() {
                break;
            }
        } else if produced >= m || expired() {
            break;
        }
        #[cfg(feature = "rayon")]
        let batch = if adaptive {
            rayon::current_num_threads()
        } else {
            m - produced
        };
        #[cfg(not(feature = "rayon"))]
        let batch = if adaptive { 1 } else { m - produced };
        let tables: Vec<_> = (0..batch)
            .map(|_| Candidate::Search(perturb(start.nodes(), sizes.len(), anchor, MU_F, &mut rng)))
            .collect();
        population.extend(improve(root, &tables, opts, solver)?);
        produced += batch;
        select(&mut population, m, opts.sense);
    }
    select(&mut population, m, opts.sense);
    for _ in 0..opts.max_iter {
        if expired() {
            break;
        }
        // With a sole distinct member, keep the elite and apply only mutation.
        // Applying the paper's parity drop here would empty the parent pool.
        if population.len() > 1 && (L * m - population.len()) % 2 != 0 {
            population.pop();
        }
        let singleton = population.len() == 1;
        let mut candidates = vec![Candidate::Keep(population[0].clone())];
        let mutate = if singleton {
            &population[..]
        } else {
            &population[1..]
        };
        for member in mutate {
            candidates.push(if rng.uniform() < P_M {
                Candidate::Search(perturb(member.nodes(), sizes.len(), anchor, MU_M, &mut rng))
            } else {
                Candidate::Keep(member.clone())
            });
        }
        if !singleton {
            for _ in 0..(L * m - population.len()) / 2 {
                let x = &population[rng.below(population.len())];
                let y = &population[rng.below(population.len())];
                let (x, y) = crossover(x.nodes(), y.nodes(), &sizes, &mut rng);
                candidates.extend([Candidate::Search(x), Candidate::Search(y)]);
            }
        }
        population = improve(root, &candidates, opts, solver)?;
        select(&mut population, m, opts.sense);
    }
    Ok(population.remove(0))
}
