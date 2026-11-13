//! Deterministic cyclic first-improvement dimensionwise variation.

use super::{Dv, Options, better};
use crate::{Acc, Cost, KapError, LapSolver, Quotient};

/// One representative of each {D, complement}, ordered by (popcount, mask).
pub(crate) fn masks(k: usize, kind: Dv) -> Result<Vec<u64>, KapError> {
    if kind == Dv::Subsets && k > 16 {
        return Err(KapError::Unsupported("sDV requires k <= 16"));
    }
    if kind == Dv::One {
        return Ok((0..if k == 2 { 1 } else { k }).map(|p| 1 << p).collect());
    }
    let mut masks: Vec<_> = (1..(1u64 << k) - 1)
        .filter(|&m| {
            let twice = 2 * m.count_ones() as usize;
            twice < k || (twice == k && m & 1 != 0)
        })
        .collect();
    masks.sort_unstable_by_key(|&m| (m.count_ones(), m));
    Ok(masks)
}

pub(crate) fn improve<A: Acc + Cost<Acc = A>>(
    root: &Quotient<A>,
    mut current: Quotient<A>,
    kind: Dv,
    opts: &Options,
    solver: &mut LapSolver<A>,
) -> Result<Quotient<A>, KapError> {
    let masks = masks(root.n_parts(), kind)?;
    let mut stalled = 0;
    // Nested loops avoid overflow in max_iter * masks.len(). The stalled count
    // continues across sweep boundaries, and each accepted move applies at once.
    for _ in 0..opts.max_iter {
        for &mask in &masks {
            let (two, mut induced) = current.bipartition(root, mask)?;
            let mut moved = false;
            if two.n_nodes(0) != 0 && two.n_nodes(1) != 0 {
                let m = two.matrix(0, 1);
                let cross = induced.iter().fold(A::ZERO, |cost, &(r, c)| {
                    cost + m.get(r as usize, c as usize)
                });
                let (mut matching, cost) = two.solve_pair(0, 1, solver, opts.sense)?;
                induced.sort_unstable();
                matching.sort_unstable();
                if matching != induced && better(cost, cross, opts.sense) {
                    current = two.merge(0, 1, &matching, cost)?;
                    moved = true;
                }
            }
            stalled = if moved { 0 } else { stalled + 1 };
            if stalled == masks.len() {
                return Ok(current);
            }
        }
    }
    Ok(current)
}
