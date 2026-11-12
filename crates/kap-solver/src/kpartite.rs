//! Dense k-partite problems and shared quotient graphs.

use std::{fmt, sync::Arc};

use crate::{Acc, Cost, LapError, LapSolver, MatrixView, Sense};

/// Missing original vertex in a quotient node.
pub const NONE: u32 = u32::MAX;

/// Invalid input or failure of a constituent LAP.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KapError {
    /// Matrix count does not describe 2 through 64 partites.
    BadCount(usize),
    /// A matrix dimension disagrees with a previously seen partite size.
    ShapeMismatch {
        /// Condensed matrix index.
        matrix: usize,
        /// Original partite index.
        partite: usize,
        /// Previously established size.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
    /// Invalid part positions, vertex indices, or repeated matching vertices.
    BadMatching,
    /// A partite is too large for u32 vertex identifiers.
    TooManyVertices(usize),
    /// Constituent LAP error.
    Lap(LapError),
    /// Algorithm reserved for a later implementation.
    Unimplemented(crate::algo::Algo),
}

impl fmt::Display for KapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadCount(n) => write!(
                f,
                "invalid matrix count {n}: expected k(k-1)/2 for 2 <= k <= 64"
            ),
            Self::ShapeMismatch {
                matrix,
                partite,
                expected,
                actual,
            } => write!(
                f,
                "matrix {matrix}: partite {partite} has size {actual}, expected {expected}"
            ),
            Self::BadMatching => write!(f, "invalid part positions or matching"),
            Self::TooManyVertices(n) => write!(f, "partite size {n} exceeds u32 vertex capacity"),
            Self::Lap(e) => e.fmt(f),
            Self::Unimplemented(a) => write!(f, "algorithm {a:?} is not implemented"),
        }
    }
}

impl std::error::Error for KapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Lap(e) => Some(e),
            _ => None,
        }
    }
}

impl From<LapError> for KapError {
    fn from(e: LapError) -> Self {
        Self::Lap(e)
    }
}

fn index(k: usize, a: usize, b: usize) -> usize {
    debug_assert!(a < b && b < k);
    k * a + b - a * (a + 1) / 2 - a - 1
}

#[derive(Clone, Debug)]
struct Dense<A> {
    data: Arc<[A]>,
    rows: usize,
    cols: usize,
}

impl<A: Copy> Dense<A> {
    fn view(&self) -> MatrixView<'_, A> {
        MatrixView::from_row_major(&self.data, self.rows, self.cols).expect("dense shape")
    }
}

/// Original pair costs in combinations order, widened and copied once.
#[derive(Clone, Debug)]
pub struct KPartite<A> {
    sizes: Arc<[usize]>,
    matrices: Arc<[Dense<A>]>,
}

impl<A: Acc + Cost<Acc = A>> KPartite<A> {
    /// Validate the condensed matrix count and all shared dimensions.
    pub fn new<T: Cost<Acc = A>>(mats: &[MatrixView<'_, T>]) -> Result<Self, KapError> {
        let k = (2..=64)
            .find(|&k| k * (k - 1) / 2 == mats.len())
            .ok_or(KapError::BadCount(mats.len()))?;
        let mut sizes = vec![None; k];
        let mut matrices = Vec::with_capacity(mats.len());
        for a in 0..k {
            for b in a + 1..k {
                let cid = index(k, a, b);
                let m = &mats[cid];
                for (p, actual) in [(a, m.rows()), (b, m.cols())] {
                    if actual > u32::MAX as usize {
                        return Err(KapError::TooManyVertices(actual));
                    }
                    if let Some(expected) = sizes[p] {
                        if actual != expected {
                            return Err(KapError::ShapeMismatch {
                                matrix: cid,
                                partite: p,
                                expected,
                                actual,
                            });
                        }
                    } else {
                        sizes[p] = Some(actual);
                    }
                }
                let data = (0..m.rows())
                    .flat_map(|r| (0..m.cols()).map(move |c| m.get(r, c).widen()))
                    .collect::<Vec<_>>();
                matrices.push(Dense {
                    data: data.into(),
                    rows: m.rows(),
                    cols: m.cols(),
                });
            }
        }
        Ok(Self {
            sizes: sizes.into_iter().map(Option::unwrap).collect(),
            matrices: matrices.into(),
        })
    }

    /// Original partite sizes.
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }
}

#[derive(Clone, Debug)]
struct Part<A> {
    members: u64,
    nodes: Arc<[u32]>,
    costs: Arc<[A]>,
}

/// A merge state; unchanged parts and matrices are shared across clones.
#[derive(Clone, Debug)]
pub struct Quotient<A> {
    k_orig: usize,
    parts: Vec<Part<A>>,
    matrices: Vec<Dense<A>>,
    /// Total cost already fixed inside quotient nodes.
    pub score: A,
}

/// Reconstructed cliques and unmatched vertices in quotient node order.
#[derive(Clone, Debug, PartialEq)]
pub struct KSolution<A> {
    /// Cliques, each in increasing original partite order.
    pub cliques: Vec<Vec<(usize, usize)>>,
    /// Internal edge sum of each clique.
    pub costs: Vec<A>,
    /// Singleton vertices, in quotient node order.
    pub free: Vec<(usize, usize)>,
    /// Sum of clique costs.
    pub cost: A,
}

impl<A: Acc + Cost<Acc = A>> Quotient<A> {
    /// Unmerged state of an original problem.
    pub fn root(problem: &KPartite<A>) -> Self {
        let k = problem.sizes.len();
        let parts = problem
            .sizes
            .iter()
            .enumerate()
            .map(|(p, &n)| {
                let mut nodes = vec![NONE; n * k];
                for i in 0..n {
                    nodes[i * k + p] = i as u32;
                }
                Part {
                    members: 1 << p,
                    nodes: nodes.into(),
                    costs: vec![A::ZERO; n].into(),
                }
            })
            .collect();
        Self {
            k_orig: k,
            parts,
            matrices: problem.matrices.to_vec(),
            score: A::ZERO,
        }
    }

    /// Number of current super-partites.
    pub fn n_parts(&self) -> usize {
        self.parts.len()
    }

    /// Number of nodes in a current super-partite.
    pub fn n_nodes(&self, part: usize) -> usize {
        self.parts[part].costs.len()
    }

    /// Original partites contained in a current super-partite.
    pub fn members(&self, part: usize) -> u64 {
        self.parts[part].members
    }

    /// Current pair matrix, transposed when the requested positions are reversed.
    pub fn matrix(&self, a: usize, b: usize) -> MatrixView<'_, A> {
        if a < b {
            self.matrices[index(self.n_parts(), a, b)].view()
        } else {
            self.matrices[index(self.n_parts(), b, a)].view().t()
        }
    }

    /// Solve the current pair LAP and merge its matched nodes.
    pub fn merge_lap(
        &self,
        a: usize,
        b: usize,
        solver: &mut LapSolver<A>,
        sense: Sense,
    ) -> Result<Self, KapError> {
        let (matching, cost) = self.solve_pair(a, b, solver, sense)?;
        self.merge(a, b, &matching, cost)
    }

    pub(crate) fn solve_pair(
        &self,
        a: usize,
        b: usize,
        solver: &mut LapSolver<A>,
        sense: Sense,
    ) -> Result<(Vec<(u32, u32)>, A), KapError> {
        self.check_pair(a, b)?;
        #[cfg(test)]
        LAP_CALLS.with(|n| n.set(n.get() + 1));
        let sol = solver.solve(self.matrix(a, b), sense)?;
        Ok((
            sol.row_ind
                .into_iter()
                .zip(sol.col_ind)
                .map(|(r, c)| (r as u32, c as u32))
                .collect(),
            sol.cost,
        ))
    }

    fn check_pair(&self, a: usize, b: usize) -> Result<(), KapError> {
        if a == b || a >= self.n_parts() || b >= self.n_parts() {
            Err(KapError::BadMatching)
        } else {
            Ok(())
        }
    }

    /// Merge a fixed, possibly partial matching, computing its current pair cost.
    pub fn fix(&self, a: usize, b: usize, matching: &[(u32, u32)]) -> Result<Self, KapError> {
        self.check_pair(a, b)?;
        let m = self.matrix(a, b);
        let mut cost = A::ZERO;
        for &(r, c) in matching {
            if r as usize >= m.rows() || c as usize >= m.cols() {
                return Err(KapError::BadMatching);
            }
            cost = cost + m.get(r as usize, c as usize);
        }
        self.merge(a, b, matching, cost)
    }

    pub(crate) fn merge(
        &self,
        a: usize,
        b: usize,
        matching: &[(u32, u32)],
        cost: A,
    ) -> Result<Self, KapError> {
        self.check_pair(a, b)?;
        if a > b {
            let reversed: Vec<_> = matching.iter().map(|&(r, c)| (c, r)).collect();
            return self.merge(b, a, &reversed, cost);
        }
        let mut used_a = vec![false; self.n_nodes(a)];
        let mut used_b = vec![false; self.n_nodes(b)];
        let mut rows = Vec::with_capacity(used_a.len() + used_b.len());
        for &(r, c) in matching {
            let (r, c) = (r as usize, c as usize);
            if r >= used_a.len() || c >= used_b.len() || used_a[r] || used_b[c] {
                return Err(KapError::BadMatching);
            }
            used_a[r] = true;
            used_b[c] = true;
            rows.push((Some(r), Some(c)));
        }
        rows.extend(
            used_a
                .iter()
                .enumerate()
                .filter(|&(_, &used)| !used)
                .map(|(r, _)| (Some(r), None)),
        );
        rows.extend(
            used_b
                .iter()
                .enumerate()
                .filter(|&(_, &used)| !used)
                .map(|(c, _)| (None, Some(c))),
        );
        let mut nodes = Vec::with_capacity(rows.len() * self.k_orig);
        let mut costs = Vec::with_capacity(rows.len());
        let ab = self.matrix(a, b);
        for &(r, c) in &rows {
            for p in 0..self.k_orig {
                let left = r.map_or(NONE, |r| self.parts[a].nodes[r * self.k_orig + p]);
                nodes.push(if left != NONE {
                    left
                } else {
                    c.map_or(NONE, |c| self.parts[b].nodes[c * self.k_orig + p])
                });
            }
            costs.push(match (r, c) {
                (Some(r), Some(c)) => {
                    (ab.get(r, c) + self.parts[a].costs[r]) + self.parts[b].costs[c]
                }
                (Some(r), None) => self.parts[a].costs[r],
                (None, Some(c)) => self.parts[b].costs[c],
                _ => unreachable!(),
            });
        }
        let merged = Part {
            members: self.parts[a].members | self.parts[b].members,
            nodes: nodes.into(),
            costs: costs.into(),
        };
        let pre: Vec<_> = (0..self.n_parts()).filter(|&p| p != b).collect();
        let parts = pre
            .iter()
            .map(|&p| {
                if p == a {
                    merged.clone()
                } else {
                    self.parts[p].clone()
                }
            })
            .collect();
        let mut matrices = Vec::new();
        for i in 0..pre.len() {
            for j in i + 1..pre.len() {
                let (pi, pj) = (pre[i], pre[j]);
                if pi != a && pj != a {
                    matrices.push(self.matrices[index(self.n_parts(), pi, pj)].clone());
                    continue;
                }
                let other = if pi == a { pj } else { pi };
                let m1 = self.matrix(a, other);
                let m2 = self.matrix(b, other);
                let (nr, nc) = if pi == a {
                    (rows.len(), self.n_nodes(other))
                } else {
                    (self.n_nodes(other), rows.len())
                };
                let mut data = Vec::with_capacity(nr * nc);
                for r in 0..nr {
                    for c in 0..nc {
                        let (node, o) = if pi == a { (r, c) } else { (c, r) };
                        data.push(match rows[node] {
                            (Some(r), Some(c)) => m1.get(r, o) + m2.get(c, o),
                            (Some(r), None) => m1.get(r, o),
                            (None, Some(c)) => m2.get(c, o),
                            _ => unreachable!(),
                        });
                    }
                }
                matrices.push(Dense {
                    data: data.into(),
                    rows: nr,
                    cols: nc,
                });
            }
        }
        Ok(Self {
            k_orig: self.k_orig,
            parts,
            matrices,
            score: self.score + cost,
        })
    }

    /// Induced edges between two original partites, sorted by the first vertex.
    pub fn edges_between(&self, a: usize, b: usize) -> Vec<(u32, u32)> {
        assert!(a < self.k_orig && b < self.k_orig && a != b);
        let mut edges = Vec::new();
        for part in &self.parts {
            for node in part.nodes.chunks_exact(self.k_orig) {
                if node[a] != NONE && node[b] != NONE {
                    edges.push((node[a], node[b]));
                }
            }
        }
        edges.sort_unstable();
        edges
    }

    /// Reconstruct cliques and singleton vertices (also valid for intermediate states).
    pub fn reconstruct(&self) -> KSolution<A> {
        let mut sol = KSolution {
            cliques: Vec::new(),
            costs: Vec::new(),
            free: Vec::new(),
            cost: A::ZERO,
        };
        for part in &self.parts {
            for (node, &cost) in part.nodes.chunks_exact(self.k_orig).zip(part.costs.iter()) {
                let clique: Vec<_> = node
                    .iter()
                    .enumerate()
                    .filter(|&(_, &v)| v != NONE)
                    .map(|(p, &v)| (p, v as usize))
                    .collect();
                if clique.len() >= 2 {
                    sol.cliques.push(clique);
                    sol.costs.push(cost);
                    sol.cost = sol.cost + cost;
                } else {
                    sol.free.extend(clique);
                }
            }
        }
        sol
    }
}

#[cfg(test)]
thread_local! { pub(crate) static LAP_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

#[cfg(test)]
#[path = "kpartite_tests.rs"]
mod tests;
