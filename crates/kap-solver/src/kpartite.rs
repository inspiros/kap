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
    /// The supplied initial solution does not have max(partite sizes) nodes.
    IncompleteInitial {
        /// Required number of nodes, counting singleton vertices.
        expected: usize,
        /// Actual number after adding all unlisted vertices as singletons.
        actual: usize,
    },
    /// A partite is too large for u32 vertex identifiers.
    TooManyVertices(usize),
    /// Constituent LAP error.
    Lap(LapError),
    /// An option (`max_iter` or `n_trials`) must be positive.
    BadOption(&'static str),
    /// The request exceeds a documented limit of the algorithm.
    Unsupported(&'static str),
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
            Self::IncompleteInitial { expected, actual } => write!(
                f,
                "initial must be a complete solution: {expected} nodes expected, {actual} given"
            ),
            Self::TooManyVertices(n) => write!(f, "partite size {n} exceeds u32 vertex capacity"),
            Self::Lap(e) => e.fmt(f),
            Self::Unsupported(what) => f.write_str(what),
            Self::BadOption("initial") => write!(f, "initial is unsupported for this algorithm"),
            Self::BadOption(name) => write!(f, "{name} must be a positive integer"),
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

fn write_row<A: Copy>(dst: &mut [std::mem::MaybeUninit<A>], src: &[A]) {
    for (d, &x) in dst.iter_mut().zip(src) {
        d.write(x);
    }
}

/// `dst` = transpose of the row-major `rows x cols` matrix `src`, cache-blocked.
fn transpose_into<A: Copy>(dst: &mut Vec<A>, src: &[A], rows: usize, cols: usize) {
    const B: usize = 32;
    dst.clear();
    if rows * cols == 0 {
        return;
    }
    dst.resize(rows * cols, src[0]);
    for r0 in (0..rows).step_by(B) {
        for c0 in (0..cols).step_by(B) {
            for r in r0..(r0 + B).min(rows) {
                for c in c0..(c0 + B).min(cols) {
                    dst[c * rows + r] = src[r * cols + c];
                }
            }
        }
    }
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

    /// Build a final quotient from a row-major table of original vertices.
    /// `NONE` denotes an absent vertex. Every original vertex must occur once,
    /// and every row must contain at least one vertex.
    pub fn from_nodes(root: &Self, nodes: Vec<u32>) -> Result<Self, KapError> {
        let k = root.k_orig;
        if root.n_parts() != k || (0..k).any(|p| root.members(p) != 1 << p) || nodes.len() % k != 0
        {
            return Err(KapError::BadMatching);
        }
        let mut used: Vec<Vec<bool>> = (0..k).map(|p| vec![false; root.n_nodes(p)]).collect();
        let mut costs = Vec::with_capacity(nodes.len() / k);
        let mut score = A::ZERO;
        for node in nodes.chunks_exact(k) {
            if node.iter().all(|&v| v == NONE) {
                return Err(KapError::BadMatching);
            }
            for (p, &v) in node.iter().enumerate() {
                if v != NONE {
                    let Some(seen) = used[p].get_mut(v as usize) else {
                        return Err(KapError::BadMatching);
                    };
                    if *seen {
                        return Err(KapError::BadMatching);
                    }
                    *seen = true;
                }
            }
            let mut cost = A::ZERO;
            for a in 0..k {
                for b in a + 1..k {
                    if node[a] != NONE && node[b] != NONE {
                        cost = cost + root.matrix(a, b).get(node[a] as usize, node[b] as usize);
                    }
                }
            }
            costs.push(cost);
            score = score + cost;
        }
        if used.iter().flatten().any(|&seen| !seen) {
            return Err(KapError::BadMatching);
        }
        Ok(Self {
            k_orig: k,
            parts: vec![Part {
                members: u64::MAX >> (64 - k),
                nodes: nodes.into(),
                costs: costs.into(),
            }],
            matrices: Vec::new(),
            score,
        })
    }

    /// Row-major original vertex table of a final quotient.
    pub(crate) fn nodes(&self) -> &[u32] {
        assert_eq!(self.n_parts(), 1);
        &self.parts[0].nodes
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
        // `rows` lists the matched pairs, then a's unmatched nodes, then b's.
        let k = self.n_parts();
        let mut scratch = Vec::new();
        let mut matrices = Vec::new();
        for i in 0..pre.len() {
            for j in i + 1..pre.len() {
                let (pi, pj) = (pre[i], pre[j]);
                if pi != a && pj != a {
                    matrices.push(self.matrices[index(k, pi, pj)].clone());
                    continue;
                }
                let other = if pi == a { pj } else { pi };
                let no = self.n_nodes(other);
                let (nr, nc) = if pi == a {
                    (rows.len(), no)
                } else {
                    (no, rows.len())
                };
                let mut data = Arc::<[A]>::new_uninit_slice(nr * nc);
                let out = Arc::get_mut(&mut data).expect("fresh allocation");
                if nr * nc > 0 && pi == a {
                    // Merged nodes are the rows: add a's row r to b's row c.
                    // a < other, so (a, other) is stored with a's nodes as rows.
                    let ma = &self.matrices[index(k, a, other)].data;
                    let mb: &[A] = if b < other {
                        &self.matrices[index(k, b, other)].data
                    } else {
                        // (other, b) is stored with other's nodes as rows;
                        // transpose it once so b's rows are contiguous.
                        let src = &self.matrices[index(k, other, b)].data;
                        transpose_into(&mut scratch, src, no, self.n_nodes(b));
                        &scratch
                    };
                    for (dst, &(r, c)) in out.chunks_exact_mut(no).zip(&rows) {
                        match (r, c) {
                            (Some(r), Some(c)) => {
                                let ra = &ma[r * no..(r + 1) * no];
                                let rb = &mb[c * no..(c + 1) * no];
                                for ((d, &x), &y) in dst.iter_mut().zip(ra).zip(rb) {
                                    d.write(x + y);
                                }
                            }
                            (Some(r), None) => write_row(dst, &ma[r * no..(r + 1) * no]),
                            (None, Some(c)) => write_row(dst, &mb[c * no..(c + 1) * no]),
                            _ => unreachable!(),
                        }
                    }
                } else if nr * nc > 0 {
                    // Merged nodes are the columns; other < a < b, so both
                    // (other, a) and (other, b) are stored with other's rows.
                    let ma = &self.matrices[index(k, other, a)].data;
                    let mb = &self.matrices[index(k, other, b)].data;
                    let (na, nb) = (self.n_nodes(a), self.n_nodes(b));
                    for (o, dst) in out.chunks_exact_mut(nc).enumerate() {
                        let ra = &ma[o * na..(o + 1) * na];
                        let rb = &mb[o * nb..(o + 1) * nb];
                        for (d, &(r, c)) in dst.iter_mut().zip(&rows) {
                            d.write(match (r, c) {
                                (Some(r), Some(c)) => ra[r] + rb[c],
                                (Some(r), None) => ra[r],
                                (None, Some(c)) => rb[c],
                                _ => unreachable!(),
                            });
                        }
                    }
                }
                // SAFETY: the loops above wrote every one of the nr * nc
                // slots: chunks_exact_mut covers `out` exactly, since its
                // length is nr * nc and the chunk length is nc (no or
                // rows.len()), with nr chunks zipped against nr row entries.
                let data = unsafe { data.assume_init() };
                matrices.push(Dense {
                    data,
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

    /// Split a final solution into two portions per node, using original costs.
    /// Side 0 always contains original part 0; empty portions are dropped.
    /// Returns the two-part quotient and the matching induced by this solution.
    pub fn bipartition(&self, root: &Self, mask: u64) -> Result<(Self, Vec<(u32, u32)>), KapError> {
        let k = self.k_orig;
        let full = u64::MAX >> (64 - k);
        if self.n_parts() != 1
            || root.k_orig != k
            || root.n_parts() != k
            || (0..k).any(|p| root.members(p) != 1 << p)
            || mask == 0
            || mask == full
            || mask & !full != 0
        {
            return Err(KapError::BadMatching);
        }
        let masks = if mask & 1 != 0 {
            [mask, full ^ mask]
        } else {
            [full ^ mask, mask]
        };
        let mut counts = vec![0; k];
        for node in self.parts[0].nodes.chunks_exact(k) {
            for (p, &v) in node.iter().enumerate() {
                if v != NONE {
                    if v as usize >= root.n_nodes(p) {
                        return Err(KapError::BadMatching);
                    }
                    counts[p] += 1;
                }
            }
        }
        if (0..k).any(|p| counts[p] != root.n_nodes(p)) {
            return Err(KapError::BadMatching);
        }
        let mut nodes = [Vec::new(), Vec::new()];
        let mut matching = Vec::new();
        for node in self.parts[0].nodes.chunks_exact(k) {
            let mut ids = [None, None];
            for side in 0..2 {
                if (0..k).any(|p| masks[side] & (1 << p) != 0 && node[p] != NONE) {
                    ids[side] = Some((nodes[side].len() / k) as u32);
                    nodes[side].extend(
                        node.iter()
                            .enumerate()
                            .map(|(p, &v)| if masks[side] & (1 << p) != 0 { v } else { NONE }),
                    );
                }
            }
            if let [Some(r), Some(c)] = ids {
                matching.push((r, c));
            }
        }
        let mut parts = Vec::with_capacity(2);
        let mut score = A::ZERO;
        for (side, nodes) in nodes.into_iter().enumerate() {
            let mut costs = vec![A::ZERO; nodes.len() / k];
            for a in 0..k {
                for b in a + 1..k {
                    if masks[side] & (1 << a) == 0 || masks[side] & (1 << b) == 0 {
                        continue;
                    }
                    let m = root.matrix(a, b);
                    for (node, cost) in nodes.chunks_exact(k).zip(&mut costs) {
                        if node[a] != NONE && node[b] != NONE {
                            *cost = *cost + m.get(node[a] as usize, node[b] as usize);
                        }
                    }
                }
            }
            for &cost in &costs {
                score = score + cost;
            }
            parts.push(Part {
                members: masks[side],
                nodes: nodes.into(),
                costs: costs.into(),
            });
        }
        let (nr, nc) = (parts[0].costs.len(), parts[1].costs.len());
        let mut data = vec![A::ZERO; nr * nc];
        let mut scratch = Vec::new();
        if nr != 0 && nc != 0 {
            for a in 0..k {
                if masks[0] & (1 << a) == 0 {
                    continue;
                }
                for b in 0..k {
                    if masks[1] & (1 << b) == 0 {
                        continue;
                    }
                    // Orient each original matrix once; its rows are contiguous.
                    let m: &[A] = if a < b {
                        &root.matrices[index(k, a, b)].data
                    } else {
                        let src = &root.matrices[index(k, b, a)].data;
                        transpose_into(&mut scratch, src, root.n_nodes(b), root.n_nodes(a));
                        &scratch
                    };
                    let nb = root.n_nodes(b);
                    for (dst, row) in data
                        .chunks_exact_mut(nc)
                        .zip(parts[0].nodes.chunks_exact(k))
                    {
                        if row[a] == NONE {
                            continue;
                        }
                        let r = row[a] as usize;
                        let src = &m[r * nb..(r + 1) * nb];
                        for (out, col) in dst.iter_mut().zip(parts[1].nodes.chunks_exact(k)) {
                            if col[b] != NONE {
                                *out = *out + src[col[b] as usize];
                            }
                        }
                    }
                }
            }
        }
        // The largest original part has a vertex in every node: one side has N
        // nodes and the current matching is maximum-cardinality. Rectangular LAP
        // cannot worsen its optimum or change the final node count N.
        Ok((
            Self {
                k_orig: k,
                parts,
                matrices: vec![Dense {
                    data: data.into(),
                    rows: nr,
                    cols: nc,
                }],
                score,
            },
            matching,
        ))
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

    /// Canonical fingerprint of the vertex partition: for every original vertex,
    /// in (partite, vertex) order, the global index of the first member of its
    /// node. Two states have equal labels iff they group the vertices alike.
    pub fn labels(&self) -> Vec<u32> {
        let k = self.k_orig;
        let mut sizes = vec![0usize; k];
        for part in &self.parts {
            for node in part.nodes.chunks_exact(k) {
                for (p, &v) in node.iter().enumerate() {
                    if v != NONE {
                        sizes[p] += 1;
                    }
                }
            }
        }
        let mut offsets = vec![0usize; k];
        for p in 1..k {
            offsets[p] = offsets[p - 1] + sizes[p - 1];
        }
        let mut labels = vec![0u32; offsets[k - 1] + sizes[k - 1]];
        for part in &self.parts {
            for node in part.nodes.chunks_exact(k) {
                let mut first = None;
                for (p, &v) in node.iter().enumerate() {
                    if v != NONE {
                        let g = offsets[p] + v as usize;
                        let label = *first.get_or_insert(g as u32);
                        labels[g] = label;
                    }
                }
            }
        }
        labels
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
