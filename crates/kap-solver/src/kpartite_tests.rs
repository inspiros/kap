use super::*;
use crate::{
    Algo, Options,
    algo::{better, bm},
    k_assignment,
};
use proptest::prelude::*;

fn views<'a>(sizes: &[usize], data: &'a [Vec<i64>]) -> Vec<MatrixView<'a, i64>> {
    let mut mats = Vec::new();
    for a in 0..sizes.len() {
        for b in a + 1..sizes.len() {
            mats.push(MatrixView::from_row_major(&data[mats.len()], sizes[a], sizes[b]).unwrap());
        }
    }
    mats
}

fn instance() -> impl Strategy<Value = (Vec<usize>, Vec<Vec<i64>>, Sense)> {
    (2usize..=5, any::<bool>(), any::<bool>())
        .prop_flat_map(|(k, equal, maximize)| {
            (0usize..=4, prop::collection::vec(0usize..=4, k)).prop_flat_map(
                move |(n, mut sizes)| {
                    if equal {
                        sizes.fill(n);
                    }
                    let entries = (0..k)
                        .flat_map(|a| (a + 1..k).map(move |b| (a, b)))
                        .map(|(a, b)| sizes[a] * sizes[b])
                        .sum::<usize>();
                    (
                        Just(sizes),
                        prop::collection::vec(-20i64..40, entries),
                        Just(maximize),
                    )
                },
            )
        })
        .prop_map(|(sizes, flat, maximize)| {
            let mut offset = 0;
            let mut data = Vec::new();
            for a in 0..sizes.len() {
                for b in a + 1..sizes.len() {
                    let end = offset + sizes[a] * sizes[b];
                    data.push(flat[offset..end].to_vec());
                    offset = end;
                }
            }
            (
                sizes,
                data,
                if maximize {
                    Sense::Maximize
                } else {
                    Sense::Minimize
                },
            )
        })
}

fn check_valid(sol: &KSolution<i64>, sizes: &[usize], mats: &[MatrixView<'_, i64>]) {
    let mut used: Vec<Vec<bool>> = sizes.iter().map(|&n| vec![false; n]).collect();
    let mut total = 0;
    assert_eq!(sol.cliques.len(), sol.costs.len());
    for (clique, &cost) in sol.cliques.iter().zip(&sol.costs) {
        assert!(clique.len() >= 2);
        assert!(clique.windows(2).all(|w| w[0].0 < w[1].0));
        let mut recomputed = 0;
        for (i, &(p, v)) in clique.iter().enumerate() {
            assert!(!used[p][v]);
            used[p][v] = true;
            for &(q, w) in &clique[i + 1..] {
                recomputed += mats[index(sizes.len(), p, q)].get(v, w);
            }
        }
        assert_eq!(cost, recomputed);
        total += cost;
    }
    for &(p, v) in &sol.free {
        assert!(!used[p][v]);
        used[p][v] = true;
    }
    assert!(used.iter().flatten().all(|&v| v));
    assert_eq!(total, sol.cost);
    if sizes.iter().all(|&n| n == sizes[0]) {
        assert!(sol.free.is_empty());
        assert_eq!(sol.cliques.len(), sizes[0]);
        assert!(sol.cliques.iter().all(|c| c.len() == sizes.len()));
    }
}

fn exhaustive(
    q: &Quotient<i64>,
    sense: Sense,
    solver: &mut LapSolver<i64>,
) -> (Quotient<i64>, (usize, usize)) {
    if q.n_parts() == 1 {
        return (q.clone(), (0, 0));
    }
    let mut best: Option<(Quotient<i64>, (usize, usize))> = None;
    for a in 0..q.n_parts() {
        for b in a + 1..q.n_parts() {
            let next = q.merge_lap(a, b, solver, sense).unwrap();
            let candidate = exhaustive(&next, sense, solver).0;
            if best
                .as_ref()
                .is_none_or(|(best, _)| better(candidate.score, best.score, sense))
            {
                best = Some((candidate, (a, b)));
            }
        }
    }
    best.unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(80))]
    #[test]
    fn solutions_and_memo_match_exhaustive((sizes,data,sense) in instance()) {
        let mats = views(&sizes,&data);
        let root = Quotient::root(&KPartite::new(&mats).unwrap());
        for algo in [Algo::Am,Algo::Bm,Algo::Dm] {
            let sol = k_assignment(&mats,algo,&Options { sense,..Options::default() }).unwrap();
            check_valid(&sol,&sizes,&mats);
        }
        let mut solver = LapSolver::new();
        let (memo,pair) = bm(&root,sense,&mut solver).unwrap();
        let (plain,plain_pair) = exhaustive(&root,sense,&mut solver);
        prop_assert_eq!(memo.reconstruct(),plain.reconstruct());
        prop_assert_eq!(pair,plain_pair);
        // A fresh Bm also works after a fixed partial merge, with its own labels.
        let fixed = root.fix(0,1,&[]).unwrap();
        let (memo,pair) = bm(&fixed,sense,&mut solver).unwrap();
        let (plain,plain_pair) = exhaustive(&fixed,sense,&mut solver);
        prop_assert_eq!(memo.reconstruct(),plain.reconstruct());
        prop_assert_eq!(pair,plain_pair);
    }
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    fn visit(v: &mut Vec<usize>, remaining: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if remaining.is_empty() {
            out.push(v.clone());
            return;
        }
        for i in 0..remaining.len() {
            let x = remaining.remove(i);
            v.push(x);
            visit(v, remaining, out);
            v.pop();
            remaining.insert(i, x);
        }
    }
    let mut out = Vec::new();
    visit(&mut Vec::new(), &mut (0..n).collect(), &mut out);
    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(80))]
    #[test]
    fn three_partite_bound((n,flat) in (0usize..=3).prop_flat_map(|n| (Just(n),prop::collection::vec(-20i64..40,3*n*n)))) {
        let data: Vec<_> = (0..3).map(|i| flat[i*n*n..(i+1)*n*n].to_vec()).collect();
        let mats = views(&[n;3],&data);
        let perms = permutations(n);
        let totals: Vec<i64> = perms.iter().flat_map(|b| perms.iter().map(|c| {
            (0..n).map(|a| mats[0].get(a,b[a])+mats[1].get(a,c[a])+mats[2].get(b[a],c[a])).sum()
        })).collect();
        for sense in [Sense::Minimize,Sense::Maximize] {
            let optimum = if sense == Sense::Minimize { *totals.iter().min().unwrap() } else { *totals.iter().max().unwrap() };
            for algo in [Algo::Am,Algo::Bm,Algo::Dm] {
                let sol = k_assignment(&mats,algo,&Options { sense,..Options::default() }).unwrap();
                prop_assert!(!better(sol.cost,optimum,sense));
            }
        }
    }
}

fn measured_counts(end: usize) {
    let expected = [6, 39, 310, 3000, 34566, 463981];
    for k in 3..=end {
        let sizes = vec![1; k];
        let data = vec![vec![0]; k * (k - 1) / 2];
        let mats = views(&sizes, &data);
        let root = Quotient::root(&KPartite::new(&mats).unwrap());
        LAP_CALLS.with(|n| n.set(0));
        bm(&root, Sense::Minimize, &mut LapSolver::new()).unwrap();
        let count = LAP_CALLS.with(|n| n.get());
        eprintln!("Bm k={k}: {count} LAP calls");
        assert_eq!(count, expected[k - 3]);
    }
}

#[test]
fn bm_lap_counts() {
    measured_counts(6);
}

#[test]
#[ignore = "extended LAP count measurement"]
fn bm_lap_counts_extended() {
    measured_counts(8);
}

#[test]
fn fixed_order_sharing_and_edges() {
    let sizes = [3, 2, 1, 1];
    let data: Vec<_> = (0..4)
        .flat_map(|a| (a + 1..4).map(move |b| vec![1; sizes[a] * sizes[b]]))
        .collect();
    let mats = views(&sizes, &data);
    let root = Quotient::root(&KPartite::new(&mats).unwrap());
    let q = root.fix(1, 0, &[(1, 2), (0, 0)]).unwrap();
    assert!(Arc::ptr_eq(
        &root.matrices[index(4, 2, 3)].data,
        &q.matrices[index(3, 1, 2)].data
    ));
    assert!(Arc::ptr_eq(&root.parts[2].nodes, &q.parts[1].nodes));
    assert_eq!(q.members(0), 3);
    assert_eq!(q.edges_between(0, 1), vec![(0, 0), (2, 1)]);
    let sol = q.reconstruct();
    assert_eq!(
        sol.cliques,
        vec![vec![(0, 2), (1, 1)], vec![(0, 0), (1, 0)]]
    );
    assert_eq!(sol.free, vec![(0, 1), (2, 0), (3, 0)]);
    assert_eq!(q.score, 2);
    assert_eq!(sol.cost, 2);
    assert!(matches!(
        root.fix(0, 1, &[(0, 0), (0, 1)]),
        Err(KapError::BadMatching)
    ));
    assert!(matches!(
        root.fix(0, 1, &[(0, 2)]),
        Err(KapError::BadMatching)
    ));
    assert!(matches!(root.fix(0, 0, &[]), Err(KapError::BadMatching)));
}

#[test]
fn input_errors_and_limits() {
    assert!(matches!(
        KPartite::<i64>::new::<i64>(&[]),
        Err(KapError::BadCount(0))
    ));
    let m = MatrixView::from_row_major(&[0i64; 6], 2, 3).unwrap();
    assert!(matches!(KPartite::new(&[m, m]), Err(KapError::BadCount(2))));
    assert!(matches!(
        KPartite::new(&[m, m, m]),
        Err(KapError::ShapeMismatch {
            partite: 1,
            expected: 3,
            actual: 2,
            ..
        })
    ));
    let empty = MatrixView::from_row_major(&[] as &[i64], 0, 0).unwrap();
    let mats = vec![empty; 64 * 63 / 2];
    let root = Quotient::root(&KPartite::new(&mats).unwrap());
    assert_eq!(root.members(63), 1u64 << 63);
    assert_eq!(root.fix(0, 63, &[]).unwrap().members(0), 1 | (1u64 << 63));
    assert!(matches!(
        KPartite::new(&vec![empty; 65 * 64 / 2]),
        Err(KapError::BadCount(_))
    ));
    let huge = MatrixView::from_row_major(&[] as &[i64], u32::MAX as usize + 1, 0).unwrap();
    assert!(matches!(
        KPartite::new(&[huge]),
        Err(KapError::TooManyVertices(_))
    ));
    for algo in [Algo::Cm, Algo::Em, Algo::Fm] {
        assert!(
            matches!(k_assignment(&[m],algo,&Options::default()),Err(KapError::Unimplemented(a)) if a == algo)
        );
    }
    let nan = MatrixView::from_row_major(&[f64::NAN], 1, 1).unwrap();
    assert!(matches!(
        k_assignment(&[nan], Algo::Am, &Options::default()),
        Err(KapError::Lap(LapError::InvalidCost(0, 0)))
    ));
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Quotient<i64>>();
    send_sync::<Quotient<f64>>();
}
