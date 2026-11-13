use super::*;
use crate::{
    Algo, Options,
    algo::{better, bm_path as bm},
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
            for algo in [Algo::Am,Algo::Bm,Algo::Cm,Algo::Dm,Algo::Em,Algo::Fm,Algo::Shift,Algo::Dv1,Algo::DvS] {
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
    for (algo, max_iter, n_trials) in [(Algo::Em, 0, 1), (Algo::Em, 1, 0), (Algo::Fm, 0, 1)] {
        let opts = Options {
            max_iter,
            n_trials,
            ..Options::default()
        };
        assert!(matches!(
            k_assignment(&[m], algo, &opts),
            Err(KapError::BadOption(_))
        ));
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

fn all_pairs(k: usize) -> Vec<(usize, usize)> {
    (0..k)
        .flat_map(|a| (a + 1..k).map(move |b| (a, b)))
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(60))]
    #[test]
    fn local_search_properties((sizes,data,sense) in instance(), seed in any::<u64>()) {
        let mats = views(&sizes,&data);
        let root = Quotient::root(&KPartite::new(&mats).unwrap());
        let mut solver = LapSolver::new();
        let start = bm(&root,sense,&mut solver).unwrap().0.reconstruct().cost;
        let opts = |max_iter, n_trials| Options { sense, max_iter, n_trials, seed, ..Options::default() };
        for algo in [Algo::Cm,Algo::Em,Algo::Fm] {
            let sol = k_assignment(&mats,algo,&opts(100,3)).unwrap();
            check_valid(&sol,&sizes,&mats);
            prop_assert!(!better(start,sol.cost,sense), "{:?} worse than Bm", algo);
            // Same seed, same result.
            prop_assert_eq!(&sol,&k_assignment(&mats,algo,&opts(100,3)).unwrap());
        }
        // Cm (and each Em trial) ends in a local optimum: no neighbour, over
        // every pair, is strictly better.
        let cm = crate::algo::cm(&root,&opts(1000,1),&mut solver).unwrap();
        for (a,b) in all_pairs(sizes.len()) {
            let fixed = root.fix(a,b,&cm.edges_between(a,b)).unwrap();
            let nb = bm(&fixed,sense,&mut solver).unwrap().0;
            prop_assert!(!better(nb.score,cm.score,sense));
        }
        // Trial 0 of Em uses the same stream whatever n_trials is.
        let one = k_assignment(&mats,Algo::Em,&opts(100,1)).unwrap().cost;
        let five = k_assignment(&mats,Algo::Em,&opts(100,5)).unwrap().cost;
        prop_assert!(!better(one,five,sense));
    }
}

#[test]
fn labels_identify_partitions() {
    let data: Vec<Vec<i64>> = (0..3)
        .map(|i| (0..9).map(|j| (i * 7 + j * 5) % 11).collect())
        .collect();
    let mats = views(&[3, 3, 3], &data);
    let root = Quotient::root(&KPartite::new(&mats).unwrap());
    let a = root.fix(0, 1, &[(0, 1), (1, 0)]).unwrap();
    let b = root.fix(0, 1, &[(1, 0), (0, 1)]).unwrap();
    let c = root.fix(0, 1, &[(0, 0), (1, 1)]).unwrap();
    assert_eq!(a.labels(), b.labels());
    assert_ne!(a.labels(), c.labels());
    assert_eq!(root.labels(), (0..9).collect::<Vec<u32>>());
}

const BASE_ALGOS: [Algo; 6] = [Algo::Am, Algo::Bm, Algo::Cm, Algo::Dm, Algo::Em, Algo::Fm];

fn canonical_cliques(sol: KSolution<i64>) -> Vec<(Vec<(usize, usize)>, i64)> {
    let mut cliques: Vec<_> = sol.cliques.into_iter().zip(sol.costs).collect();
    cliques.extend(sol.free.into_iter().map(|v| (vec![v], 0)));
    cliques.sort_unstable();
    cliques
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(60))]
    #[test]
    fn from_nodes_round_trip((sizes, data, sense) in instance()) {
        let mats = views(&sizes, &data);
        let root = Quotient::root(&KPartite::new(&mats).unwrap());
        let mut solver = LapSolver::new();
        for kind in [crate::Dv::One, crate::Dv::Subsets] {
            let start = crate::algo::am(&root, sense, &mut solver).unwrap();
            let current = crate::algo::dv::improve(&root, start, kind, usize::MAX, sense, &mut solver).unwrap();
            let rebuilt = Quotient::from_nodes(&root, current.nodes().to_vec()).unwrap();
            prop_assert_eq!(rebuilt.score, current.score);
            prop_assert_eq!(rebuilt.members(0), (1 << sizes.len()) - 1);
            prop_assert!(rebuilt.matrices.is_empty());
            check_valid(&rebuilt.reconstruct(), &sizes, &mats);
            prop_assert_eq!(canonical_cliques(rebuilt.reconstruct()), canonical_cliques(current.reconstruct()));
        }
    }

    #[test]
    fn ils_properties((sizes, data, sense) in instance(), seed in any::<u64>()) {
        use crate::{Dv, algo::{dv, ils}};
        let mats = views(&sizes, &data);
        let root = Quotient::root(&KPartite::new(&mats).unwrap());
        let mut solver = LapSolver::new();
        for (kind, base_algo) in [(Dv::One, Algo::Dv1), (Dv::Subsets, Algo::DvS)] {
            let opts = Options { sense, seed, dv: kind, max_iter: 40, ..Options::default() };
            let base = k_assignment(&mats, base_algo, &Options { max_iter: usize::MAX, ..opts.clone() }).unwrap();
            let zero = k_assignment(&mats, Algo::Ils, &Options { max_iter: 0, ..opts.clone() }).unwrap();
            prop_assert_eq!(&zero, &base);
            let current = ils::ils(&root, &opts, &mut solver).unwrap();
            let sol = k_assignment(&mats, Algo::Ils, &opts).unwrap();
            prop_assert_eq!(current.reconstruct(), sol.clone());
            check_valid(&sol, &sizes, &mats);
            prop_assert_eq!(current.n_nodes(0), *sizes.iter().max().unwrap());
            prop_assert!(!better(base.cost, sol.cost, sense));
            prop_assert_eq!(&sol, &k_assignment(&mats, Algo::Ils, &opts).unwrap());
            let four = k_assignment(&mats, Algo::Ils, &Options { n_trials: 4, ..opts.clone() }).unwrap();
            prop_assert!(!better(sol.cost, four.cost, sense));
            for mask in dv::masks(sizes.len(), kind).unwrap() {
                let (two, matching) = current.bipartition(&root, mask).unwrap();
                let cross: i64 = matching.iter().map(|&(r,c)| two.matrix(0,1).get(r as usize,c as usize)).sum();
                let (_, best) = two.solve_pair(0, 1, &mut solver, sense).unwrap();
                prop_assert!(!better(best, cross, sense));
            }
            let timed = k_assignment(&mats, Algo::Ils, &Options {
                time_limit: Some(std::time::Duration::ZERO), max_iter: usize::MAX, ..opts
            }).unwrap();
            prop_assert_eq!(timed, zero);
        }
    }
}

#[test]
fn from_nodes_bad_tables_and_limits() {
    let data = vec![vec![0; 4], vec![0; 6], vec![0; 6]];
    let mats = views(&[2, 2, 3], &data);
    let root = Quotient::root(&KPartite::new(&mats).unwrap());
    let good = vec![0, 0, 0, 1, 1, 1, NONE, NONE, 2];
    let mut duplicate = good.clone();
    duplicate[6] = 0;
    let mut missing = good.clone();
    missing[3] = NONE;
    let mut empty_row = good.clone();
    empty_row.extend([NONE; 3]);
    let mut wrong_length = good.clone();
    wrong_length.pop();
    let mut out_of_range = good.clone();
    out_of_range[3] = 2;
    for nodes in [duplicate, missing, empty_row, wrong_length, out_of_range] {
        assert!(matches!(
            Quotient::from_nodes(&root, nodes),
            Err(KapError::BadMatching)
        ));
    }
    let final_q = Quotient::from_nodes(&root, good.clone()).unwrap();
    assert!(matches!(
        Quotient::from_nodes(&final_q, good),
        Err(KapError::BadMatching)
    ));
    // A valid table can contain singleton nodes and need not have max(size) rows.
    let singletons = vec![
        0, NONE, NONE, 1, NONE, NONE, NONE, 0, NONE, NONE, 1, NONE, NONE, NONE, 0, NONE, NONE, 1,
        NONE, NONE, 2,
    ];
    let q = Quotient::from_nodes(&root, singletons).unwrap();
    check_valid(&q.reconstruct(), &[2, 2, 3], &mats);
    assert_eq!(q.n_nodes(0), 7);
    let empty = MatrixView::from_row_major(&[] as &[i64], 0, 0).unwrap();
    let root64 = Quotient::root(&KPartite::new(&vec![empty; 64 * 63 / 2]).unwrap());
    let q = Quotient::from_nodes(&root64, vec![]).unwrap();
    assert_eq!(q.members(0), u64::MAX);
    assert_eq!(q.score, 0);
}

#[test]
fn ils_unequal_sizes_deadline_and_options() {
    use crate::{Dv, algo::ils};
    for sizes in [vec![2, 2, 3], vec![1, 4, 2, 3]] {
        let data: Vec<Vec<i64>> = all_pairs(sizes.len())
            .iter()
            .enumerate()
            .map(|(i, &(a, b))| {
                (0..sizes[a] * sizes[b])
                    .map(|j| ((i * 13 + j * 7) % 31) as i64 - 10)
                    .collect()
            })
            .collect();
        let mats = views(&sizes, &data);
        let root = Quotient::root(&KPartite::new(&mats).unwrap());
        for sense in [Sense::Minimize, Sense::Maximize] {
            for dv in [Dv::One, Dv::Subsets] {
                let opts = Options {
                    sense,
                    dv,
                    max_iter: 200,
                    n_trials: 4,
                    ..Options::default()
                };
                let q = ils::ils(&root, &opts, &mut LapSolver::new()).unwrap();
                check_valid(&q.reconstruct(), &sizes, &mats);
                assert_eq!(q.n_nodes(0), *sizes.iter().max().unwrap());
                let start = k_assignment(
                    &mats,
                    Algo::Ils,
                    &Options {
                        max_iter: 0,
                        ..opts.clone()
                    },
                )
                .unwrap();
                let distant = k_assignment(
                    &mats,
                    Algo::Ils,
                    &Options {
                        max_iter: 0,
                        time_limit: Some(std::time::Duration::MAX),
                        ..opts.clone()
                    },
                )
                .unwrap();
                assert_eq!(distant, start);
                let before = std::time::Instant::now();
                let timed = k_assignment(
                    &mats,
                    Algo::Ils,
                    &Options {
                        max_iter: usize::MAX,
                        time_limit: Some(std::time::Duration::ZERO),
                        ..opts.clone()
                    },
                )
                .unwrap();
                assert!(before.elapsed() < std::time::Duration::from_secs(5));
                assert_eq!(timed, start);
                let refined = k_assignment(
                    &mats,
                    Algo::Ils,
                    &Options {
                        refine: Some(dv),
                        ..opts
                    },
                )
                .unwrap();
                assert_eq!(refined, q.reconstruct());
            }
        }
        assert_eq!(
            k_assignment(
                &mats,
                Algo::Ils,
                &Options {
                    n_trials: 0,
                    ..Options::default()
                }
            ),
            Err(KapError::BadOption("n_trials"))
        );
    }
    let empty = MatrixView::from_row_major(&[] as &[i64], 0, 0).unwrap();
    assert_eq!(
        k_assignment(
            &vec![empty; 17 * 16 / 2],
            Algo::Ils,
            &Options {
                dv: Dv::Subsets,
                max_iter: 0,
                ..Options::default()
            }
        ),
        Err(KapError::Unsupported("sDV requires k <= 16"))
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(60))]
    #[test]
    fn dv_shift_and_refinement((sizes, data, sense) in instance(), seed in any::<u64>()) {
        use crate::{Dv, algo::{am, dv}};
        let mats = views(&sizes, &data);
        let opts = Options { sense, seed, max_iter: 1000, n_trials: 2, ..Options::default() };
        let root = Quotient::root(&KPartite::new(&mats).unwrap());
        let mut solver = LapSolver::new();
        let start = am(&root, sense, &mut solver).unwrap();
        for algo in [Algo::Shift, Algo::Dv1, Algo::DvS] {
            let sol = k_assignment(&mats, algo, &opts).unwrap();
            check_valid(&sol, &sizes, &mats);
            prop_assert!(!better(start.score, sol.cost, sense));
        }
        for algo in BASE_ALGOS {
            let base = k_assignment(&mats, algo, &opts).unwrap();
            for kind in [Dv::One, Dv::Subsets] {
                let refined = k_assignment(&mats, algo, &Options { refine: Some(kind), ..opts.clone() }).unwrap();
                check_valid(&refined, &sizes, &mats);
                prop_assert!(!better(base.cost, refined.cost, sense));
            }
        }
        for kind in [Dv::One, Dv::Subsets] {
            let current = dv::improve(&root, start.clone(), kind, opts.max_iter, sense, &mut solver).unwrap();
            for mask in dv::masks(sizes.len(), kind).unwrap() {
                let (two, matching) = current.bipartition(&root, mask).unwrap();
                let fixed = two.fix(0, 1, &matching).unwrap();
                prop_assert_eq!(fixed.score, current.score);
                prop_assert_eq!(canonical_cliques(fixed.reconstruct()), canonical_cliques(current.reconstruct()));
                prop_assert_eq!(two.n_nodes(0).max(two.n_nodes(1)), current.n_nodes(0));
                prop_assert_eq!(matching.len(), two.n_nodes(0).min(two.n_nodes(1)));
                let cross: i64 = matching.iter().map(|&(r,c)| two.matrix(0,1).get(r as usize,c as usize)).sum();
                let (_, best) = two.solve_pair(0, 1, &mut solver, sense).unwrap();
                prop_assert!(!better(best, cross, sense));
            }
        }
        if sizes.len() == 2 {
            let optimum = root.solve_pair(0, 1, &mut solver, sense).unwrap().1;
            prop_assert_eq!(k_assignment(&mats, Algo::Dv1, &opts).unwrap().cost, optimum);
        }
        if sizes.len() == 3 {
            prop_assert_eq!(k_assignment(&mats, Algo::Dv1, &opts).unwrap(), k_assignment(&mats, Algo::DvS, &opts).unwrap());
        }
        prop_assert_eq!(k_assignment(&mats, Algo::Dv1, &Options { max_iter: 0, ..opts }).unwrap(), start.reconstruct());
    }
}

fn wide_instance() -> impl Strategy<Value = (Vec<usize>, Vec<Vec<i64>>, Sense)> {
    (2usize..=5)
        .prop_flat_map(|k| {
            prop::collection::vec(0usize..=5, k).prop_flat_map(move |sizes| {
                let entries = (0..k)
                    .flat_map(|a| (a + 1..k).map(move |b| (a, b)))
                    .map(|(a, b)| sizes[a] * sizes[b])
                    .sum::<usize>();
                (
                    Just(sizes),
                    prop::collection::vec(0i64..1_000_000_000, entries),
                    any::<bool>(),
                )
            })
        })
        .prop_map(|(sizes, flat, maximize)| {
            let mut data = Vec::new();
            let mut offset = 0;
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(80))]
    #[test]
    fn shift_matches_independent_rotations((sizes, data, sense) in wide_instance()) {
        let k = sizes.len();
        let mats = views(&sizes, &data);
        let opts = Options { sense, ..Options::default() };
        let mut best = k_assignment(&mats, Algo::Am, &opts).unwrap().cost;
        for r in 0..k {
            let perm: Vec<_> = (0..k).map(|p| (p+r)%k).collect();
            let rotated_sizes: Vec<_> = perm.iter().map(|&p| sizes[p]).collect();
            let mut rotated_data = Vec::new();
            for a in 0..k {
                for b in a+1..k {
                    let (p,q) = (perm[a],perm[b]);
                    let m = if p < q { mats[index(k,p,q)] } else { mats[index(k,q,p)].t() };
                    rotated_data.push((0..m.rows()).flat_map(|i| (0..m.cols()).map(move |j| m.get(i,j))).collect());
                }
            }
            let rotated_mats = views(&rotated_sizes, &rotated_data);
            let sol = k_assignment(&rotated_mats, Algo::Am, &opts).unwrap();
            if better(sol.cost, best, sense) { best = sol.cost; }
        }
        prop_assert_eq!(k_assignment(&mats, Algo::Shift, &opts).unwrap().cost, best);
    }
}

#[test]
fn dv_masks_errors_and_shift_ties() {
    use crate::{
        Dv,
        algo::{am, dv, shift},
    };
    assert_eq!(dv::masks(2, Dv::One).unwrap(), vec![1]);
    assert_eq!(dv::masks(3, Dv::Subsets).unwrap(), vec![1, 2, 4]);
    assert_eq!(
        dv::masks(4, Dv::Subsets).unwrap(),
        vec![1, 2, 4, 8, 3, 5, 9]
    );
    for k in 2..=16 {
        let masks = dv::masks(k, Dv::Subsets).unwrap();
        assert_eq!(masks.len(), (1 << (k - 1)) - 1);
        assert!(
            masks
                .windows(2)
                .all(|w| (w[0].count_ones(), w[0]) < (w[1].count_ones(), w[1]))
        );
    }
    assert_eq!(dv::masks(64, Dv::One).unwrap().len(), 64);
    let empty = MatrixView::from_row_major(&[] as &[i64], 0, 0).unwrap();
    for (algo, refine) in [(Algo::DvS, None), (Algo::Am, Some(Dv::Subsets))] {
        let err = k_assignment(
            &vec![empty; 17 * 16 / 2],
            algo,
            &Options {
                refine,
                ..Options::default()
            },
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "sDV requires k <= 16");
    }
    let data = vec![vec![0i64; 4]; 6];
    let mats = views(&[2; 4], &data);
    let root = Quotient::root(&KPartite::new(&mats).unwrap());
    let mut solver = LapSolver::new();
    let current = am(&root, Sense::Minimize, &mut solver).unwrap();
    assert_eq!(
        shift(&root, Sense::Minimize, &mut solver)
            .unwrap()
            .reconstruct(),
        current.reconstruct()
    );
    for mask in [0, 15, 16, u64::MAX] {
        assert!(current.bipartition(&root, mask).is_err());
    }
    assert!(root.bipartition(&root, 1).is_err());
    assert!(current.bipartition(&current, 1).is_err());
    let other = Quotient::root(&KPartite::new(&[empty]).unwrap());
    assert!(current.bipartition(&other, 1).is_err());
    let wrong_sizes = Quotient::root(&KPartite::new(&vec![empty; 6]).unwrap());
    assert!(current.bipartition(&wrong_sizes, 1).is_err());
    // k=64 must not overflow full-mask construction.
    let root64 = Quotient::root(&KPartite::new(&vec![empty; 64 * 63 / 2]).unwrap());
    let final64 = am(&root64, Sense::Minimize, &mut solver).unwrap();
    assert!(final64.bipartition(&root64, 1u64 << 63).is_ok());
}

fn check_memetic_operators(sizes: &[usize], x: &[u32], y: &[u32], root: &Quotient<i64>, seed: u64) {
    use crate::algo::{
        Rng,
        memetic::{crossover, perturb},
    };
    let k = sizes.len();
    let anchor = (0..k)
        .find(|&p| sizes[p] == *sizes.iter().max().unwrap())
        .unwrap();
    let mut rng = Rng::new(seed, 0);
    let perturbed = perturb(x, k, anchor, 0.2, &mut rng);
    Quotient::from_nodes(root, perturbed.clone()).unwrap();
    assert_eq!(
        x.chunks_exact(k).map(|r| r[anchor]).collect::<Vec<_>>(),
        perturbed
            .chunks_exact(k)
            .map(|r| r[anchor])
            .collect::<Vec<_>>()
    );
    let (a, b) = crossover(x, y, sizes, &mut rng);
    for child in [&a, &b] {
        Quotient::from_nodes(root, child.to_vec()).unwrap();
        for row in x
            .chunks_exact(k)
            .filter(|row| y.chunks_exact(k).any(|r| r == *row))
        {
            assert!(child.chunks_exact(k).any(|r| r == row));
        }
    }
    assert_eq!(crossover(x, x, sizes, &mut rng), (x.to_vec(), x.to_vec()));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(60))]
    #[test]
    fn memetic_operator_properties((sizes, data, sense) in instance(), seed in any::<u64>()) {
        use crate::algo::{Rng, memetic::perturb, shift};
        let mats = views(&sizes, &data);
        let root = Quotient::root(&KPartite::new(&mats).unwrap());
        let x = shift(&root, sense, &mut LapSolver::new()).unwrap();
        let anchor = (0..sizes.len()).find(|&p| sizes[p] == *sizes.iter().max().unwrap()).unwrap();
        let mut rng = Rng::new(seed, 0);
        let y = perturb(x.nodes(), sizes.len(), anchor, 4.0, &mut rng);
        // Row order is immaterial to common-clique detection.
        let mut rows: Vec<_> = y.chunks_exact(sizes.len()).map(<[u32]>::to_vec).collect();
        rng.shuffle(&mut rows);
        check_memetic_operators(&sizes, x.nodes(), &rows.concat(), &root, seed);
    }

    #[test]
    fn memetic_search_properties((sizes, data, sense) in instance(), seed in any::<u64>()) {
        use crate::{Dv, algo::{Rng, dv, memetic, shift}};
        let mats = views(&sizes, &data);
        let root = Quotient::root(&KPartite::new(&mats).unwrap());
        let mut solver = LapSolver::new();
        let anchor = (0..sizes.len()).find(|&p| sizes[p] == *sizes.iter().max().unwrap()).unwrap();
        for kind in [Dv::One, Dv::Subsets] {
            let opts = Options { sense, seed, dv: kind, max_iter: 4, population: Some(4), ..Options::default() };
            let start = shift(&root, sense, &mut solver).unwrap();
            let start = dv::improve(&root, start, kind, dv::CONVERGE, sense, &mut solver).unwrap();
            let mut best = start.clone();
            let mut rng = Rng::new(seed, 0);
            if start.n_nodes(0) >= 2 {
                for _ in 1..4 {
                    let nodes = memetic::perturb(start.nodes(), sizes.len(), anchor, 0.2, &mut rng);
                    let q = Quotient::from_nodes(&root, nodes).unwrap();
                    let q = dv::improve(&root, q, kind, dv::CONVERGE, sense, &mut solver).unwrap();
                    if better(q.score, best.score, sense) { best = q; }
                }
            }
            let zero = k_assignment(&mats, Algo::Memetic, &Options { max_iter: 0, ..opts.clone() }).unwrap();
            prop_assert_eq!(&zero, &best.reconstruct());
            let current = memetic::memetic(&root, &opts, &mut solver).unwrap();
            let sol = k_assignment(&mats, Algo::Memetic, &opts).unwrap();
            check_valid(&sol, &sizes, &mats);
            prop_assert_eq!(&sol, &current.reconstruct());
            prop_assert!(!better(start.score, sol.cost, sense));
            prop_assert_eq!(&sol, &k_assignment(&mats, Algo::Memetic, &opts).unwrap());
            for mask in dv::masks(sizes.len(), kind).unwrap() {
                let (two, matching) = current.bipartition(&root, mask).unwrap();
                let cross: i64 = matching.iter().map(|&(r,c)| two.matrix(0,1).get(r as usize,c as usize)).sum();
                let (_, best) = two.solve_pair(0, 1, &mut solver, sense).unwrap();
                prop_assert!(!better(best, cross, sense));
            }
            let timed = k_assignment(&mats, Algo::Memetic, &Options {
                time_limit: Some(std::time::Duration::ZERO), max_iter: usize::MAX, ..opts.clone()
            }).unwrap();
            prop_assert_eq!(timed, start.reconstruct());
            let refined = k_assignment(&mats, Algo::Memetic, &Options { refine: Some(kind), ..opts }).unwrap();
            prop_assert_eq!(&refined, &sol);
        }
    }
}

#[test]
fn memetic_padding_collapse_and_options() {
    use crate::algo::{Rng, memetic::perturb};
    for sizes in [
        vec![2, 2, 3],
        vec![1, 4, 2, 3],
        vec![4; 4],
        vec![1; 3],
        vec![2; 3],
    ] {
        let data: Vec<Vec<i64>> = all_pairs(sizes.len())
            .iter()
            .map(|&(a, b)| vec![0; sizes[a] * sizes[b]])
            .collect();
        let mats = views(&sizes, &data);
        let root = Quotient::root(&KPartite::new(&mats).unwrap());
        let x = crate::algo::shift(&root, Sense::Minimize, &mut LapSolver::new()).unwrap();
        let anchor = (0..sizes.len())
            .find(|&p| sizes[p] == *sizes.iter().max().unwrap())
            .unwrap();
        for seed in 0..100 {
            let y = perturb(x.nodes(), sizes.len(), anchor, 4.0, &mut Rng::new(seed, 0));
            check_memetic_operators(&sizes, x.nodes(), &y, &root, seed);
        }
        for population in [2, 3, 4] {
            let opts = Options {
                population: Some(population),
                max_iter: 8,
                ..Options::default()
            };
            check_valid(
                &k_assignment(&mats, Algo::Memetic, &opts).unwrap(),
                &sizes,
                &mats,
            );
        }
        for population in [0, 1] {
            assert_eq!(
                k_assignment(
                    &mats,
                    Algo::Memetic,
                    &Options {
                        population: Some(population),
                        ..Options::default()
                    }
                ),
                Err(KapError::BadOption("population"))
            );
        }
        for population in [None, Some(3)] {
            let before = std::time::Instant::now();
            let opts = Options {
                population,
                time_limit: Some(std::time::Duration::ZERO),
                max_iter: usize::MAX,
                ..Options::default()
            };
            check_valid(
                &k_assignment(&mats, Algo::Memetic, &opts).unwrap(),
                &sizes,
                &mats,
            );
            assert!(before.elapsed() < std::time::Duration::from_secs(5));
        }
    }
    let empty = MatrixView::from_row_major(&[] as &[i64], 0, 0).unwrap();
    assert_eq!(
        k_assignment(
            &vec![empty; 17 * 16 / 2],
            Algo::Memetic,
            &Options {
                dv: crate::Dv::Subsets,
                ..Options::default()
            }
        ),
        Err(KapError::Unsupported("sDV requires k <= 16"))
    );
    // Every descent reaches the unique bipartite optimum, so deduplication
    // leaves a single parent even though N = 2 and the target is larger.
    let data = vec![vec![0, 10, 10, 0]];
    let mats = views(&[2, 2], &data);
    for sense in [Sense::Minimize, Sense::Maximize] {
        for population in [2, 3, 4] {
            let sol = k_assignment(
                &mats,
                Algo::Memetic,
                &Options {
                    sense,
                    population: Some(population),
                    max_iter: 8,
                    ..Options::default()
                },
            )
            .unwrap();
            check_valid(&sol, &[2, 2], &mats);
            assert_eq!(sol.cost, if sense == Sense::Minimize { 0 } else { 20 });
        }
    }
}

#[test]
fn memetic_generation_feature_parity() {
    let sizes = [7, 8, 6, 8];
    let data: Vec<Vec<i64>> = all_pairs(sizes.len())
        .iter()
        .enumerate()
        .map(|(i, &(a, b))| {
            (0..sizes[a] * sizes[b])
                .map(|j| ((i * 47 + j * 31 + j * j * 7) % 97) as i64 - 40)
                .collect()
        })
        .collect();
    let mats = views(&sizes, &data);
    let opts = Options {
        population: Some(5),
        max_iter: 5,
        seed: 196,
        ..Options::default()
    };
    let sol = k_assignment(&mats, Algo::Memetic, &opts).unwrap();
    // The same fixture runs with and without rayon; also check thread counts.
    assert_eq!(
        canonical_cliques(sol.clone()),
        vec![
            (vec![(0, 0), (1, 0), (2, 3), (3, 5)], -119),
            (vec![(0, 1), (1, 7), (2, 0), (3, 7)], -153),
            (vec![(0, 2), (1, 2), (2, 4), (3, 3)], -200),
            (vec![(0, 3), (1, 4), (2, 5), (3, 2)], -106),
            (vec![(0, 4), (1, 1), (3, 4)], -91),
            (vec![(0, 5), (1, 3), (3, 1)], -63),
            (vec![(0, 6), (1, 6), (2, 1), (3, 6)], -27),
            (vec![(1, 5), (2, 2), (3, 0)], -45),
        ]
    );
    #[cfg(feature = "rayon")]
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        assert_eq!(
            pool.install(|| k_assignment(&mats, Algo::Memetic, &opts).unwrap()),
            sol
        );
    }
}
