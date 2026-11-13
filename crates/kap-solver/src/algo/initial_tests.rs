use super::*;
use proptest::prelude::*;

fn views<'a, T: Cost>(sizes: &[usize], data: &'a [Vec<T>]) -> Vec<MatrixView<'a, T>> {
    (0..sizes.len())
        .flat_map(|a| (a + 1..sizes.len()).map(move |b| (a, b)))
        .enumerate()
        .map(|(i, (a, b))| MatrixView::from_row_major(&data[i], sizes[a], sizes[b]).unwrap())
        .collect()
}

fn canonical<A: Acc>(sol: &KSolution<A>) -> Vec<Vec<(usize, usize)>> {
    let mut rows = sol.cliques.clone();
    rows.extend(sol.free.iter().map(|&v| vec![v]));
    rows.sort();
    rows
}

fn check_coverage<A: Acc>(sol: &KSolution<A>, sizes: &[usize]) {
    assert_eq!(
        sol.cliques.len() + sol.free.len(),
        *sizes.iter().max().unwrap()
    );
    let mut vertices = Vec::new();
    for row in canonical(sol) {
        assert!(row.windows(2).all(|w| w[0].0 < w[1].0));
        vertices.extend(row);
    }
    vertices.sort();
    assert_eq!(
        vertices,
        sizes
            .iter()
            .enumerate()
            .flat_map(|(p, &n)| (0..n).map(move |v| (p, v)))
            .collect::<Vec<_>>()
    );
}

#[test]
fn warm_start_quality_and_zero_ils() {
    for k in 3..=5 {
        for unequal in [false, true] {
            let sizes: Vec<_> = (0..k)
                .map(|p| 3 + if unequal { p % 3 } else { 0 })
                .collect();
            let data: Vec<Vec<i64>> = (0..k)
                .flat_map(|a| (a + 1..k).map(move |b| (a, b)))
                .enumerate()
                .map(|(i, (a, b))| {
                    (0..sizes[a] * sizes[b])
                        .map(|j| ((i * 47 + j * 31 + j * j * 7) % 97) as i64 - 40)
                        .collect()
                })
                .collect();
            let mats = views(&sizes, &data);
            let floats: Vec<Vec<f64>> = data
                .iter()
                .map(|m| m.iter().map(|&v| v as f64 / 8.0).collect())
                .collect();
            let float_mats = views(&sizes, &floats);
            for sense in [Sense::Minimize, Sense::Maximize] {
                for kind in [Dv::One, Dv::Subsets] {
                    let opts = Options {
                        sense,
                        refine: Some(kind),
                        dv: kind,
                        max_iter: dv::CONVERGE,
                        seed: 93,
                        n_trials: 4,
                        ..Options::default()
                    };
                    let initial = k_assignment(
                        &mats,
                        Algo::Am,
                        &Options {
                            sense,
                            ..Options::default()
                        },
                    )
                    .unwrap();
                    let local =
                        k_assignment_from(&mats, &initial.cliques, Algo::Am, &opts).unwrap();
                    check_coverage(&local, &sizes);
                    assert!(!better(initial.cost, local.cost, sense));
                    let again = k_assignment_from(&mats, &local.cliques, Algo::Am, &opts).unwrap();
                    assert_eq!(canonical(&again), canonical(&local));
                    assert_eq!(again.cost, local.cost);
                    let zero = k_assignment_from(
                        &mats,
                        &initial.cliques,
                        Algo::Ils,
                        &Options {
                            max_iter: 0,
                            refine: None,
                            ..opts.clone()
                        },
                    )
                    .unwrap();
                    assert_eq!(zero, local);
                    let ils = k_assignment_from(
                        &mats,
                        &initial.cliques,
                        Algo::Ils,
                        &Options {
                            max_iter: 60,
                            refine: None,
                            ..opts.clone()
                        },
                    )
                    .unwrap();
                    check_coverage(&ils, &sizes);
                    assert!(!better(local.cost, ils.cost, sense));
                    for (algo, iterations) in [(Algo::Am, dv::CONVERGE), (Algo::Ils, 60)] {
                        let sol = k_assignment_from(
                            &float_mats,
                            &initial.cliques,
                            algo,
                            &Options {
                                max_iter: iterations,
                                ..opts.clone()
                            },
                        )
                        .unwrap();
                        check_coverage(&sol, &sizes);
                        assert!(!better(initial.cost as f64 / 8.0, sol.cost, sense));
                    }
                }
            }
        }
    }
}

#[test]
fn warm_start_validation_and_algorithms() {
    let data = vec![vec![0i64; 4]; 3];
    let mats = views(&[2; 3], &data);
    for cliques in [
        vec![vec![]],
        vec![vec![(3, 0)]],
        vec![vec![(0, 2)]],
        vec![vec![(usize::MAX, 0)]],
        vec![vec![(0, usize::MAX)]],
        vec![vec![(0, 0), (0, 1)]],
        vec![vec![(0, 0)], vec![(0, 0)]],
    ] {
        for algo in [Algo::Am, Algo::Ils] {
            assert_eq!(
                k_assignment_from(&mats, &cliques, algo, &Options::default()),
                Err(KapError::BadMatching)
            );
        }
    }
    for algo in [Algo::Cm, Algo::Em, Algo::Fm, Algo::Memetic] {
        assert_eq!(
            k_assignment_from(&mats, &[], algo, &Options::default()),
            Err(KapError::BadOption("initial"))
        );
    }
    let initial = vec![vec![(0, 0), (1, 1), (2, 0)], vec![(0, 1), (1, 0), (2, 1)]];
    for algo in [
        Algo::Am,
        Algo::Bm,
        Algo::Dm,
        Algo::Shift,
        Algo::Dv1,
        Algo::DvS,
    ] {
        let sol = k_assignment_from(
            &mats,
            &initial,
            algo,
            &Options {
                max_iter: 0,
                ..Options::default()
            },
        )
        .unwrap();
        assert_eq!(sol.cliques, initial);
    }
}

#[test]
fn warm_start_singletons_and_empty_parts() {
    for sizes in [vec![3; 3], vec![1, 4, 2, 3], vec![0, 3, 0], vec![0; 3]] {
        let data: Vec<_> = (0..sizes.len())
            .flat_map(|a| (a + 1..sizes.len()).map(move |b| (a, b)))
            .map(|(a, b)| vec![0i64; sizes[a] * sizes[b]])
            .collect();
        let mats = views(&sizes, &data);
        // Match-only representations omit free singletons, which are restored.
        let initial: Vec<Vec<_>> = (0..*sizes.iter().max().unwrap())
            .map(|v| {
                sizes
                    .iter()
                    .enumerate()
                    .filter(|&(_, &n)| v < n)
                    .map(|(p, _)| (p, v))
                    .collect::<Vec<_>>()
            })
            .filter(|row| row.len() > 1)
            .collect();
        for kind in [Dv::One, Dv::Subsets] {
            for sense in [Sense::Minimize, Sense::Maximize] {
                for algo in [Algo::Am, Algo::Ils] {
                    let sol = k_assignment_from(
                        &mats,
                        &initial,
                        algo,
                        &Options {
                            sense,
                            refine: Some(kind),
                            dv: kind,
                            max_iter: 60,
                            n_trials: 3,
                            ..Options::default()
                        },
                    )
                    .unwrap();
                    check_coverage(&sol, &sizes);
                    assert_eq!(
                        sol.cliques.len() + sol.free.len(),
                        *sizes.iter().max().unwrap()
                    );
                }
            }
        }
    }
}

#[test]
fn warm_start_requires_complete_solution() {
    for k in 2..=5 {
        let sizes = vec![2; k];
        let data = vec![vec![0i64; 4]; k * (k - 1) / 2];
        let mats = views(&sizes, &data);
        let singletons: Vec<_> = (0..k)
            .flat_map(|p| (0..2).map(move |v| vec![(p, v)]))
            .collect();
        for initial in [&singletons, &vec![]] {
            for algo in [Algo::Am, Algo::Ils] {
                let err = k_assignment_from(&mats, initial, algo, &Options::default()).unwrap_err();
                assert_eq!(
                    err,
                    KapError::IncompleteInitial {
                        expected: 2,
                        actual: 2 * k
                    }
                );
                assert_eq!(
                    err.to_string(),
                    format!(
                        "initial must be a complete solution: 2 nodes expected, {} given",
                        2 * k
                    )
                );
            }
        }
    }
    let data = vec![vec![0i64; 4]; 3];
    let mats = views(&[2; 3], &data);
    let unmergeable = vec![
        vec![(0, 0), (1, 0)],
        vec![(0, 1), (2, 1)],
        vec![(1, 1), (2, 0)],
    ];
    assert_eq!(
        k_assignment_from(&mats, &unmergeable, Algo::Am, &Options::default()),
        Err(KapError::IncompleteInitial {
            expected: 2,
            actual: 3
        })
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(40))]
    #[test]
    fn arbitrary_complete_warm_starts(seed in any::<u64>(), k in 3usize..=5, maximize in any::<bool>()) {
        let mut rng = super::Rng::new(seed, 0);
        let sizes: Vec<_> = (0..k).map(|_| rng.below(5)).collect();
        let data: Vec<Vec<i64>> = (0..k)
            .flat_map(|a| (a + 1..k).map(move |b| (a, b)))
            .map(|(a, b)| (0..sizes[a] * sizes[b]).map(|_| rng.below(60) as i64 - 20).collect())
            .collect();
        let mats = views(&sizes, &data);
        let recompute = |sol: &KSolution<i64>| -> i64 {
            sol.cliques.iter().map(|row| {
                row.iter().enumerate().map(|(i, &(a, v))| {
                    row[i + 1..].iter().map(|&(b, w)| {
                        let index = k * a + b - a * (a + 1) / 2 - a - 1;
                        mats[index].get(v, w)
                    }).sum::<i64>()
                }).sum::<i64>()
            }).sum()
        };
        let mut cliques = vec![Vec::new(); *sizes.iter().max().unwrap()];
        // Independent slot permutations prevent two vertices of a part in one
        // clique; a largest part fills every slot, ensuring a complete start.
        for (p, &size) in sizes.iter().enumerate() {
            let mut slots: Vec<_> = (0..cliques.len()).collect();
            rng.shuffle(&mut slots);
            for (v, slot) in slots.into_iter().take(size).enumerate() {
                cliques[slot].push((p, v));
            }
        }
        // Free singletons need not be listed in the supplied matches.
        cliques.retain(|row| row.len() > 1);
        let sense = if maximize { Sense::Maximize } else { Sense::Minimize };
        let initial = k_assignment_from(&mats, &cliques, Algo::Am, &Options { max_iter: 0, ..Options::default() }).unwrap();
        prop_assert_eq!(initial.cost, recompute(&initial));
        for kind in [Dv::One, Dv::Subsets] {
            let opts = Options { sense, seed, refine: Some(kind), dv: kind, max_iter: 60, n_trials: 3, ..Options::default() };
            let local = k_assignment_from(&mats, &cliques, Algo::Am, &opts).unwrap();
            check_coverage(&local, &sizes);
            prop_assert_eq!(local.cost, recompute(&local));
            prop_assert!(!better(initial.cost, local.cost, sense));
            let searched = k_assignment_from(&mats, &cliques, Algo::Ils, &opts).unwrap();
            check_coverage(&searched, &sizes);
            prop_assert_eq!(searched.cost, recompute(&searched));
            prop_assert!(!better(local.cost, searched.cost, sense));
            prop_assert_eq!(&searched, &k_assignment_from(&mats, &cliques, Algo::Ils, &opts).unwrap());
        }
    }
}
