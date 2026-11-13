use super::*;
use crate::{algo::Rng, linear_sum_assignment};
use proptest::prelude::*;

fn views<A: Copy>(data: &[Vec<A>], n: usize) -> Vec<MatrixView<'_, A>> {
    data.iter().map(|d| view(d, n)).collect()
}

fn random_data(k: usize, n: usize, seed: u64) -> Vec<Vec<i64>> {
    let mut rng = Rng::new(seed, k as u64);
    (0..k * (k - 1) / 2)
        .map(|_| (0..n * n).map(|_| rng.below(100) as i64 - 40).collect())
        .collect()
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    fn visit(p: &mut [usize], pos: usize, out: &mut Vec<Vec<usize>>) {
        if pos == p.len() {
            out.push(p.to_vec());
        } else {
            for j in pos..p.len() {
                p.swap(pos, j);
                visit(p, pos + 1, out);
                p.swap(pos, j);
            }
        }
    }
    let mut out = Vec::new();
    visit(&mut (0..n).collect::<Vec<_>>(), 0, &mut out);
    out
}

/// Fix part zero to the identity and enumerate every permutation of each other
/// part. Independent of the LAP and Lagrangian implementations.
fn brute<A: LowerBoundAcc>(mats: &[MatrixView<'_, A>], sense: Sense) -> A {
    fn visit<A: LowerBoundAcc>(
        mats: &[MatrixView<'_, A>],
        perms: &[Vec<usize>],
        nodes: &mut Vec<Vec<usize>>,
        k: usize,
        sense: Sense,
        best: &mut Option<A>,
    ) {
        if nodes.len() == k {
            let mut cost = A::ZERO;
            let mut m = 0;
            for p in 0..k {
                for q in p + 1..k {
                    for (&i, &j) in nodes[p].iter().zip(&nodes[q]) {
                        cost = cost + mats[m].get(i, j);
                    }
                    m += 1;
                }
            }
            if best.is_none_or(|old| crate::algo::better(cost, old, sense)) {
                *best = Some(cost);
            }
        } else {
            for p in perms {
                nodes.push(p.clone());
                visit(mats, perms, nodes, k, sense, best);
                nodes.pop();
            }
        }
    }
    let k = (2..=64).find(|&k| k * (k - 1) / 2 == mats.len()).unwrap();
    let n = mats[0].rows();
    let mut best = None;
    visit(
        mats,
        &permutations(n),
        &mut vec![(0..n).collect()],
        k,
        sense,
        &mut best,
    );
    best.unwrap()
}

fn pairwise<A: LowerBoundAcc>(mats: &[MatrixView<'_, A>], sense: Sense) -> A {
    mats.iter().fold(A::ZERO, |sum, &m| {
        sum + linear_sum_assignment(m, sense).unwrap().cost
    })
}

#[test]
fn random_bounds_against_permutation_oracle() {
    for (k, largest_n) in [(3, 4), (4, 3), (5, 3)] {
        for n in 1..=largest_n {
            for seed in 0..32 {
                let data = random_data(k, n, seed);
                let floats: Vec<Vec<_>> = data
                    .iter()
                    .map(|d| d.iter().map(|&x| x as f64 / 7.0).collect())
                    .collect();
                let (ints, floats) = (views(&data, n), views(&floats, n));
                for sense in [Sense::Minimize, Sense::Maximize] {
                    let ib: i64 = lower_bound(
                        &ints,
                        &LowerBoundOptions {
                            sense,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    let fb: f64 = lower_bound(
                        &floats,
                        &LowerBoundOptions {
                            sense,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    let (io, fo) = (brute(&ints, sense), brute(&floats, sense));
                    let (ip, fp) = (pairwise(&ints, sense), pairwise(&floats, sense));
                    let eps = 1e-9 * (1.0 + fo.abs());
                    match sense {
                        Sense::Minimize => {
                            assert!(
                                ip <= ib && ib <= io,
                                "k={k} n={n} seed={seed}: {ip} <= {ib} <= {io}"
                            );
                            assert!(
                                fp <= fb + eps && fb <= fo + eps,
                                "k={k} n={n} seed={seed}: {fp} <= {fb} <= {fo}"
                            );
                        }
                        Sense::Maximize => {
                            assert!(
                                io <= ib && ib <= ip,
                                "k={k} n={n} seed={seed}: {io} <= {ib} <= {ip}"
                            );
                            assert!(
                                fo <= fb + eps && fb <= fp + eps,
                                "k={k} n={n} seed={seed}: {fo} <= {fb} <= {fp}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn two_parts_are_exact_for_integer_and_float_costs() {
    for n in 0..=8 {
        for seed in 0..16 {
            let data = random_data(2, n, seed);
            let floats: Vec<Vec<_>> = data
                .iter()
                .map(|d| d.iter().map(|&x| x as f64 / 11.0).collect())
                .collect();
            for sense in [Sense::Minimize, Sense::Maximize] {
                let opts = LowerBoundOptions {
                    sense,
                    max_iter: 0,
                    upper: None,
                };
                let im = views(&data, n);
                assert_eq!(lower_bound(&im, &opts).unwrap(), pairwise(&im, sense));
                let fm = views(&floats, n);
                assert_eq!(
                    lower_bound(
                        &fm,
                        &LowerBoundOptions {
                            sense,
                            max_iter: 0,
                            upper: None
                        }
                    )
                    .unwrap(),
                    pairwise(&fm, sense)
                );
            }
        }
    }
    // Other supported input element types widen to the same accumulators.
    let ints = [3i16, 1, 2, 4];
    assert_eq!(
        lower_bound(&[view(&ints, 2)], &Default::default()).unwrap(),
        3i64
    );
    let floats = [3f32, 1.5, 2.25, 4.0];
    assert_eq!(
        lower_bound(&[view(&floats, 2)], &Default::default()).unwrap(),
        3.75f64
    );
    // Preserve LAP's range when its documented 2*n-entry headroom holds.
    let large = i64::MAX / 4 - 1;
    for n in 1..=2 {
        let data = vec![large; n * n];
        for sense in [Sense::Minimize, Sense::Maximize] {
            assert_eq!(
                lower_bound(
                    &[view(&data, n)],
                    &LowerBoundOptions {
                        sense,
                        ..Default::default()
                    }
                )
                .unwrap(),
                n as i64 * large
            );
        }
    }
}

#[test]
fn best_so_far_is_monotone_in_update_budget() {
    for k in 3..=5 {
        for seed in 0..16 {
            let data = random_data(k, 3, seed);
            let floats: Vec<Vec<_>> = data
                .iter()
                .map(|d| d.iter().map(|&x| x as f64 / 7.0).collect())
                .collect();
            let (im, fm) = (views(&data, 3), views(&floats, 3));
            for sense in [Sense::Minimize, Sense::Maximize] {
                let mut previous = None;
                for max_iter in [0, 1, 2, 5, 10, 20, 50, 100, 300] {
                    let ib = lower_bound(
                        &im,
                        &LowerBoundOptions {
                            sense,
                            max_iter,
                            upper: None,
                        },
                    )
                    .unwrap();
                    let fb = lower_bound(
                        &fm,
                        &LowerBoundOptions {
                            sense,
                            max_iter,
                            upper: None,
                        },
                    )
                    .unwrap();
                    if let Some((pi, pf)) = previous {
                        match sense {
                            Sense::Minimize => assert!(ib >= pi && fb >= pf),
                            Sense::Maximize => assert!(ib <= pi && fb <= pf),
                        }
                    }
                    previous = Some((ib, fb));
                }
            }
        }
    }
}

#[test]
fn zero_multipliers_alone_can_be_weaker_than_pairwise() {
    let a = [0i64; 4];
    let bc = [0, 100, 0, 100];
    let mats = [view(&a, 2), view(&bc, 2), view(&bc, 2)];
    assert_eq!(Axial::new(2).evaluate(mats, &[0, 0]).unwrap(), 0);
    assert_eq!(pairwise(&mats, Sense::Minimize), 200);
    assert_eq!(
        lower_bound(
            &mats,
            &LowerBoundOptions {
                max_iter: 0,
                ..Default::default()
            }
        )
        .unwrap(),
        200
    );
}

#[test]
fn planted_solution_is_exact_even_without_updates() {
    for k in 2..=5 {
        let n = 4;
        let data: Vec<Vec<i64>> = (0..k * (k - 1) / 2)
            .map(|_| {
                (0..n * n)
                    .map(|ij| if ij / n == ij % n { 0 } else { 23 })
                    .collect()
            })
            .collect();
        let negative: Vec<Vec<_>> = data
            .iter()
            .map(|d| d.iter().map(|&x| -x).collect())
            .collect();
        let floats: Vec<Vec<_>> = data
            .iter()
            .map(|d| d.iter().map(|&x| x as f64 / 7.0).collect())
            .collect();
        assert_eq!(
            lower_bound(
                &views(&data, n),
                &LowerBoundOptions {
                    max_iter: 0,
                    ..Default::default()
                }
            )
            .unwrap(),
            0
        );
        assert_eq!(
            lower_bound(&views(&floats, n), &Default::default()).unwrap(),
            0.0
        );
        assert_eq!(
            lower_bound(
                &views(&negative, n),
                &LowerBoundOptions {
                    sense: Sense::Maximize,
                    ..Default::default()
                }
            )
            .unwrap(),
            0
        );
    }
}

#[test]
fn integer_evaluation_and_averaging_are_exact_above_two_to_53() {
    let base = (1i64 << 53) + 1;
    for k in 3..=5 {
        for sense in [Sense::Minimize, Sense::Maximize] {
            let small = random_data(k, 2, 95);
            let big: Vec<Vec<_>> = small
                .iter()
                .map(|d| d.iter().map(|&v| v + base).collect())
                .collect();
            let opts = LowerBoundOptions {
                sense,
                ..Default::default()
            };
            let lb = lower_bound(&views(&big, 2), &opts).unwrap();
            let sb = lower_bound(&views(&small, 2), &opts).unwrap();
            assert_eq!(lb, sb + base * k as i64 * (k as i64 - 1));
            let optimum = brute(&views(&big, 2), sense);
            match sense {
                Sense::Minimize => assert!(lb <= optimum),
                Sense::Maximize => assert!(lb >= optimum),
            }
        }
        let planted: Vec<Vec<_>> = (0..k * (k - 1) / 2)
            .map(|_| vec![base, base + 100, base + 100, base])
            .collect();
        assert_eq!(
            lower_bound(&views(&planted, 2), &Default::default()).unwrap(),
            base * k as i64 * (k as i64 - 1)
        );
    }
    for sum in [-11i64, -10, -1, 0, 1, 10, 11, (1i64 << 54) + 1] {
        let expected = ((sum as i128 + 2).div_euclid(3)) as i64;
        assert_eq!(sum.lb_ceil_div(3), expected);
    }
}

#[test]
fn exact_triple_sum_division_and_maximize_sign() {
    let mut nondivisible = false;
    for k in 4..=5 {
        for seed in 0..20 {
            let data = random_data(k, 3, seed);
            let mats = views(&data, 3);
            for sense in [Sense::Minimize, Sense::Maximize] {
                let opts = LowerBoundOptions {
                    sense,
                    max_iter: 7,
                    upper: None,
                };
                let mut sum = 0i64;
                for p in 0..k {
                    for q in p + 1..k {
                        for r in q + 1..k {
                            let triple = [
                                mats[index(k, p, q)],
                                mats[index(k, p, r)],
                                mats[index(k, q, r)],
                            ];
                            sum += lower_bound(&triple, &opts).unwrap();
                        }
                    }
                }
                let d = (k - 2) as i64;
                nondivisible |= sum % d != 0;
                let expected = match sense {
                    Sense::Minimize => sum.lb_ceil_div(k - 2),
                    Sense::Maximize => sum.div_euclid(d),
                };
                assert_eq!(lower_bound(&mats, &opts).unwrap(), expected);
            }
        }
    }
    assert!(
        nondivisible,
        "exercise rounding of nonintegral triple averages"
    );
}

#[test]
fn whole_problem_upper_never_becomes_a_triple_target() {
    // Negative costs make the whole optimum lower than each triple optimum.
    let data = vec![vec![-10i64; 9]; 10];
    let mats = views(&data, 3);
    let opts = LowerBoundOptions {
        upper: Some(-300),
        ..Default::default()
    };
    assert_eq!(lower_bound(&mats, &opts).unwrap(), -300);
    let data = random_data(5, 3, 417);
    let mats = views(&data, 3);
    for sense in [Sense::Minimize, Sense::Maximize] {
        let opts = LowerBoundOptions {
            sense,
            ..Default::default()
        };
        let unbounded = lower_bound(&mats, &opts).unwrap();
        let optimum = brute(&mats, sense);
        let upper = optimum + 1;
        assert_eq!(
            lower_bound(
                &mats,
                &LowerBoundOptions {
                    upper: Some(upper),
                    ..opts
                }
            )
            .unwrap(),
            unbounded.min(upper)
        );
    }
}

#[test]
fn validates_shapes_counts_options_and_costs() {
    assert_eq!(
        lower_bound::<i64>(&[], &Default::default()),
        Err(KapError::BadCount(0))
    );
    let data = [0i64; 6];
    let rect = MatrixView::from_row_major(&data, 2, 3).unwrap();
    assert!(matches!(
        lower_bound(&[rect], &Default::default()),
        Err(KapError::Unsupported(_))
    ));
    let square = view(&data, 2);
    assert!(matches!(
        lower_bound(&[square, square, rect], &Default::default()),
        Err(KapError::ShapeMismatch { .. })
    ));
    assert_eq!(
        lower_bound(&[square; 2], &Default::default()),
        Err(KapError::BadCount(2))
    );
    let unequal = [square, rect, rect];
    assert!(matches!(
        lower_bound(&unequal, &Default::default()),
        Err(KapError::Unsupported(_))
    ));
    for k in 2..=5 {
        assert_eq!(
            lower_bound(
                &vec![view::<i64>(&[], 0); k * (k - 1) / 2],
                &Default::default()
            )
            .unwrap(),
            0
        );
    }
    for x in [f64::NAN, f64::NEG_INFINITY] {
        assert!(matches!(
            lower_bound(&[view(&[x], 1)], &Default::default()),
            Err(KapError::Lap(LapError::InvalidCost(0, 0)))
        ));
    }
    assert!(
        lower_bound(
            &[view(&[0f64], 1)],
            &LowerBoundOptions {
                upper: Some(f64::NAN),
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        lower_bound(
            &[view(&[f64::INFINITY], 1)],
            &LowerBoundOptions {
                sense: Sense::Maximize,
                ..Default::default()
            }
        )
        .is_err()
    );
    for x in [i64::MIN, i64::MAX, i64::MAX / 4 - 1] {
        for sense in [Sense::Minimize, Sense::Maximize] {
            assert!(
                lower_bound(
                    &[view(&[x], 1); 3],
                    &LowerBoundOptions {
                        sense,
                        ..Default::default()
                    }
                )
                .is_err()
            );
        }
    }
    assert!(i64::MAX.lb_add(1).is_err());
    assert!(i64::MIN.lb_sub(1).is_err());
    assert!(f64::MAX.lb_add(f64::MAX).is_err());
}

#[test]
fn strided_views_and_forbidden_entries_follow_lap_conventions() {
    let data = [1i64, 8, 9, 2];
    let flipped = MatrixView::from_strided(&data, 3, 2, 2, -2, -1).unwrap();
    assert_eq!(lower_bound(&[flipped; 3], &Default::default()).unwrap(), 9);
    let diagonal = [0.0, f64::INFINITY, f64::INFINITY, 0.0];
    assert_eq!(
        lower_bound(&[view(&diagonal, 2); 3], &Default::default()).unwrap(),
        0.0
    );
    let negative = [0.0, f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0];
    assert_eq!(
        lower_bound(
            &[view(&negative, 2); 3],
            &LowerBoundOptions {
                sense: Sense::Maximize,
                ..Default::default()
            }
        )
        .unwrap(),
        0.0
    );
    let forbidden = [f64::INFINITY; 4];
    assert_eq!(
        lower_bound(&[view(&forbidden, 2); 3], &Default::default()),
        Err(KapError::Lap(LapError::Infeasible))
    );
}

#[test]
fn feasible_forbidden_instance_does_not_require_a_successful_heuristic() {
    let inf = f64::INFINITY;
    let a = [0.0, 100.0, 100.0, 100.0, 0.0, 100.0, 100.0, 100.0, 0.0];
    let b = [0.0, 100.0, inf, 0.0, inf, 100.0, inf, 0.0, 0.0];
    let c = [0.0, inf, 100.0, 0.0, 100.0, inf, inf, 0.0, 0.0];
    let mats = [view(&a, 3), view(&b, 3), view(&c, 3)];
    assert_eq!(
        k_assignment(&mats, Algo::Dv1, &Options::default()),
        Err(KapError::Lap(LapError::Infeasible))
    );
    let optimum = brute(&mats, Sense::Minimize);
    assert!(optimum.is_finite());
    let bound = lower_bound(&mats, &Default::default()).unwrap();
    assert!(pairwise(&mats, Sense::Minimize) <= bound && bound <= optimum);
}

#[test]
fn subgradient_improves_on_pairwise_for_some_instances() {
    let mut improved = false;
    let mut nonoptimal_target = false;
    for seed in 0..64 {
        let data = random_data(3, 4, seed);
        let mats = views(&data, 4);
        let initial = lower_bound(
            &mats,
            &LowerBoundOptions {
                max_iter: 0,
                ..Default::default()
            },
        )
        .unwrap();
        let final_bound = lower_bound(&mats, &Default::default()).unwrap();
        improved |= final_bound > initial;
        let optimum = brute(&mats, Sense::Minimize);
        let target = k_assignment(&mats, Algo::Dv1, &Default::default())
            .unwrap()
            .cost;
        nonoptimal_target |= target > optimum;
        assert!(final_bound <= optimum);
    }
    assert!(
        improved,
        "subgradient updates must actually strengthen the bound"
    );
    assert!(
        nonoptimal_target,
        "exercise genuinely nonoptimal Polyak targets"
    );
}

#[test]
fn deterministic_fixture_with_or_without_rayon() {
    let data = random_data(5, 3, 987);
    let floats: Vec<Vec<_>> = data
        .iter()
        .map(|d| d.iter().map(|&x| x as f64 / 7.0).collect())
        .collect();
    let (im, fm) = (views(&data, 3), views(&floats, 3));
    let integer = lower_bound(&im, &Default::default()).unwrap();
    let float = lower_bound(&fm, &Default::default()).unwrap();
    // Same constants are asserted in serial, release and rayon builds.
    assert_eq!(integer, -138);
    assert_eq!(float.to_bits(), 13_849_627_736_219_520_047);
    assert!(integer <= brute(&im, Sense::Minimize));
    assert!(float <= brute(&fm, Sense::Minimize) + 1e-9);
    #[cfg(feature = "rayon")]
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        assert_eq!(
            pool.install(|| lower_bound(&im, &Default::default()).unwrap()),
            integer
        );
        assert_eq!(
            pool.install(|| lower_bound(&fm, &Default::default()).unwrap())
                .to_bits(),
            float.to_bits()
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(80))]
    #[test]
    fn arbitrary_small_bounds_are_valid(
        k in 3usize..=5,
        n in 1usize..=3,
        seed in any::<u64>(),
        maximize in any::<bool>(),
        max_iter in 0usize..=30,
    ) {
        let data = random_data(k, n, seed);
        let mats = views(&data, n);
        let sense = if maximize { Sense::Maximize } else { Sense::Minimize };
        let bound = lower_bound(&mats, &LowerBoundOptions { sense, max_iter, upper: None }).unwrap();
        let optimum = brute(&mats, sense);
        let pair = pairwise(&mats, sense);
        if maximize {
            prop_assert!(optimum <= bound && bound <= pair);
        } else {
            prop_assert!(pair <= bound && bound <= optimum);
        }
    }
}
