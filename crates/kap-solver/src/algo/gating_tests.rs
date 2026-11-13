use super::Rng;
use super::*;
use proptest::prelude::*;

const ALGOS: [Algo; 7] = [
    Algo::Am,
    Algo::Bm,
    Algo::Cm,
    Algo::Dm,
    Algo::Em,
    Algo::Fm,
    Algo::Shift,
];
const SENSES: [Sense; 2] = [Sense::Minimize, Sense::Maximize];

fn options(tau: f64, sense: Sense) -> Options {
    Options {
        threshold: Some(tau),
        sense,
        max_iter: 6,
        n_trials: 2,
        seed: 31,
        ..Options::default()
    }
}

fn views<'a, T: Copy>(sizes: &[usize], data: &'a [Vec<T>]) -> Vec<MatrixView<'a, T>> {
    let mut mats = Vec::new();
    for a in 0..sizes.len() {
        for b in a + 1..sizes.len() {
            mats.push(MatrixView::from_row_major(&data[mats.len()], sizes[a], sizes[b]).unwrap());
        }
    }
    mats
}

fn split<T: Clone>(sizes: &[usize], flat: &[T]) -> Vec<Vec<T>> {
    let mut offset = 0;
    let mut data = Vec::new();
    for a in 0..sizes.len() {
        for b in a + 1..sizes.len() {
            let end = offset + sizes[a] * sizes[b];
            data.push(flat[offset..end].to_vec());
            offset = end;
        }
    }
    data
}

// Independent indexing and exhaustive oracles: neither uses quotient graphs,
// LAP, nor any solver objective/partition helper.
fn edge<T: Copy>(sizes: &[usize], data: &[Vec<T>], x: (usize, usize), y: (usize, usize)) -> T {
    let ((a, r), (b, c)) = if x.0 < y.0 { (x, y) } else { (y, x) };
    let mut id = 0;
    for p in 0..a {
        id += sizes.len() - p - 1;
    }
    id += b - a - 1;
    data[id][r * sizes[b] + c]
}

fn partial_matching_oracle(data: &[f64], rows: usize, cols: usize, sense: Sense) -> f64 {
    fn visit(data: &[f64], cols: usize, row: usize, used: u32, sense: Sense) -> f64 {
        if row * cols == data.len() {
            return 0.0;
        }
        let mut best = visit(data, cols, row + 1, used, sense);
        for c in 0..cols {
            let value = data[row * cols + c];
            if used & (1 << c) == 0 && value.is_finite() {
                let candidate = value + visit(data, cols, row + 1, used | (1 << c), sense);
                if (sense == Sense::Minimize && candidate < best)
                    || (sense == Sense::Maximize && candidate > best)
                {
                    best = candidate;
                }
            }
        }
        best
    }
    if rows == 0 || cols == 0 {
        0.0
    } else {
        visit(data, cols, 0, 0, sense)
    }
}

fn partition_oracle(sizes: &[usize], data: &[Vec<i64>], tau: i64, sense: Sense) -> i64 {
    struct Search<'a> {
        sizes: &'a [usize],
        data: &'a [Vec<i64>],
        vertices: Vec<(usize, usize)>,
        tau: i64,
        sense: Sense,
        best: i64,
    }
    impl Search<'_> {
        fn visit(&mut self, next: usize, cliques: &mut Vec<Vec<(usize, usize)>>, score: i64) {
            if next == self.vertices.len() {
                if (self.sense == Sense::Minimize && score < self.best)
                    || (self.sense == Sense::Maximize && score > self.best)
                {
                    self.best = score;
                }
                return;
            }
            let vertex = self.vertices[next];
            for i in 0..cliques.len() {
                if cliques[i].iter().any(|v| v.0 == vertex.0) {
                    continue;
                }
                let added: i64 = cliques[i]
                    .iter()
                    .map(|&v| edge(self.sizes, self.data, vertex, v) - self.tau)
                    .sum();
                cliques[i].push(vertex);
                self.visit(next + 1, cliques, score + added);
                cliques[i].pop();
            }
            cliques.push(vec![vertex]);
            self.visit(next + 1, cliques, score);
            cliques.pop();
        }
    }
    let mut search = Search {
        sizes,
        data,
        vertices: sizes
            .iter()
            .enumerate()
            .flat_map(|(p, &n)| (0..n).map(move |v| (p, v)))
            .collect(),
        tau,
        sense,
        best: 0,
    };
    search.visit(0, &mut Vec::new(), 0);
    search.best
}

fn validate<T: Acc>(sizes: &[usize], data: &[Vec<T>], sol: &KSolution<T>) -> usize {
    assert_eq!(sol.cliques.len(), sol.costs.len());
    let mut used: Vec<_> = sizes.iter().map(|&n| vec![false; n]).collect();
    let mut total = T::ZERO;
    let mut pairs = 0;
    for (clique, &cost) in sol.cliques.iter().zip(&sol.costs) {
        assert!(clique.len() >= 2);
        assert!(clique.windows(2).all(|w| w[0].0 < w[1].0));
        let mut sum = T::ZERO;
        for (i, &(p, v)) in clique.iter().enumerate() {
            assert!(!used[p][v]);
            used[p][v] = true;
            for &other in &clique[i + 1..] {
                let value = edge(sizes, data, (p, v), other);
                assert!(value.is_finite());
                sum = sum + value;
                pairs += 1;
            }
        }
        assert_eq!(sum, cost);
        total = total + cost;
    }
    for &(p, v) in &sol.free {
        assert!(!used[p][v]);
        used[p][v] = true;
    }
    assert!(used.iter().flatten().all(|&v| v));
    assert_eq!(total, sol.cost);
    pairs
}

fn instance(k: usize, largest: usize) -> impl Strategy<Value = (Vec<usize>, Vec<Vec<i64>>, i64)> {
    prop::collection::vec(1usize..=largest, k)
        .prop_flat_map(|sizes| {
            let entries = sizes
                .iter()
                .enumerate()
                .map(|(a, n)| n * sizes[a + 1..].iter().sum::<usize>())
                .sum::<usize>();
            (
                Just(sizes),
                prop::collection::vec(-6i64..=6, entries),
                -7i64..=7,
            )
        })
        .prop_map(|(sizes, flat, tau)| {
            let data = split(&sizes, &flat);
            (sizes, data, tau)
        })
}

fn check_partition_case(sizes: &[usize], data: &[Vec<i64>], tau: i64) {
    let mats = views(sizes, data);
    for sense in SENSES {
        let optimum = partition_oracle(sizes, data, tau, sense);
        for algo in ALGOS {
            let sol = k_assignment(&mats, algo, &options(tau as f64, sense)).unwrap();
            let pairs = validate(sizes, data, &sol);
            let shifted = sol.cost - tau * pairs as i64;
            match sense {
                Sense::Minimize => assert!(shifted >= optimum, "{algo:?}"),
                Sense::Maximize => assert!(shifted <= optimum, "{algo:?}"),
            }
            if sizes.len() == 2 {
                assert_eq!(shifted, optimum, "{algo:?}");
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(40))]

    #[test]
    fn clipped_lap_integer(
        (rows, cols, data) in (0usize..=4, 0usize..=4).prop_flat_map(|(r,c)|
            (Just(r), Just(c), prop::collection::vec(-6i64..=6, r*c)))
    ) {
        let floats: Vec<_> = data.iter().map(|&v| v as f64).collect();
        let mats = [MatrixView::from_row_major(&data, rows, cols).unwrap()];
        for sense in SENSES {
            let sol = k_assignment(&mats, Algo::Am, &options(0.0, sense)).unwrap();
            prop_assert_eq!(sol.cost as f64, partial_matching_oracle(&floats, rows, cols, sense));
            validate(&[rows, cols], std::slice::from_ref(&data), &sol);
            for clique in &sol.cliques {
                let value = data[clique[0].1 * cols + clique[1].1];
                prop_assert!(match sense { Sense::Minimize => value < 0, Sense::Maximize => value > 0 }, "pair must improve");
            }
        }
    }

    #[test]
    fn clipped_lap_float(
        (rows, cols, entries) in (0usize..=4, 0usize..=4).prop_flat_map(|(r,c)|
            (Just(r), Just(c), prop::collection::vec((-6i32..=6, any::<bool>()), r*c)))
    ) {
        for sense in SENSES {
            let data: Vec<_> = entries.iter().map(|&(v, forbidden)| if forbidden {
                if sense == Sense::Minimize { f64::INFINITY } else { f64::NEG_INFINITY }
            } else { v as f64 / 4.0 }).collect();
            let mats = [MatrixView::from_row_major(&data, rows, cols).unwrap()];
            let sol = k_assignment(&mats, Algo::Am, &options(0.0, sense)).unwrap();
            prop_assert_eq!(sol.cost, partial_matching_oracle(&data, rows, cols, sense));
            validate(&[rows, cols], std::slice::from_ref(&data), &sol);
            for clique in &sol.cliques {
                let value = data[clique[0].1 * cols + clique[1].1];
                prop_assert!(value.is_finite());
                prop_assert!(match sense { Sense::Minimize => value < 0.0, Sense::Maximize => value > 0.0 }, "pair must improve");
            }
        }
    }

    #[test]
    fn partitions_two((sizes, data, tau) in instance(2, 4)) {
        check_partition_case(&sizes, &data, tau);
    }

    #[test]
    fn partitions_three((sizes, data, tau) in instance(3, 3)) {
        check_partition_case(&sizes, &data, tau);
    }

    #[test]
    fn partitions_four((sizes, data, tau) in instance(4, 2)) {
        check_partition_case(&sizes, &data, tau);
    }

    #[test]
    fn no_improving_edges((sizes, data, _) in instance(3, 3)) {
        let mats = views(&sizes, &data);
        for sense in SENSES {
            let tau = if sense == Sense::Minimize { -7.0 } else { 7.0 };
            for algo in ALGOS {
                let sol = k_assignment(&mats, algo, &options(tau, sense)).unwrap();
                prop_assert!(sol.cliques.is_empty());
                prop_assert_eq!(sol.free.len(), sizes.iter().sum::<usize>());
                prop_assert_eq!(sol.cost, 0);
                validate(&sizes, &data, &sol);
            }
        }
    }

    #[test]
    fn forbidden_float_cliques(
        (sizes, data, tau) in instance(3, 4),
        forbidden in prop::collection::vec(any::<bool>(), 48)
    ) {
        for sense in SENSES {
            let mut i = 0;
            let floats: Vec<Vec<_>> = data.iter().map(|m| m.iter().map(|&v| {
                let value = if forbidden[i] {
                    if sense == Sense::Minimize { f64::INFINITY } else { f64::NEG_INFINITY }
                } else { v as f64 / 4.0 };
                i += 1;
                value
            }).collect()).collect();
            let mats = views(&sizes, &floats);
            for algo in ALGOS {
                let sol = k_assignment(&mats, algo, &options(tau as f64 / 4.0, sense)).unwrap();
                prop_assert!(sol.cost.is_finite());
                validate(&sizes, &floats, &sol);
            }
        }
    }

    #[test]
    fn sign_symmetry((sizes, data, tau) in instance(3, 3)) {
        let negated: Vec<Vec<_>> = data.iter().map(|m| m.iter().map(|&v| -v).collect()).collect();
        for algo in ALGOS {
            let max = k_assignment(&views(&sizes, &data), algo, &options(tau as f64, Sense::Maximize)).unwrap();
            let min = k_assignment(&views(&sizes, &negated), algo, &options(-tau as f64, Sense::Minimize)).unwrap();
            let max_pairs = validate(&sizes, &data, &max);
            let min_pairs = validate(&sizes, &negated, &min);
            prop_assert_eq!(max.cost - tau * max_pairs as i64, -(min.cost + tau * min_pairs as i64));
        }
    }
}

#[test]
fn complete_matching_unchanged_when_every_edge_improves() {
    let mut rng = Rng::new(0x57ab_19ef, 0);
    for (k, n) in [(2, 4), (3, 4), (4, 3)] {
        let sizes = vec![n; k];
        let data: Vec<Vec<i64>> = (0..k * (k - 1) / 2)
            .map(|_| (0..n * n).map(|_| rng.below(1_000_000) as i64).collect())
            .collect();
        let mats = views(&sizes, &data);
        // Dm is excluded: it ranks intermediate merges by score, and the shift
        // adds -tau per pair, so merges that fix more pairs rank differently.
        // The other algorithms compare states with equal pair counts.
        for algo in ALGOS.into_iter().filter(|&algo| algo != Algo::Dm) {
            let opts = options(1_000_001.0, Sense::Minimize);
            let gated = k_assignment(&mats, algo, &opts).unwrap();
            let ungated = k_assignment(
                &mats,
                algo,
                &Options {
                    threshold: None,
                    ..opts
                },
            )
            .unwrap();
            assert_eq!(gated.cost, ungated.cost, "{algo:?}, k={k}");
        }
    }
}

#[test]
fn all_forbidden_and_zero_edges_stay_free_without_lap() {
    use crate::kpartite::LAP_CALLS;
    for sense in SENSES {
        for value in [
            0.0,
            if sense == Sense::Minimize {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            },
        ] {
            let sizes = [2, 3, 4];
            let data = split(&sizes, &vec![value; 26]);
            for algo in ALGOS {
                LAP_CALLS.with(|n| n.set(0));
                let sol = k_assignment(&views(&sizes, &data), algo, &options(0.0, sense)).unwrap();
                assert!(sol.cliques.is_empty());
                assert_eq!(sol.free.len(), 9);
                assert_eq!(sol.cost, 0.0);
                LAP_CALLS.with(|n| assert_eq!(n.get(), 0));
                validate(&sizes, &data, &sol);
            }
        }
    }
}

#[test]
fn recost_uses_original_float_entries() {
    // Subtracting this threshold loses the original small costs entirely.
    let data = vec![vec![1.0], vec![2.0], vec![3.0]];
    for algo in ALGOS {
        let sol = k_assignment(
            &views(&[1, 1, 1], &data),
            algo,
            &options(1e20, Sense::Minimize),
        )
        .unwrap();
        assert_eq!(sol.cost, 6.0);
        assert_eq!(sol.costs, vec![6.0]);
    }
}

#[test]
fn threshold_conversion_boundaries_and_overflow() {
    assert_eq!(
        <i64 as Acc>::from_f64_exact(i64::MIN as f64),
        Some(i64::MIN)
    );
    assert_eq!(<i64 as Acc>::from_f64_exact(-(i64::MIN as f64)), None);
    let largest = f64::from_bits((-(i64::MIN as f64)).to_bits() - 1);
    assert_eq!(<i64 as Acc>::from_f64_exact(largest), Some(i64::MAX - 1023));
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.5] {
        assert_eq!(<i64 as Acc>::from_f64_exact(value), None);
    }
    for (value, tau) in [(0, i64::MIN as f64), (i64::MIN, 1.0), (i64::MAX, -1.0)] {
        let data = [value];
        let mats = [MatrixView::from_row_major(&data, 1, 1).unwrap()];
        assert_eq!(
            k_assignment(&mats, Algo::Am, &options(tau, Sense::Minimize)),
            Err(KapError::Unsupported("cost overflow"))
        );
    }
    // The full integer range has no forbidden sentinel; non-improving extreme
    // values are clipped before the LAP's integer headroom validation.
    for (value, sense) in [(i64::MAX, Sense::Minimize), (i64::MIN, Sense::Maximize)] {
        let data = [value];
        let mats = [MatrixView::from_row_major(&data, 1, 1).unwrap()];
        assert_eq!(
            k_assignment(&mats, Algo::Am, &options(0.0, sense))
                .unwrap()
                .cost,
            0
        );
    }
}

#[test]
fn invalid_entries_thresholds_and_unsupported_options() {
    for sense in SENSES {
        let wrong_inf = if sense == Sense::Minimize {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        for invalid in [f64::NAN, wrong_inf] {
            let data = [0.0, 0.0, invalid, invalid];
            let mats = [MatrixView::from_row_major(&data, 2, 2).unwrap()];
            assert_eq!(
                k_assignment(&mats, Algo::Am, &options(0.0, sense)),
                Err(KapError::Lap(crate::LapError::InvalidCost(1, 0)))
            );
        }
        let data = [1.0];
        let mats = [MatrixView::from_row_major(&data, 1, 1).unwrap()];
        for tau in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                k_assignment(&mats, Algo::Am, &options(tau, sense)),
                Err(KapError::BadOption("threshold"))
            );
        }
        let forbidden = [-wrong_inf];
        let mats = [MatrixView::from_row_major(&forbidden, 1, 1).unwrap()];
        assert_eq!(
            k_assignment(
                &mats,
                Algo::Am,
                &Options {
                    sense,
                    ..Options::default()
                }
            ),
            Err(KapError::Lap(crate::LapError::Infeasible))
        );
    }
    let data = [1i64];
    let mats = [MatrixView::from_row_major(&data, 1, 1).unwrap()];
    for tau in [0.5, -(i64::MIN as f64), f64::NAN, f64::INFINITY] {
        assert_eq!(
            k_assignment(&mats, Algo::Am, &options(tau, Sense::Minimize)),
            Err(KapError::BadOption("threshold"))
        );
    }
    let unsupported = Err(KapError::Unsupported(
        "threshold is not yet supported with this algorithm or option",
    ));
    for algo in [Algo::Dv1, Algo::DvS, Algo::Ils, Algo::Memetic] {
        assert_eq!(
            k_assignment(&mats, algo, &options(0.0, Sense::Minimize)),
            unsupported
        );
    }
    for algo in ALGOS {
        for refine in [Dv::One, Dv::Subsets] {
            let opts = Options {
                refine: Some(refine),
                ..options(0.0, Sense::Minimize)
            };
            assert_eq!(k_assignment(&mats, algo, &opts), unsupported);
        }
        assert_eq!(
            k_assignment_from(&mats, &[], algo, &options(0.0, Sense::Minimize)),
            unsupported
        );
    }
    assert_eq!(
        KapError::BadOption("threshold").to_string(),
        "threshold must be finite, and an integer for integer costs"
    );
}

#[test]
fn gated_flag_survives_node_rebuild_and_bipartition() {
    use crate::kpartite::NONE;
    let data = [2i64, 3, 4, 5];
    let mats = [MatrixView::from_row_major(&data, 2, 2).unwrap()];
    let original = KPartite::new(&mats).unwrap();
    let gated = original.gated(1.0, Sense::Minimize).unwrap();
    let root = Quotient::root(&gated);
    assert_eq!(root.matrix(0, 1).get(0, 0), 1);
    assert_eq!(Quotient::root(&original).matrix(0, 1).get(0, 0), 2);
    let nodes = vec![0, NONE, 1, NONE, NONE, 0, NONE, 1];
    for rebuilt in [
        Quotient::from_nodes(&root, nodes.clone()).unwrap(),
        Quotient::from_nodes_checked(&root, nodes).unwrap(),
    ] {
        let (split, matching) = rebuilt.bipartition(&root, 1).unwrap();
        assert!(matching.is_empty());
        let (matching, cost) = split
            .solve_pair(0, 1, &mut LapSolver::new(), Sense::Minimize)
            .unwrap();
        assert!(matching.is_empty());
        assert_eq!(cost, 0);
    }
}
