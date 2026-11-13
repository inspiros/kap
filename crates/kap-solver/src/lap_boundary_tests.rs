//! Integer LAP regression tests with an independent i128 objective oracle.
use super::*;
use proptest::prelude::*;

fn optimum(c: &[i64], nr: usize, nc: usize, sense: Sense) -> i128 {
    fn visit(c: &[i64], nc: usize, row: usize, nr: usize, used: &mut [bool], sense: Sense) -> i128 {
        if row == nr {
            return 0;
        }
        let mut best = None;
        for col in 0..nc {
            if used[col] {
                continue;
            }
            used[col] = true;
            let value = i128::from(c[row * nc + col]) + visit(c, nc, row + 1, nr, used, sense);
            used[col] = false;
            best = Some(match (best, sense) {
                (None, _) => value,
                (Some(old), Sense::Minimize) => value.min(old),
                (Some(old), Sense::Maximize) => value.max(old),
            });
        }
        best.unwrap()
    }
    if nr > nc {
        let t: Vec<_> = (0..nc)
            .flat_map(|j| (0..nr).map(move |i| c[i * nc + j]))
            .collect();
        return optimum(&t, nc, nr, sense);
    }
    visit(c, nc, 0, nr, &mut vec![false; nc], sense)
}

fn verify(
    solver: &LapSolver<i64>,
    sol: &LapSolution<i64>,
    c: &[i64],
    nr: usize,
    nc: usize,
    sense: Sense,
) {
    assert_eq!(sol.row_ind.len(), nr.min(nc));
    assert!(sol.row_ind.windows(2).all(|x| x[0] < x[1]));
    let mut columns = sol.col_ind.clone();
    columns.sort_unstable();
    columns.dedup();
    assert_eq!(columns.len(), sol.col_ind.len());
    let actual: i128 = sol
        .row_ind
        .iter()
        .zip(&sol.col_ind)
        .map(|(&i, &j)| i128::from(c[i * nc + j]))
        .sum();
    assert_eq!(i128::from(sol.cost), actual);
    assert_eq!(actual, optimum(c, nr, nc, sense));
    let d = solver.duals();
    for i in 0..nr {
        for j in 0..nc {
            let slack = i128::from(c[i * nc + j]) - i128::from(d.row[i]) - i128::from(d.col[j]);
            assert!(if sense == Sense::Minimize {
                slack >= 0
            } else {
                slack <= 0
            });
        }
    }
    for (&i, &j) in sol.row_ind.iter().zip(&sol.col_ind) {
        assert_eq!(
            i128::from(c[i * nc + j]),
            i128::from(d.row[i]) + i128::from(d.col[j])
        );
    }
    if nr < nc {
        for j in 0..nc {
            if !sol.col_ind.contains(&j) {
                assert_eq!(d.col[j], 0);
            }
        }
    } else if nr > nc {
        for i in 0..nr {
            if !sol.row_ind.contains(&i) {
                assert_eq!(d.row[i], 0);
            }
        }
    }
}

#[test]
fn jv_overflow_restarts_from_clean_sap_state() {
    let m = i64::MAX / 4 - 1;
    let c = [-m, m, m, -m];
    for mode in [
        JvMode::Fused,
        JvMode::Lapjv,
        JvMode::InitThenSap,
        JvMode::Off,
    ] {
        let mut solver = LapSolver::new().with_jv_mode(mode);
        for sense in [Sense::Minimize, Sense::Maximize] {
            let sol = solver
                .solve(MatrixView::from_row_major(&c, 2, 2).unwrap(), sense)
                .unwrap();
            assert_eq!(
                sol.cost,
                if sense == Sense::Minimize {
                    -2 * m
                } else {
                    2 * m
                }
            );
            verify(&solver, &sol, &c, 2, 2, sense);
        }
    }
}

#[test]
fn unrepresentable_total_is_an_error_and_solver_can_be_reused() {
    let m = i64::MAX / 4 - 1;
    for sign in [-1, 1] {
        let c = vec![sign * m; 25];
        for mode in [
            JvMode::Fused,
            JvMode::Lapjv,
            JvMode::InitThenSap,
            JvMode::Off,
        ] {
            let mut solver = LapSolver::new().with_jv_mode(mode);
            for sense in [Sense::Minimize, Sense::Maximize] {
                assert_eq!(
                    solver.solve(MatrixView::from_row_major(&c, 5, 5).unwrap(), sense),
                    Err(LapError::Overflow)
                );
                let small = [4, 1, 2, 3, 0, 5];
                let sol = solver
                    .solve(MatrixView::from_row_major(&small, 2, 3).unwrap(), sense)
                    .unwrap();
                verify(&solver, &sol, &small, 2, 3, sense);
            }
        }
    }
}

#[test]
fn most_negative_integer_is_rejected_before_maximization_negation() {
    for shape in [(1, 2), (2, 1)] {
        let c = [i64::MIN, 0];
        assert!(matches!(
            LapSolver::new().solve(
                MatrixView::from_row_major(&c, shape.0, shape.1).unwrap(),
                Sense::Maximize
            ),
            Err(LapError::InvalidCost(0, 0))
        ));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn large_integer_solutions_match_i128_oracle(
        (nr,nc,small) in (1usize..=5,1usize..=5).prop_flat_map(|(nr,nc)| (Just(nr),Just(nc),prop::collection::vec(-2i64..=2,nr*nc))),
        maximize in any::<bool>(),
    ) {
        let scale=i64::MAX / (64 * nr.min(nc) as i64);
        let c:Vec<_>=small.into_iter().map(|v|v*scale).collect();
        let sense=if maximize {Sense::Maximize} else {Sense::Minimize};
        for mode in [JvMode::Fused,JvMode::Lapjv,JvMode::InitThenSap,JvMode::Off] {
            let mut solver=LapSolver::new().with_jv_mode(mode);
            let sol=solver.solve(MatrixView::from_row_major(&c,nr,nc).unwrap(),sense).unwrap();
            verify(&solver,&sol,&c,nr,nc,sense);
        }
    }

    #[test]
    fn boundary_entries_never_wrap_or_panic(
        (nr,nc,small) in (1usize..=5,1usize..=5).prop_flat_map(|(nr,nc)| (Just(nr),Just(nc),prop::collection::vec((-1i64..=1, 0i64..=16),nr*nc))),
        maximize in any::<bool>(),
    ) {
        let m=i64::MAX/4-1;
        let c:Vec<_>=small.into_iter().map(|(sign,delta)| sign * (m-delta)).collect();
        let sense=if maximize {Sense::Maximize} else {Sense::Minimize};
        for mode in [JvMode::Fused,JvMode::Lapjv,JvMode::InitThenSap,JvMode::Off] {
            let mut solver=LapSolver::new().with_jv_mode(mode);
            match solver.solve(MatrixView::from_row_major(&c,nr,nc).unwrap(),sense) {
                Ok(sol)=>verify(&solver,&sol,&c,nr,nc,sense),
                Err(error)=>prop_assert_eq!(error,LapError::Overflow),
            }
        }
    }
}

#[test]
fn representable_extreme_total_costs_remain_exact() {
    let m = i64::MAX / 4 - 1;
    for (values, expected) in [
        ([m, m, m, m, 7], i64::MAX),
        ([-m, -m, -m, -m, -8], i64::MIN),
    ] {
        let c: Vec<_> = values.into_iter().flat_map(|x| [x; 5]).collect();
        let mut solver = LapSolver::new();
        let sol = solver
            .solve(
                MatrixView::from_row_major(&c, 5, 5).unwrap(),
                Sense::Minimize,
            )
            .unwrap();
        assert_eq!(sol.cost, expected);
        verify(&solver, &sol, &c, 5, 5, Sense::Minimize);
    }
}
