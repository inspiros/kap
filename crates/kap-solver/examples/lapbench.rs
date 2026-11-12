//! Quick timing of the LAP solver paths: `cargo run -r --example lapbench -- [n...]`.
use kap::lap::JvMode;
use kap::{LapSolver, MatrixView, Sense};
use std::time::Instant;

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 11) as f64 / (1u64 << 53) as f64
}

fn time(c: &[f64], r: usize, cols: usize, mode: JvMode) -> (f64, f64) {
    let m = MatrixView::from_row_major(c, r, cols).unwrap();
    let mut s = LapSolver::new().with_jv_mode(mode);
    let mut best = f64::INFINITY;
    let mut cost = 0.0;
    let t0 = Instant::now();
    let mut reps = 0;
    while reps < 3 || (t0.elapsed().as_secs_f64() < 0.5 && reps < 10000) {
        let t = Instant::now();
        cost = s.solve(m, Sense::Minimize).unwrap().cost;
        best = best.min(t.elapsed().as_secs_f64());
        reps += 1;
    }
    (best * 1e6, cost)
}

fn main() {
    let ns: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let ns = if ns.is_empty() {
        vec![10, 100, 500, 1000]
    } else {
        ns
    };
    let mut seed = 1u64;
    for n in ns {
        for (name, rect) in [
            ("uniform", false),
            ("uniform", true),
            ("int100", false),
            ("int100", true),
            ("geometric", false),
            ("geometric", true),
        ] {
            let cols = if rect { 2 * n } else { n };
            let c: Vec<f64> = (0..n * cols)
                .map(|k| {
                    let x = lcg(&mut seed);
                    match name {
                        "geometric" => ((k / cols) * (k % cols)) as f64 + x,
                        "int100" => (x * 100.0).floor(),
                        _ => x,
                    }
                })
                .collect();
            let (tf, cf) = time(&c, n, cols, JvMode::Fused);
            let (tj, cj) = time(&c, n, cols, JvMode::Lapjv);
            let (th, ch) = time(&c, n, cols, JvMode::InitThenSap);
            let (ts, cs) = time(&c, n, cols, JvMode::Off);
            assert!((cf - cs).abs() <= 1e-6 * (1.0 + cs.abs()));
            assert!((cj - cs).abs() <= 1e-6 * (1.0 + cs.abs()));
            assert!((ch - cs).abs() <= 1e-6 * (1.0 + cs.abs()));
            println!(
                "{name:10} {n:5}x{cols:<5} fused {tf:11.1}  lapjv {tj:11.1}  init+sap {th:11.1}  sap {ts:11.1} us"
            );
        }
    }
}
