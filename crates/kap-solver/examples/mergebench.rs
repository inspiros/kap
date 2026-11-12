//! Time quotient merges against the LAP they follow:
//! `cargo run -r --example mergebench -- [k n ...]` (pairs of k and n).
use kap::{KPartite, LapSolver, MatrixView, Quotient, Sense};
use std::time::Instant;

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 11) as f64 / (1u64 << 53) as f64
}

fn best_us(mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    let t0 = Instant::now();
    let mut reps = 0;
    while reps < 3 || (t0.elapsed().as_secs_f64() < 0.3 && reps < 100_000) {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
        reps += 1;
    }
    best * 1e6
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let cases: Vec<(usize, usize)> = if args.is_empty() {
        vec![(4, 100), (6, 100), (6, 500), (8, 200)]
    } else {
        args.chunks_exact(2).map(|c| (c[0], c[1])).collect()
    };
    println!("k,n,pair,merge_us,lap_us,merge_share");
    for (k, n) in cases {
        let mut seed = 1;
        let data: Vec<Vec<f64>> = (0..k * (k - 1) / 2)
            .map(|_| (0..n * n).map(|_| lcg(&mut seed)).collect())
            .collect();
        let views: Vec<_> = data
            .iter()
            .map(|d| MatrixView::from_row_major(d, n, n).unwrap())
            .collect();
        let problem = KPartite::<f64>::new(&views).unwrap();
        let root = Quotient::root(&problem);
        let mut solver = LapSolver::new();
        // first pair (0, 1) and a middle pair, which exercises transposed reads
        for (a, b) in [(0, 1), (1, k - 1)] {
            let sol = solver.solve(root.matrix(a, b), Sense::Minimize).unwrap();
            let matching: Vec<(u32, u32)> = sol
                .row_ind
                .iter()
                .zip(&sol.col_ind)
                .map(|(&r, &c)| (r as u32, c as u32))
                .collect();
            let merge = best_us(|| {
                std::hint::black_box(root.fix(a, b, &matching).unwrap());
            });
            let lap = best_us(|| {
                std::hint::black_box(solver.solve(root.matrix(a, b), Sense::Minimize).unwrap());
            });
            println!(
                "{k},{n},{a}-{b},{merge:.1},{lap:.1},{:.2}",
                merge / (merge + lap)
            );
        }
    }
}
