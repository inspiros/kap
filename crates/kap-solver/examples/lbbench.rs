//! Deterministic integer benchmark: cargo run -r -p kap-solver --example lbbench.
use kap::{Algo, LowerBoundOptions, MatrixView, Options, k_assignment, lower_bound};
use std::time::Instant;

fn next(seed: &mut u64) -> i64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed as u128 * 100) >> 64) as i64
}

fn main() -> Result<(), kap::KapError> {
    println!("k,n,bound,ils_upper,gap_percent,lower_bound_ms");
    for (k, n) in [(3, 10), (3, 20), (3, 50), (4, 10), (5, 10)] {
        let mut seed = 1;
        let data: Vec<Vec<i64>> = (0..k * (k - 1) / 2)
            .map(|_| (0..n * n).map(|_| next(&mut seed)).collect())
            .collect();
        let mats: Vec<_> = data
            .iter()
            .map(|d| MatrixView::from_row_major(d, n, n).unwrap())
            .collect();
        let start = Instant::now();
        let bound = lower_bound(&mats, &LowerBoundOptions::default())?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let upper = k_assignment(&mats, Algo::Ils, &Options::default())?.cost;
        let gap = 100.0 * (upper - bound) as f64 / upper as f64;
        println!("{k},{n},{bound},{upper},{gap:.2},{elapsed:.3}");
    }
    Ok(())
}
