use kap::{Algo, MatrixView, Options, k_assignment};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c01 = [18, 8, 29, 28, 2, 6, 6, 6, 17i64];
    let c02 = [11, 14, 7, 28, 20, 20, 4, 5, 26i64];
    let c12 = [10, 25, 23, 1, 14, 16, 14, 4, 21i64];
    let costs = [
        MatrixView::from_row_major(&c01, 3, 3)?,
        MatrixView::from_row_major(&c02, 3, 3)?,
        MatrixView::from_row_major(&c12, 3, 3)?,
    ];
    let opts = Options {
        max_iter: 20,
        n_trials: 2,
        seed: 0,
        ..Options::default()
    };
    let result = k_assignment(&costs, Algo::Ils, &opts)?;
    println!("matches = {:?}", result.cliques);
    println!("matching_costs = {:?}", result.costs);
    println!("free = {:?}", result.free);
    Ok(())
}
