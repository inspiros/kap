use kap::{Algo, LowerBoundOptions, MatrixView, Options, k_assignment, lower_bound};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c01 = [19, 22, 6, 27, 16, 22, 16, 26, 10i64];
    let c02 = [7, 10, 22, 15, 19, 26, 9, 27, 9i64];
    let c12 = [6, 25, 5, 7, 3, 12, 19, 13, 2i64];
    let costs = [
        MatrixView::from_row_major(&c01, 3, 3)?,
        MatrixView::from_row_major(&c02, 3, 3)?,
        MatrixView::from_row_major(&c12, 3, 3)?,
    ];
    let result = k_assignment(&costs, Algo::Dv1, &Options::default())?;
    let opts = LowerBoundOptions {
        max_iter: 30,
        ..LowerBoundOptions::default()
    };
    let bound = lower_bound(&costs, &opts)?;
    println!("cost = {}", result.cost);
    println!("bound = {bound}");
    println!(
        "gap = {:.3}",
        (result.cost - bound) as f64 / result.cost as f64
    );
    Ok(())
}
