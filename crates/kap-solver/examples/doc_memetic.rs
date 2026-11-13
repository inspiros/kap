use kap::{Algo, MatrixView, Options, k_assignment};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c01 = [12, 10, 22, 12, 23, 17, 4, 15, 12i64];
    let c02 = [17, 17, 17, 17, 26, 10, 3, 20, 22i64];
    let c12 = [14, 24, 9, 21, 20, 12, 23, 28, 8i64];
    let costs = [
        MatrixView::from_row_major(&c01, 3, 3)?,
        MatrixView::from_row_major(&c02, 3, 3)?,
        MatrixView::from_row_major(&c12, 3, 3)?,
    ];
    let opts = Options {
        max_iter: 4,
        population: Some(4),
        seed: 0,
        ..Options::default()
    };
    let result = k_assignment(&costs, Algo::Memetic, &opts)?;
    println!("matches = {:?}", result.cliques);
    println!("matching_costs = {:?}", result.costs);
    println!("free = {:?}", result.free);
    Ok(())
}
