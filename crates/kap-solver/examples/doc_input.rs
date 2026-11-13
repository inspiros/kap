use kap::{Algo, MatrixView, Options, Sense, k_assignment};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c01 = [9, 2, 7, 6, 4, 3, 5, 8, 1i64];
    let c02 = [3, 8, 7, 2, 6, 4i64];
    let c12 = [4, 9, 1, 5, 8, 2i64];
    let costs = [
        MatrixView::from_row_major(&c01, 3, 3)?,
        MatrixView::from_row_major(&c02, 3, 2)?,
        MatrixView::from_row_major(&c12, 3, 2)?,
    ];
    println!(
        "shapes = {:?}",
        costs
            .iter()
            .map(|m| (m.rows(), m.cols()))
            .collect::<Vec<_>>()
    );
    let result = k_assignment(&costs, Algo::Am, &Options::default())?;
    println!("matches = {:?}", result.cliques);
    println!("matching_costs = {:?}", result.costs);
    println!("free = {:?}", result.free);
    let opts = Options {
        sense: Sense::Maximize,
        ..Options::default()
    };
    let result = k_assignment(&costs, Algo::Am, &opts)?;
    println!("maximize:");
    println!("matches = {:?}", result.cliques);
    println!("matching_costs = {:?}", result.costs);
    println!("free = {:?}", result.free);
    Ok(())
}
