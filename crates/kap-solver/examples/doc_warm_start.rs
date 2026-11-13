use kap::{Algo, MatrixView, Options, k_assignment_from};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c01 = [28, 25, 18, 1, 12, 7, 3, 4, 14i64];
    let c02 = [21, 18, 10, 12, 17, 24, 2, 17, 9i64];
    let c12 = [8, 28, 25, 27, 8, 11, 2, 20, 3i64];
    let costs = [
        MatrixView::from_row_major(&c01, 3, 3)?,
        MatrixView::from_row_major(&c02, 3, 3)?,
        MatrixView::from_row_major(&c12, 3, 3)?,
    ];
    let initial: Vec<_> = (0..3).map(|i| vec![(0, i), (1, i), (2, i)]).collect();
    let result = k_assignment_from(&costs, &initial, Algo::Dv1, &Options::default())?;
    println!("matches = {:?}", result.cliques);
    println!("matching_costs = {:?}", result.costs);
    println!("free = {:?}", result.free);
    Ok(())
}
