use kap::{Algo, MatrixView, Options, k_assignment};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c01 = [4, 12, 15, 18, 20, 15, 2, 21, 28i64];
    let c02 = [19, 25, 26, 18, 18, 4, 16, 23, 23i64];
    let c12 = [24, 7, 24, 4, 11, 3, 11, 14, 2i64];
    let costs = [
        MatrixView::from_row_major(&c01, 3, 3)?,
        MatrixView::from_row_major(&c02, 3, 3)?,
        MatrixView::from_row_major(&c12, 3, 3)?,
    ];
    let result = k_assignment(&costs, Algo::Bm, &Options::default())?;
    println!("cliques = {:?}", result.cliques);
    println!("costs = {:?}", result.costs);
    println!("free = {:?}", result.free);
    Ok(())
}
