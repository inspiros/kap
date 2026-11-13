use kap::{Algo, Dv, MatrixView, Options, k_assignment};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c01 = [12, 6, 13, 12, 13, 26, 28, 12, 27i64];
    let c02 = [20, 14, 15, 27, 16, 14, 26, 7, 8i64];
    let c12 = [3, 5, 22, 10, 5, 12, 1, 20, 26i64];
    let costs = [
        MatrixView::from_row_major(&c01, 3, 3)?,
        MatrixView::from_row_major(&c02, 3, 3)?,
        MatrixView::from_row_major(&c12, 3, 3)?,
    ];
    let opts = Options {
        max_iter: 10,
        refine: Some(Dv::Subsets),
        ..Options::default()
    };
    let result = k_assignment(&costs, Algo::Dm, &opts)?;
    println!("cliques = {:?}", result.cliques);
    println!("costs = {:?}", result.costs);
    println!("free = {:?}", result.free);
    Ok(())
}
