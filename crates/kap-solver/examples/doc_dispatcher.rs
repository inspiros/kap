use kap::{Algo, Dv, MatrixView, Options, Sense, k_assignment};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c01 = [6, 10, 12, 1, 29, 2, 28, 29, 28i64];
    let c02 = [26, 29, 16, 7, 29, 26, 29, 2, 12i64];
    let c12 = [15, 10, 28, 20, 17, 1, 13, 17, 13i64];
    let costs = [
        MatrixView::from_row_major(&c01, 3, 3)?,
        MatrixView::from_row_major(&c02, 3, 3)?,
        MatrixView::from_row_major(&c12, 3, 3)?,
    ];
    let opts = Options {
        sense: Sense::Maximize,
        refine: Some(Dv::One),
        ..Options::default()
    };
    let result = k_assignment(&costs, Algo::Shift, &opts)?;
    println!("cliques = {:?}", result.cliques);
    println!("costs = {:?}", result.costs);
    println!("free = {:?}", result.free);
    Ok(())
}
