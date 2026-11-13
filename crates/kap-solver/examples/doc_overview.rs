use kap::{Algo, MatrixView, Options, k_assignment};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = [4i64, 1, 2, 3];
    let matrix = MatrixView::from_row_major(&data, 2, 2)?;
    let costs = [matrix, matrix, matrix];
    let result = k_assignment(&costs, Algo::Ils, &Options::default())?;
    println!("{:?} {:?}", result.cliques, result.costs);
    Ok(())
}
