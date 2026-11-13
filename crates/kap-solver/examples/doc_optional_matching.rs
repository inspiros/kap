use kap::{Algo, MatrixView, Options, k_assignment};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = [1.0, f64::INFINITY, 9.0, 2.0, 8.0, 7.0];
    let costs = [MatrixView::from_row_major(&data, 3, 2)?];
    let opts = Options {
        threshold: Some(5.0),
        max_iter: 10,
        seed: 0,
        ..Options::default()
    };
    let result = k_assignment(&costs, Algo::Ils, &opts)?;
    println!("matches = {:?}", result.cliques);
    println!("matching_costs = {:?}", result.costs);
    println!("free = {:?}", result.free);
    Ok(())
}
