use kap::{MatrixView, Sense, linear_sum_assignment};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = [8i64, 2, 7, 6, 6, 9, 3, 5, 4, 7, 8, 1];
    let matrix = MatrixView::from_row_major(&data, 3, 4)?;
    let result = linear_sum_assignment(matrix, Sense::Minimize)?;
    println!("rows = {:?}", result.row_ind);
    println!("columns = {:?}", result.col_ind);
    println!("cost = {}", result.cost);
    Ok(())
}
