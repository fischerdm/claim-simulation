mod model;
mod portfolio;
mod simulator_multiyear;

use std::path::Path;
use std::time::Instant;

use portfolio::load_from_csv;
use simulator_multiyear::{
    print_stats as print_stats_multiyear, run_parallel as run_parallel_multiyear,
};

const N_SIMS:         usize = 10_000;
const MODEL_PATH:     &str  = "../models/frequency_model_v2.onnx";
const PORTFOLIO_PATH: &str  = "../data/portfolio_v2.csv";

fn main() -> anyhow::Result<()> {
    println!(
        "=== Multi-year simulation ({} years) ===",
        simulator_multiyear::N_YEARS,
    );
    println!("Loading portfolio from {} ...", PORTFOLIO_PATH);
    let policies = load_from_csv(Path::new(PORTFOLIO_PATH))?;
    println!("Portfolio: {} policies", policies.len());

    println!("Running {} simulations in parallel ...", N_SIMS);
    let t0      = Instant::now();
    let results = run_parallel_multiyear(Path::new(MODEL_PATH), &policies, N_SIMS);
    let elapsed = t0.elapsed();

    println!(
        "Done in {:.3?}  ({:.1} ms/simulation)",
        elapsed,
        elapsed.as_millis() as f64 / N_SIMS as f64,
    );
    print_stats_multiyear(&results, policies.len());

    Ok(())
}
