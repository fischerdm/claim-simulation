mod model;
mod portfolio;
mod simulator;

use std::path::Path;
use std::time::Instant;

use model::FrequencyModel;
use portfolio::load_from_csv;
use simulator::{print_stats, run_parallel};

const N_SIMS: usize = 10_000;

// Paths are relative to the rust/ directory where `cargo run` is executed.
const MODEL_PATH:     &str = "../models/frequency_model.onnx";
const PORTFOLIO_PATH: &str = "../data/portfolio.csv";

/// Entry point.
///
/// In Rust, `main` can return `Result` — errors are printed and the process
/// exits with a non-zero code. The `?` operator propagates errors upward,
/// equivalent to re-raising an exception in Python.
fn main() -> anyhow::Result<()> {
    println!("Loading ONNX model from {} ...", MODEL_PATH);
    let mut model = FrequencyModel::load(Path::new(MODEL_PATH))?;

    println!("Loading portfolio from {} ...", PORTFOLIO_PATH);
    let policies = load_from_csv(Path::new(PORTFOLIO_PATH))?;
    let total_exposure: f64 = policies.iter().map(|p| p.exposure as f64).sum();

    println!(
        "Portfolio: {} policies, {:.2} total policy-years",
        policies.len(),
        total_exposure
    );

    // Run ONNX inference ONCE to get λ per policy.
    // The model is deterministic: same policy → same λ every time.
    // All 10,000 simulations share these lambdas and differ only in their
    // Poisson random draws.
    println!("Computing claim rates (λ per policy) ...");
    let lambdas = model.compute_lambdas(&policies)?;

    println!("Running {} simulations in parallel ...", N_SIMS);
    let t0 = Instant::now();
    let results = run_parallel(&lambdas, total_exposure, N_SIMS);
    let elapsed = t0.elapsed();

    println!(
        "Done in {:.2?}  ({:.1} µs/simulation)",
        elapsed,
        elapsed.as_micros() as f64 / N_SIMS as f64
    );

    print_stats(&results, total_exposure);

    Ok(())
}
