mod model;
mod portfolio;
mod simulator;
mod simulator_multiyear;

use std::path::Path;
use std::time::Instant;

use model::FrequencyModel;
use portfolio::{load_from_csv, load_multiyear_from_csv};
use simulator::{print_stats, run_parallel};
use simulator_multiyear::{print_stats as print_stats_multiyear, run_parallel as run_parallel_multiyear};

// --- v1: single-year simulation (original) ---
const N_SIMS: usize = 10_000;

// Paths are relative to the rust/ directory where `cargo run` is executed.
const MODEL_PATH:     &str = "../models/frequency_model.onnx";
const PORTFOLIO_PATH: &str = "../data/portfolio.csv";

// --- v2: multi-year simulation with claim history feedback ---
const N_SIMS_V2:         usize = 10_000;
const MODEL_PATH_V2:     &str  = "../models/frequency_model_v2.onnx";
const PORTFOLIO_PATH_V2: &str  = "../data/portfolio_v2.csv";

/// Entry point.
///
/// Runs two simulations back-to-back:
///
///   v1 — Single-year: λ computed ONCE, all simulations share the same rates.
///         Parallelism over simulations; ONNX called exactly once.
///
///   v2 — Multi-year (5 years): λ recomputed each year per simulation because
///         PriorClaims3Y (the rolling 3-year claim window) changes as claims
///         are drawn. Each Rayon worker thread owns one ONNX session; there is
///         no lock contention between threads.
fn main() -> anyhow::Result<()> {
    // ── v1: single-year ──────────────────────────────────────────────────────
    println!("=== v1: single-year simulation ===");
    println!("Loading ONNX model from {} ...", MODEL_PATH);
    let mut model = FrequencyModel::load(Path::new(MODEL_PATH))?;

    println!("Loading portfolio from {} ...", PORTFOLIO_PATH);
    let policies       = load_from_csv(Path::new(PORTFOLIO_PATH))?;
    let total_exposure = policies.iter().map(|p| p.exposure as f64).sum::<f64>();
    println!(
        "Portfolio: {} policies, {:.2} total policy-years",
        policies.len(),
        total_exposure
    );

    println!("Computing λ per policy ...");
    let lambdas = model.compute_lambdas(&policies)?;

    println!("Running {} simulations in parallel ...", N_SIMS);
    let t0      = Instant::now();
    let results = run_parallel(&lambdas, total_exposure, N_SIMS);
    let elapsed = t0.elapsed();

    println!(
        "Done in {:.2?}  ({:.1} µs/simulation)",
        elapsed,
        elapsed.as_micros() as f64 / N_SIMS as f64
    );
    print_stats(&results, total_exposure);

    // ── v2: multi-year ───────────────────────────────────────────────────────
    println!("=== v2: multi-year simulation ({} years) ===", simulator_multiyear::N_YEARS);
    println!("Loading multi-year portfolio from {} ...", PORTFOLIO_PATH_V2);
    let policies_v2 = load_multiyear_from_csv(Path::new(PORTFOLIO_PATH_V2))?;
    println!("Portfolio: {} policies", policies_v2.len());

    println!("Running {} multi-year simulations in parallel ...", N_SIMS_V2);
    let t0_v2  = Instant::now();
    let res_v2 = run_parallel_multiyear(Path::new(MODEL_PATH_V2), &policies_v2, N_SIMS_V2);
    let elapsed_v2 = t0_v2.elapsed();

    println!(
        "Done in {:.2?}  ({:.1} ms/simulation)",
        elapsed_v2,
        elapsed_v2.as_millis() as f64 / N_SIMS_V2 as f64
    );
    print_stats_multiyear(&res_v2, policies_v2.len());

    Ok(())
}
