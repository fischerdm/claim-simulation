use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Poisson};
use rayon::prelude::*;

use crate::model::FrequencyModel;
use crate::portfolio::Policy;

// One ONNX session per Rayon worker thread, initialized lazily on first use.
// This avoids paying N_THREADS × load_time upfront regardless of how many
// threads actually receive work (critical on large machines with many cores).
thread_local! {
    static THREAD_MODEL: RefCell<Option<FrequencyModel>> = RefCell::new(None);
}

/// Per-year aggregate result for one simulation run.
pub struct YearResult {
    pub total_claims: u64,
    pub frequency:    f64, // total_claims / n_policies
}

/// Run one complete multi-year simulation for a single seed.
///
/// The simulation loop for years t = 0 … n_years-1:
///   1. Build the feature matrix using the current VehAge, DrivAge, PriorClaims3Y.
///   2. Call ONNX once (batch over all policies) → λ per policy.
///   3. Draw Poisson(λ × exposure) per policy.
///   4. Shift the rolling 3-year claim window: drop oldest, append new claims.
///   5. Increment VehAge and DrivAge by 1.
///
/// Exposure at t=0 uses the value from the portfolio (fraction of year active).
/// For t=1..n_years-1 exposure = 1.0 (full renewal years).
fn run_one(
    model:   &mut FrequencyModel,
    policies: &[Policy],
    seed:     u64,
    n_years:  usize,
) -> Vec<YearResult> {
    let n = policies.len();
    let mut rng = SmallRng::seed_from_u64(seed);

    // Per-policy mutable state — cloned from the portfolio at the start of each sim.
    let mut veh_age:      Vec<f32>      = policies.iter().map(|p| p.veh_age).collect();
    let mut driv_age:     Vec<f32>      = policies.iter().map(|p| p.driv_age).collect();
    // Rolling 3-year window: [oldest (t-3), t-2, newest (t-1)]
    let mut claim_window: Vec<[u32; 3]> = policies.iter().map(|p| p.claims_hist).collect();

    let mut results = Vec::with_capacity(n_years);

    for year in 0..n_years {
        // Build the [n × 9] feature matrix for this projection year.
        let flat: Vec<f32> = (0..n)
            .flat_map(|i| {
                let prior_3y = (claim_window[i][0]
                    + claim_window[i][1]
                    + claim_window[i][2]) as f32;
                policies[i].to_feature_row(veh_age[i], driv_age[i], prior_3y)
            })
            .collect();

        // One ONNX call batching all policies → λ per policy (annual frequency).
        let lambdas = model
            .run_inference(flat, n, 9)
            .expect("ONNX inference failed");

        // Draw Poisson(λ × exposure) for each policy.
        // At year 0 we honour the original exposure (partial year).
        // From year 1 onward every policy is a full renewal year (exposure = 1.0).
        let mut total_claims: u64 = 0;
        for (i, &lambda) in lambdas.iter().enumerate() {
            let exposure = if year == 0 { policies[i].exposure as f64 } else { 1.0 };
            let mu       = (lambda * exposure).max(1e-9);
            let claims   = Poisson::new(mu).expect("mu must be positive").sample(&mut rng) as u32;
            total_claims += claims as u64;

            // Advance the rolling window: shift left, insert the new draw at position 2.
            claim_window[i] = [claim_window[i][1], claim_window[i][2], claims];
        }

        results.push(YearResult {
            total_claims,
            frequency: total_claims as f64 / n as f64,
        });

        // Age vehicles and drivers for the next projection year.
        for i in 0..n {
            veh_age[i]  += 1.0;
            driv_age[i] += 1.0;
        }
    }

    results
}

/// Run `n_sims` multi-year simulations in parallel using Rayon.
///
/// Each simulation is independent: from year 1 onward every sim has its own
/// distinct PriorClaims3Y driven by its own Poisson draws, so ONNX must be
/// called separately per simulation per year.
///
/// Session management: ONNX sessions are held in thread-local storage
/// (see `THREAD_MODEL`) and initialised lazily on the first simulation each
/// worker thread receives.  This has two advantages over pre-allocating
/// N_THREADS sessions upfront:
///
/// 1. **No blocking startup cost.**  Pre-allocation is sequential and pays
///    load_time × N_THREADS before a single simulation runs (~30 s/session on
///    this machine → 48 min wasted startup on a 96-core box).  With lazy init
///    sessions load in parallel as Rayon dispatches the first wave of work;
///    wall-clock overhead ≈ one session load time regardless of core count.
///
/// 2. **Cores are not over-provisioned.**  For small grids (e.g. QUICK_TEST)
///    only the threads that receive work ever load a session.
pub fn run_parallel(
    model_path: &Path,
    policies:   &[Policy],
    n_sims:     usize,
    n_years:    usize,
) -> Vec<Vec<YearResult>> {
    // Share the path with worker threads without copying the string N times.
    let model_path: Arc<PathBuf> = Arc::new(model_path.to_path_buf());

    (0..n_sims)
        .into_par_iter()
        .map(|sim_i| {
            let path = Arc::clone(&model_path);
            THREAD_MODEL.with(|cell| {
                let mut opt = cell.borrow_mut();
                // Initialize this thread's session on first use.
                if opt.is_none() {
                    *opt = Some(
                        FrequencyModel::load(&path)
                            .expect("failed to load ONNX model for thread"),
                    );
                }
                run_one(opt.as_mut().unwrap(), policies, sim_i as u64, n_years)
            })
        })
        .collect()
}

/// Print per-year summary statistics across all simulations.
pub fn print_stats(results: &[Vec<YearResult>], n_policies: usize) {
    let n_years = results[0].len();

    println!();
    println!(
        "=== Multi-Year Claim Simulation ({} sims × {} years × {} policies) ===",
        results.len(),
        n_years,
        n_policies,
    );
    println!();
    println!(
        "{:<6}  {:>12}  {:>10}  {:>10}  {:>10}  {:>10}",
        "Year", "Mean claims", "Mean freq", "P50 freq", "P95 freq", "P99 freq"
    );
    println!("{}", "-".repeat(62));

    for year in 0..n_years {
        let mut freqs: Vec<f64> = results.iter().map(|r| r[year].frequency).collect();
        let mean_claims =
            results.iter().map(|r| r[year].total_claims as f64).sum::<f64>() / results.len() as f64;
        let mean_freq = freqs.iter().sum::<f64>() / freqs.len() as f64;

        freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pct = |p: f64| -> f64 {
            let idx = p * (freqs.len() - 1) as f64;
            let lo  = idx.floor() as usize;
            let hi  = idx.ceil()  as usize;
            let frc = idx - lo as f64;
            freqs[lo] * (1.0 - frc) + freqs[hi] * frc
        };

        println!(
            "t={:<4}  {:>12.1}  {:>10.5}  {:>10.5}  {:>10.5}  {:>10.5}",
            year,
            mean_claims,
            mean_freq,
            pct(0.50),
            pct(0.95),
            pct(0.99),
        );
    }
    println!();
}
