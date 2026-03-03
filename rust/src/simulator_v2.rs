use std::path::Path;
use std::sync::Mutex;

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Poisson};
use rayon::prelude::*;

use crate::model::FrequencyModel;
use crate::portfolio_v2::PolicyV2;

/// Number of projection years.
pub const N_YEARS: usize = 5;

/// Per-year aggregate result for one simulation run.
pub struct YearResult {
    pub total_claims: u64,
    pub frequency:    f64, // total_claims / n_policies (exposure = 1.0 per year)
}

/// Run one complete multi-year simulation for a single seed.
///
/// The simulation loop for years t = 0 … N_YEARS-1:
///   1. Build the feature matrix using the current VehAge, DrivAge, PriorClaims3Y.
///   2. Call ONNX once (batch over all policies) → λ per policy.
///   3. Draw Poisson(λ × exposure) per policy.
///   4. Shift the rolling 3-year claim window: drop oldest, append new claims.
///   5. Increment VehAge and DrivAge by 1.
///
/// Exposure at t=0 uses the value from the portfolio (fraction of year active).
/// For t=1..N_YEARS-1 exposure = 1.0 (full renewal years).
fn run_one(model: &mut FrequencyModel, policies: &[PolicyV2], seed: u64) -> [YearResult; N_YEARS] {
    let n = policies.len();
    let mut rng = SmallRng::seed_from_u64(seed);

    // Per-policy mutable state — cloned from the portfolio at the start of each sim.
    let mut veh_age:      Vec<f32>     = policies.iter().map(|p| p.veh_age).collect();
    let mut driv_age:     Vec<f32>     = policies.iter().map(|p| p.driv_age).collect();
    // Rolling 3-year window: [oldest (t-3), t-2, newest (t-1)]
    let mut claim_window: Vec<[u32; 3]> = policies.iter().map(|p| p.claims_hist).collect();

    // Pre-allocate the result array. MaybeUninit would avoid the dummy values but
    // this is cleaner — we overwrite every element in the loop below.
    let mut results: [YearResult; N_YEARS] = std::array::from_fn(|_| YearResult {
        total_claims: 0,
        frequency:    0.0,
    });

    for year in 0..N_YEARS {
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
            .expect("ONNX inference failed in multi-year simulation");

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

        // Frequency relative to the portfolio size (all policies active = n).
        results[year] = YearResult {
            total_claims,
            frequency: total_claims as f64 / n as f64,
        };

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
/// Key architectural point: from year 1 onward each simulation has its own
/// distinct PriorClaims3Y (because the Poisson draws in year 0 differ).
/// We therefore need an independent ONNX session per simulation.  Loading
/// n_sims separate sessions would be wasteful, so we pre-allocate exactly
/// one session per Rayon worker thread and reuse it across that thread's sims.
///
/// rayon::current_thread_index() returns the index of the calling worker in
/// [0, n_threads), which we use to index into the per-thread model pool.
/// Since each Rayon thread only ever calls models[its_own_index].lock(), there
/// is zero lock contention — the Mutex is just a safe way to hold &mut access.
pub fn run_parallel(
    model_path: &Path,
    policies:   &[PolicyV2],
    n_sims:     usize,
) -> Vec<[YearResult; N_YEARS]> {
    let n_threads = rayon::current_num_threads();

    // Load one FrequencyModel per Rayon worker thread.
    let models: Vec<Mutex<FrequencyModel>> = (0..n_threads)
        .map(|_| {
            Mutex::new(
                FrequencyModel::load(model_path)
                    .expect("failed to load v2 ONNX model for thread"),
            )
        })
        .collect();

    (0..n_sims)
        .into_par_iter()
        .map(|sim_i| {
            let idx   = rayon::current_thread_index().unwrap_or(0);
            let mut m = models[idx].lock().unwrap();
            run_one(&mut m, policies, sim_i as u64)
        })
        .collect()
}

/// Print per-year summary statistics across all simulations.
pub fn print_stats(results: &[[YearResult; N_YEARS]], n_policies: usize) {
    println!();
    println!(
        "=== Multi-Year Claim Simulation ({} sims × {} years × {} policies) ===",
        results.len(),
        N_YEARS,
        n_policies,
    );
    println!();
    println!(
        "{:<6}  {:>12}  {:>10}  {:>10}  {:>10}  {:>10}",
        "Year", "Mean claims", "Mean freq", "P50 freq", "P95 freq", "P99 freq"
    );
    println!("{}", "-".repeat(62));

    for year in 0..N_YEARS {
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
