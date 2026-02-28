use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Poisson};
use rayon::prelude::*;

/// Result of a single simulation run over the full portfolio.
pub struct SimResult {
    /// Total number of claims drawn across all policies.
    pub total_claims: u64,
    /// Claims per unit of exposure: total_claims / total_exposure.
    pub frequency: f64,
}

/// Run one simulation: draw Poisson(λ) for each policy and sum.
///
/// `lambdas` are pre-computed by `FrequencyModel::compute_lambdas` — one λ
/// per policy, already exposure-adjusted. This avoids repeating ONNX inference
/// inside the parallel loop (the model is deterministic anyway).
pub fn run_one(lambdas: &[f64], total_exposure: f64, rng: &mut impl rand::Rng) -> SimResult {
    let total_claims: u64 = lambdas
        .iter()
        .map(|&lambda| {
            // Draw from Poisson(λ). The result is f64 in rand_distr; we cast to u64.
            Poisson::new(lambda)
                .expect("lambda is positive")
                .sample(rng) as u64
        })
        .sum();

    SimResult {
        total_claims,
        frequency: total_claims as f64 / total_exposure,
    }
}

/// Run `n_sims` simulations in parallel using Rayon.
///
/// Each simulation gets its own RNG seeded by its index, giving:
/// - Reproducibility: same seeds → same results every run.
/// - Independence: no shared state, no locks needed between threads.
///
/// In Python terms this is like:
///   [run_one(lambdas, seed=i) for i in range(n_sims)]
/// but executed in parallel across all CPU cores.
pub fn run_parallel(lambdas: &[f64], total_exposure: f64, n_sims: usize) -> Vec<SimResult> {
    (0..n_sims)
        .into_par_iter()
        .map(|i| {
            // Each thread gets its own lightweight RNG.
            // SmallRng is fast and sufficient for simulation (not cryptographic).
            let mut rng = SmallRng::seed_from_u64(i as u64);
            run_one(lambdas, total_exposure, &mut rng)
        })
        .collect()
}

/// Print summary statistics — standard actuarial output.
///
/// P50   = best estimate / median
/// P95   = 1-in-20 year loss
/// P99.5 = Solvency II VaR level
pub fn print_stats(results: &[SimResult], total_exposure: f64) {
    let n = results.len() as f64;

    let mean_claims = results.iter().map(|r| r.total_claims as f64).sum::<f64>() / n;
    let mean_freq = results.iter().map(|r| r.frequency).sum::<f64>() / n;
    let var_freq = results.iter().map(|r| (r.frequency - mean_freq).powi(2)).sum::<f64>() / n;
    let std_freq = var_freq.sqrt();

    // Sort a copy of frequencies for percentile calculation.
    let mut freqs: Vec<f64> = results.iter().map(|r| r.frequency).collect();
    freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Linear interpolation percentile — same as numpy's default.
    let pct = |p: f64| -> f64 {
        let idx = p * (freqs.len() - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        let frac = idx - lo as f64;
        freqs[lo] * (1.0 - frac) + freqs[hi] * frac
    };

    println!();
    println!("=== Claim Simulation Results ({} simulations) ===", results.len());
    println!("Portfolio exposure: {:.2} policy-years", total_exposure);
    println!();
    println!("Expected claims per simulation:  {:.2}", mean_claims);
    println!();
    println!("Claim frequency (claims / policy-year):");
    println!("  Mean:   {:.5}", mean_freq);
    println!("  Std:    {:.5}", std_freq);
    println!("  P50:    {:.5}", pct(0.50));
    println!("  P75:    {:.5}", pct(0.75));
    println!("  P95:    {:.5}", pct(0.95));
    println!("  P99:    {:.5}", pct(0.99));
    println!("  P99.5:  {:.5}", pct(0.995));
    println!();
}
