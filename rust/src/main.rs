mod model;
mod portfolio;
mod simulator_multiyear;

use std::path::Path;
use std::time::Instant;

use portfolio::load_from_csv;
use simulator_multiyear::{print_stats, run_parallel};

const MODEL_PATH:     &str  = "../models/frequency_model_v2.onnx";
const PORTFOLIO_PATH: &str  = "../data/portfolio_v2.csv";

struct Config {
    n_sims:   usize,
    n_years:  usize,
    fraction: f64,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut cfg = Config { n_sims: 10_000, n_years: 5, fraction: 1.0 };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--n-sims" => {
                cfg.n_sims = args.get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .expect("--n-sims requires a positive integer");
                i += 2;
            }
            "--years" => {
                cfg.n_years = args.get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .expect("--years requires a positive integer");
                i += 2;
            }
            "--fraction" => {
                cfg.fraction = args.get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .expect("--fraction requires a float in (0, 1]");
                i += 2;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                i += 1;
            }
        }
    }
    cfg
}

fn main() -> anyhow::Result<()> {
    let cfg = parse_args();

    let all_policies = load_from_csv(Path::new(PORTFOLIO_PATH))?;
    let n_take = ((cfg.fraction * all_policies.len() as f64).round() as usize)
        .clamp(1, all_policies.len());
    let policies = &all_policies[..n_take];

    println!(
        "Portfolio: {}/{} policies ({:.0}%)  |  {} sims  |  {} years",
        policies.len(),
        all_policies.len(),
        cfg.fraction * 100.0,
        cfg.n_sims,
        cfg.n_years,
    );

    let t0      = Instant::now();
    let results = run_parallel(Path::new(MODEL_PATH), policies, cfg.n_sims, cfg.n_years);
    let elapsed = t0.elapsed();

    println!(
        "Done in {:.3?}  ({:.1} ms/simulation)",
        elapsed,
        elapsed.as_millis() as f64 / cfg.n_sims as f64,
    );
    print_stats(&results, policies.len());

    Ok(())
}
