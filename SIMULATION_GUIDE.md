# Simulation Guide — Claim Simulation

This guide covers the data pipeline, benchmark runs, and instance shutdown.
It assumes you have a fully set up environment (Python venv active, Rust built).

For EC2 setup, see [EC2_SETUP_GUIDE.md](EC2_SETUP_GUIDE.md).

---

## 1. Run the Data Pipeline

From the repo root:

```bash
make
```

This runs all seven steps in dependency order, skipping any that are already up to date:

| Step | Script | Output |
|------|--------|--------|
| 1 | `download.py` | `data/freMTPL2freq.csv` |
| 2 | `train.py v1` | `models/frequency_model.lgb` |
| 3 | `export_onnx.py v1` | `models/frequency_model.onnx` |
| 4 | `generate_history.py` | `data/freMTPL2freq_with_history.csv` |
| 5 | `train.py v2` | `models/frequency_model_v2.lgb` |
| 6 | `export_onnx.py v2` | `models/frequency_model_v2.onnx` |
| 7 | `export_portfolio.py` | `data/portfolio.csv`, `data/portfolio_v2.csv` |

Steps 4–6 depend on v1 (v2 uses synthetic claim history generated from the v1 ONNX model).
To rebuild from scratch: `make clean && make`.

---

## 2. Build the Rust Engine

```bash
cd rust
cargo build --release
cd ..
```

---

## 3. Run the Benchmark

### Quick test (pipeline validation, < 5 minutes)

```bash
QUICK_TEST=1 python python/benchmark.py
```

Uses a tiny grid (0.5% / 1% of portfolio, 200 / 500 sims) to validate the full pipeline
end-to-end. Not suitable for extrapolating compute throughput.

### Full benchmark

```bash
python python/benchmark.py
```

Results are saved to `results/benchmark_results.csv`. See [BENCHMARK.md](BENCHMARK.md) for
the study design, observed runtimes, and capacity planning guidance.

> For long runs (hours), use `tmux` to protect against connection drops:
> ```bash
> tmux new -s bench
> source .venv/bin/activate
> python python/benchmark.py
> # Detach: Ctrl+B then D
> # Reattach: tmux attach -t bench
> ```

---

## 4. Stopping the EC2 Instance

When done, **stop** (do not terminate) the instance in the AWS Console to preserve the
filesystem. You only pay for EBS storage (~$2.40/month for 30 GB) while stopped.

> If you launched with Terraform and used an Elastic IP, the IP stays stable across
> stop/start — no need to update `~/.ssh/config`. Without an Elastic IP, update the
> `HostName` in `~/.ssh/config` after each restart.
