# Simulation Guide — Claim Simulation

This guide covers the data pipeline, benchmark runs, and instance shutdown.
It assumes you have a fully set up environment (Python venv active, Rust built).

For EC2 setup, see [EC2_SETUP_GUIDE.md](EC2_SETUP_GUIDE.md).

---

## 1. Run the Data Pipeline

Run these steps in order from the repo root:

```bash
# 1. Download raw data
python python/data/download.py

# 2. Train v1 (will error at v2 — that's expected, v1 model + metadata are saved)
python python/train.py || true

# 3. Export v1 ONNX (will error at v2 — that's expected, v1 .onnx is saved)
python python/export_onnx.py || true

# 4. Generate history (now has feature_metadata.json + frequency_model.onnx)
python python/generate_history.py

# 5. Full train — now both v1 and v2 succeed
python python/train.py

# 6. Full ONNX export — now both succeed
python python/export_onnx.py

# 7. Portfolio export
python python/export_portfolio.py
```

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
