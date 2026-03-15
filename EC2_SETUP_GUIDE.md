# EC2 Setup Guide — Claim Simulation

This guide documents how to launch an AWS EC2 instance for running the Monte Carlo simulation, install all dependencies, and connect VS Code remotely.

---

## Prerequisites

- AWS account with access to EC2 (eu-central-1 / Frankfurt)
- VS Code with the **Remote - SSH** extension (Microsoft) installed
- A GitHub Personal Access Token (PAT) with repo read access

---

## 1. Launch EC2 Instance

In the AWS Console → EC2 → Launch Instance:

| Setting | Value |
|---|---|
| Name | `claim-simulation` |
| AMI | Amazon Linux 2023 |
| Instance type | `c6i.4xlarge` (16 vCPU, 32 GB RAM) |
| Storage | 30 GiB gp3 |
| Security group | Allow SSH (port 22) from your IP only |

**Key pair:**
- Click "Create new key pair"
- Name: `claim-simulation-frankfurt`
- Type: RSA, format: `.pem`
- Download and move to `~/.ssh/`:

```bash
mv ~/Downloads/claim-simulation-frankfurt.pem ~/.ssh/
chmod 400 ~/.ssh/claim-simulation-frankfurt.pem
```

Click **Launch instance** and wait for status to show **Running**. Note the **Public IPv4 address**.

> ⚠️ The public IP changes on every restart. Consider allocating an Elastic IP for repeated use.

---

## 2. Configure VS Code Remote SSH

Add the host to your SSH config:

```bash
echo '
Host claim-sim-ec2
    HostName <PUBLIC_IPv4>
    User ec2-user
    IdentityFile ~/.ssh/claim-simulation-frankfurt.pem' >> ~/.ssh/config
```

Connect from VS Code:
```
Cmd+Shift+P → "Remote-SSH: Connect to Host" → claim-sim-ec2
```

When prompted about the host fingerprint, type `yes` and press Enter.

The bottom-left corner of VS Code will show **`SSH: claim-sim-ec2`** when connected.

---

## 3. Bootstrap the Instance

Open a terminal in the VS Code remote window (`Ctrl+```) and run:

```bash
# System updates
sudo dnf update -y

# Python 3.11
sudo dnf install -y python3.11 python3.11-pip git

# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Build essentials
sudo dnf install -y gcc gcc-c++ make cmake openssl-devel

# ONNX Runtime 1.24.3
wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-linux-x64-1.24.3.tgz
tar -xzf onnxruntime-linux-x64-1.24.3.tgz
sudo mv onnxruntime-linux-x64-1.24.3 /opt/onnxruntime

# Environment variables
echo 'export ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so.1.24.3' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify:
```bash
rustc --version        # rustc 1.94.0 or later
python3.11 --version   # Python 3.11.x
echo $ORT_DYLIB_PATH   # /opt/onnxruntime/lib/libonnxruntime.so.1.24.3
```

---

## 4. Clone the Repository

```bash
git clone https://<YOUR_PAT>@github.com/fischerdm/claim-simulation.git
cd claim-simulation
```

---

## 5. Set Up Python Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## 6. Run the Data Pipeline & Simulation

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

# 8. Build Rust
cd rust/
cargo build --release

# 9. Benchmark
QUICK_TEST=1 python python/benchmark.py

```

Build the Rust binary:
```bash
cd rust
cargo build --release
cd ..
```

Run benchmark:
```bash
QUICK_TEST=1 python python/benchmark.py
```

---

## 7. Stopping the Instance

When done, **stop** (do not terminate) the instance in the AWS Console to preserve the filesystem. You only pay for EBS storage (~$2.40/month for 30 GB) while stopped.

> When you restart, update the IP in `~/.ssh/config` with the new Public IPv4, or set up an Elastic IP to avoid this.

---

## Known Issues

- `Cargo.toml` was originally committed as `cargo.toml` (lowercase). Linux is case-sensitive — ensure the filename is `Cargo.toml` in the repo.
- The `.venv` must be created at the **project root** (`~/claim-simulation/`), not inside `rust/`.
