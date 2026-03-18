# EC2 Setup Guide — Claim Simulation

This guide documents how to launch an AWS EC2 instance for running the Monte Carlo simulation, install all dependencies, and connect VS Code remotely.

---

## Prerequisites

- AWS account with access to EC2 (eu-central-1 / Frankfurt)
- VS Code with the **Remote - SSH** extension (Microsoft) installed
- A GitHub Personal Access Token (PAT) with repo read access

---

## 1. Launch EC2 Instance

> **Alternatively**, use Terraform to automate steps 1–3 (instance provisioning, SSH config,
> and bootstrapping). See [terraform/TERRAFORM_GUIDE.md](terraform/TERRAFORM_GUIDE.md),
> then continue from **step 4** of this guide.

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

## 6. Run the Simulation

Once the environment is set up, follow [SIMULATION_GUIDE.md](SIMULATION_GUIDE.md) for the
data pipeline, benchmark, and instance shutdown instructions.

---

## Known Issues

- `Cargo.toml` was originally committed as `cargo.toml` (lowercase). Linux is case-sensitive — ensure the filename is `Cargo.toml` in the repo.
- The `.venv` must be created at the **project root** (`~/claim-simulation/`), not inside `rust/`.
