# Terraform Guide — Claim Simulation EC2

This guide automates the EC2 setup described in `EC2_SETUP_GUIDE.md` using Terraform.

---

## What It Provisions

| Resource | Value |
|---|---|
| Region | eu-central-1 (Frankfurt) |
| Instance type | c6i.4xlarge (16 vCPU, 32 GB RAM) |
| AMI | Amazon Linux 2023 (latest, resolved automatically) |
| Storage | 30 GiB gp3 |
| Security group | SSH port 22 from your IP only |
| Key pair | claim-simulation-frankfurt (must exist in AWS) |
| Elastic IP | Yes — IP is stable across stop/start |

The user data script automatically installs: Python 3.11, git, build essentials, Rust, and
ONNX Runtime 1.24.3 to `/opt/onnxruntime`. `ORT_DYLIB_PATH` and `LD_LIBRARY_PATH` are set
in `/etc/environment` and `~/.bashrc`.

---

## Prerequisites

- Terraform >= 1.5 installed (`brew install terraform`)
- AWS CLI installed (`brew install awscli`)
- An IAM user with `AmazonEC2FullAccess` and an access key
- The key pair `claim-simulation-frankfurt` already exists in eu-central-1

---

## 1. Configure AWS Credentials

```bash
aws configure --profile terraform-ec2
```

Enter your access key ID, secret access key, default region `eu-central-1`, output format `json`.

Verify:
```bash
aws sts get-caller-identity --profile terraform-ec2
```

---

## 2. Adapt `main.tf` for Your Setup

> **Other users must review these two settings in `main.tf` before running:**

### AWS profile name
```hcl
provider "aws" {
  region  = "eu-central-1"
  profile = "terraform-ec2"   # ← change to your AWS CLI profile name
}
```

### Key pair name
```hcl
variable "key_name" {
  default = "claim-simulation-frankfurt"   # ← must match a key pair in your AWS account
}
```

If you use a different key pair name, either edit the default in `variables.tf` or pass it
at apply time:
```bash
terraform apply -var="key_name=your-key-pair-name" ...
```

---

## 3. Initialize and Apply

```bash
cd terraform

# One-time: download the AWS provider
terraform init

# Deploy — inserts your current public IP for SSH access
terraform apply -var="allowed_cidr=$(curl -s https://checkip.amazonaws.com)/32"
```

Review the plan Terraform prints, then type `yes` to confirm.

After apply, Terraform outputs:
- `public_ip` — the Elastic IP (stable across restarts)
- `instance_id` — the EC2 instance ID
- `ssh_command` — ready-to-use SSH command

---

## 4. Configure VS Code Remote SSH

Add the host to `~/.ssh/config` (replace `<ELASTIC_IP>` with the `public_ip` output):

```
Host claim-sim-ec2
    HostName <ELASTIC_IP>
    User ec2-user
    IdentityFile ~/.ssh/claim-simulation-frankfurt.pem
```

Connect from VS Code:
```
Cmd+Shift+P → "Remote-SSH: Connect to Host" → claim-sim-ec2
```

Wait ~3–5 minutes after `terraform apply` for the user data bootstrap to complete before
connecting.

---

## 5. Run the Pipeline

Once connected, follow `EC2_SETUP_GUIDE.md` from **section 4** onward (clone repo, set up
venv, run the data pipeline and benchmark).

---

## 6. Stopping the Instance

**Stop** the instance from the AWS Console when done — do not run `terraform destroy` unless
you want to delete everything including the EBS disk.

- Stopped instances incur only EBS storage cost (~$2.40/month for 30 GiB)
- The Elastic IP stays associated, so the IP does not change on restart
- To restart: start the instance in the console, then SSH in as before

### Full teardown (when no longer needed)

```bash
terraform destroy -var="allowed_cidr=$(curl -s https://checkip.amazonaws.com)/32"
```

This deletes the instance, security group, and Elastic IP.
