terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region  = "eu-central-1"
  profile = "terraform-ec2"
}

# Latest Amazon Linux 2023 x86_64 AMI
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "root-device-type"
    values = ["ebs"]
  }
}

resource "aws_security_group" "claim_sim" {
  name        = "claim-simulation-sg"
  description = "SSH access for claim simulation benchmark runs"

  ingress {
    description = "SSH from my IP"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "claim-simulation-sg"
    Project = "claim-simulation"
  }
}

resource "aws_instance" "claim_sim" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.claim_sim.id]

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.volume_size_gb
    delete_on_termination = true
  }

  user_data = <<-EOF
    #!/bin/bash
    set -euo pipefail

    # System updates
    dnf update -y

    # Python 3.11, git, build essentials
    dnf install -y python3.11 python3.11-pip git gcc gcc-c++ make cmake openssl-devel wget

    # Rust (install for ec2-user)
    su -l ec2-user -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path'
    echo 'source $HOME/.cargo/env' >> /home/ec2-user/.bashrc

    # ONNX Runtime 1.24.3
    cd /tmp
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-linux-x64-1.24.3.tgz
    tar -xzf onnxruntime-linux-x64-1.24.3.tgz
    mv onnxruntime-linux-x64-1.24.3 /opt/onnxruntime

    # System-wide env vars (read by PAM/login shells via pam_env)
    cat >> /etc/environment <<'ENVEOF'
ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so.1.24.3
LD_LIBRARY_PATH=/opt/onnxruntime/lib
ENVEOF

    # Also export in ec2-user's bashrc for interactive terminal sessions
    cat >> /home/ec2-user/.bashrc <<'BASHEOF'
export ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so.1.24.3
export LD_LIBRARY_PATH=/opt/onnxruntime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
BASHEOF
  EOF

  tags = {
    Name    = "claim-simulation"
    Project = "claim-simulation"
  }
}

resource "aws_eip" "claim_sim" {
  domain = "vpc"

  tags = {
    Name    = "claim-simulation-eip"
    Project = "claim-simulation"
  }
}

resource "aws_eip_association" "claim_sim" {
  instance_id   = aws_instance.claim_sim.id
  allocation_id = aws_eip.claim_sim.id
}
