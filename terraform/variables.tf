variable "allowed_cidr" {
  description = "Your IP address in CIDR notation for SSH access (e.g. 203.0.113.42/32)"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "c6i.4xlarge"
}

variable "key_name" {
  description = "Name of the existing EC2 key pair"
  type        = string
  default     = "claim-simulation-frankfurt"
}

variable "volume_size_gb" {
  description = "Root EBS volume size in GiB"
  type        = number
  default     = 30
}
