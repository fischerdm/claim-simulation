output "public_ip" {
  description = "Elastic IP address of the instance (stable across restarts)"
  value       = aws_eip.claim_sim.public_ip
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.claim_sim.id
}

output "ssh_command" {
  description = "SSH command to connect"
  value       = "ssh -i ~/.ssh/claim-simulation-frankfurt.pem ec2-user@${aws_eip.claim_sim.public_ip}"
}
