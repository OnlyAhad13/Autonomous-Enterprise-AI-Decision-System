# Terraform Infrastructure

Infrastructure as Code for the Autonomous Enterprise AI Decision System.

> ⚠️ **PLACEHOLDER**: This directory contains infrastructure templates to be customized for your environment.

## Overview

This module provisions:
- Kubernetes cluster (EKS/GKE/AKS)
- Managed Kafka (MSK/Confluent Cloud)
- Object storage (S3/GCS/Azure Blob)
- Managed databases (RDS/Cloud SQL)
- Networking (VPC, subnets, security groups)
- IAM roles and policies

## Structure

```
terraform/
├── environments/
│   ├── dev/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── terraform.tfvars
│   ├── staging/
│   └── prod/
├── modules/
│   ├── kubernetes/      # K8s cluster
│   ├── kafka/           # Kafka cluster
│   ├── storage/         # Object storage
│   ├── database/        # Databases
│   ├── networking/      # VPC, subnets
│   └── iam/             # IAM roles
├── main.tf
├── variables.tf
├── outputs.tf
└── versions.tf
```

## Cloud Provider Support

| Provider | Status | Notes |
|----------|--------|-------|
| AWS | ✅ Ready | Primary target |
| GCP | 🔄 Planned | Terraform modules prepared |
| Azure | 🔄 Planned | Terraform modules prepared |

## Prerequisites

- Terraform >= 1.6.0
- Cloud provider CLI configured
- Required IAM permissions

## Usage

```bash
# Initialize Terraform
cd environments/dev
terraform init

# Plan changes
terraform plan -out=tfplan

# Apply changes
terraform apply tfplan

# Destroy (careful!)
terraform destroy
```

## State Management

Remote state is stored in:
- **AWS**: S3 with DynamoDB locking
- **GCP**: GCS with locking
- **Azure**: Azure Storage with locking

```hcl
# Example backend configuration
terraform {
  backend "s3" {
    bucket         = "your-terraform-state"
    key            = "enterprise-ai/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}
```

## Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `environment` | Deployment environment | Yes |
| `region` | Cloud region | Yes |
| `cluster_name` | Kubernetes cluster name | Yes |
| `node_count` | Number of K8s nodes | No (default: 3) |

## Security Notes

- Never commit `terraform.tfvars` with secrets
- Use Vault or cloud secrets manager for sensitive values
- Enable state encryption
- Restrict state bucket access
