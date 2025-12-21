# ==============================================================================
# Enterprise AI Platform - Terraform Configuration
# ==============================================================================
#
# This Terraform configuration provisions the infrastructure for the
# Enterprise AI Decision System including:
# - EKS/GKE Kubernetes cluster
# - Managed Kafka (MSK/Confluent)
# - RDS PostgreSQL for MLflow
# - Supporting networking and security
#
# ==============================================================================

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
  
  # Backend configuration - uncomment for production
  # backend "s3" {
  #   bucket         = "your-terraform-state-bucket"
  #   key            = "enterprise-ai/terraform.tfstate"
  #   region         = "us-west-2"
  #   encrypt        = true
  #   dynamodb_table = "terraform-locks"
  # }
}

# ==============================================================================
# Provider Configuration
# ==============================================================================

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "enterprise-ai"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# ==============================================================================
# Variables
# ==============================================================================

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "staging"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "enterprise-ai-cluster"
}

variable "cluster_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "node_instance_types" {
  description = "EC2 instance types for nodes"
  type        = list(string)
  default     = ["m5.large", "m5.xlarge"]
}

variable "min_nodes" {
  description = "Minimum number of nodes"
  type        = number
  default     = 2
}

variable "max_nodes" {
  description = "Maximum number of nodes"
  type        = number
  default     = 10
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

# ==============================================================================
# Data Sources
# ==============================================================================

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# ==============================================================================
# VPC Configuration
# ==============================================================================

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"
  
  name = "${var.cluster_name}-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway   = true
  single_nat_gateway   = var.environment != "production"
  enable_dns_hostnames = true
  
  public_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                    = 1
  }
  
  private_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb"           = 1
  }
}

# ==============================================================================
# EKS Cluster
# ==============================================================================

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"
  
  cluster_name    = var.cluster_name
  cluster_version = var.cluster_version
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  cluster_endpoint_public_access = true
  
  # Managed node groups
  eks_managed_node_groups = {
    ml-workers = {
      name = "ml-workers"
      
      instance_types = var.node_instance_types
      capacity_type  = "ON_DEMAND"
      
      min_size     = var.min_nodes
      max_size     = var.max_nodes
      desired_size = var.min_nodes
      
      labels = {
        role = "ml-worker"
      }
      
      taints = []
    }
    
    # GPU nodes for ML training (optional)
    # gpu-workers = {
    #   name           = "gpu-workers"
    #   instance_types = ["g4dn.xlarge"]
    #   capacity_type  = "SPOT"
    #   min_size       = 0
    #   max_size       = 4
    #   desired_size   = 0
    #   
    #   labels = {
    #     role = "gpu-worker"
    #     "nvidia.com/gpu" = "true"
    #   }
    # }
  }
  
  # OIDC for service accounts
  enable_irsa = true
  
  # Cluster addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
  }
}

# ==============================================================================
# Amazon MSK (Managed Kafka)
# ==============================================================================
#
# For managed Kafka, you have several options:
#
# OPTION 1: Amazon MSK (AWS Managed Kafka)
# -----------------------------------------

resource "aws_msk_cluster" "kafka" {
  count = var.environment == "production" ? 1 : 0
  
  cluster_name           = "${var.cluster_name}-kafka"
  kafka_version          = "3.5.1"
  number_of_broker_nodes = 3
  
  broker_node_group_info {
    instance_type   = "kafka.m5.large"
    client_subnets  = module.vpc.private_subnets
    security_groups = [aws_security_group.kafka[0].id]
    
    storage_info {
      ebs_storage_info {
        volume_size = 100
      }
    }
  }
  
  encryption_info {
    encryption_in_transit {
      client_broker = "TLS"
      in_cluster    = true
    }
  }
  
  configuration_info {
    arn      = aws_msk_configuration.kafka[0].arn
    revision = aws_msk_configuration.kafka[0].latest_revision
  }
  
  tags = {
    Name = "${var.cluster_name}-kafka"
  }
}

resource "aws_msk_configuration" "kafka" {
  count = var.environment == "production" ? 1 : 0
  
  name              = "${var.cluster_name}-kafka-config"
  kafka_versions    = ["3.5.1"]
  
  server_properties = <<PROPERTIES
auto.create.topics.enable=true
default.replication.factor=3
min.insync.replicas=2
num.io.threads=8
num.network.threads=5
num.partitions=6
num.replica.fetchers=2
socket.request.max.bytes=104857600
PROPERTIES
}

resource "aws_security_group" "kafka" {
  count = var.environment == "production" ? 1 : 0
  
  name_prefix = "${var.cluster_name}-kafka-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 9092
    to_port     = 9098
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# OPTION 2: Confluent Cloud (Notes)
# ---------------------------------
# For Confluent Cloud, use the Confluent Terraform provider:
#
# provider "confluent" {
#   cloud_api_key    = var.confluent_api_key
#   cloud_api_secret = var.confluent_api_secret
# }
#
# resource "confluent_environment" "production" {
#   display_name = "enterprise-ai"
# }
#
# resource "confluent_kafka_cluster" "main" {
#   display_name = "enterprise-ai-kafka"
#   availability = "MULTI_ZONE"
#   cloud        = "AWS"
#   region       = var.aws_region
#   standard {}
#   environment {
#     id = confluent_environment.production.id
#   }
# }

# ==============================================================================
# RDS PostgreSQL for MLflow
# ==============================================================================

resource "aws_db_subnet_group" "mlflow" {
  name       = "${var.cluster_name}-mlflow"
  subnet_ids = module.vpc.private_subnets
  
  tags = {
    Name = "${var.cluster_name}-mlflow"
  }
}

resource "aws_security_group" "mlflow_db" {
  name_prefix = "${var.cluster_name}-mlflow-db-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_instance" "mlflow" {
  identifier = "${var.cluster_name}-mlflow"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.db_instance_class
  
  allocated_storage     = 50
  max_allocated_storage = 200
  storage_type          = "gp3"
  storage_encrypted     = true
  
  db_name  = "mlflow"
  username = "mlflow_admin"
  password = random_password.mlflow_db.result
  
  db_subnet_group_name   = aws_db_subnet_group.mlflow.name
  vpc_security_group_ids = [aws_security_group.mlflow_db.id]
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "Mon:04:00-Mon:05:00"
  
  skip_final_snapshot       = var.environment != "production"
  final_snapshot_identifier = var.environment == "production" ? "${var.cluster_name}-mlflow-final" : null
  
  tags = {
    Name = "${var.cluster_name}-mlflow"
  }
}

resource "random_password" "mlflow_db" {
  length  = 32
  special = false
}

# Store password in Secrets Manager
resource "aws_secretsmanager_secret" "mlflow_db" {
  name = "${var.cluster_name}/mlflow-db-password"
}

resource "aws_secretsmanager_secret_version" "mlflow_db" {
  secret_id = aws_secretsmanager_secret.mlflow_db.id
  secret_string = jsonencode({
    username = aws_db_instance.mlflow.username
    password = random_password.mlflow_db.result
    host     = aws_db_instance.mlflow.endpoint
    port     = 5432
    database = aws_db_instance.mlflow.db_name
  })
}

# ==============================================================================
# S3 Bucket for MLflow Artifacts
# ==============================================================================

resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "${var.cluster_name}-mlflow-artifacts-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ==============================================================================
# Outputs
# ==============================================================================

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "mlflow_db_endpoint" {
  description = "MLflow database endpoint"
  value       = aws_db_instance.mlflow.endpoint
}

output "mlflow_artifacts_bucket" {
  description = "S3 bucket for MLflow artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.bucket
}

output "kafka_bootstrap_brokers" {
  description = "Kafka bootstrap brokers"
  value       = var.environment == "production" ? aws_msk_cluster.kafka[0].bootstrap_brokers_tls : "N/A (not provisioned in ${var.environment})"
}

output "kubectl_config_command" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}
