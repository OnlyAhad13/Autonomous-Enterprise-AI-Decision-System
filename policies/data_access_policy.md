# Data Access Policy

Access control policies for the Enterprise AI Decision System.

## Table of Contents

- [Principles](#principles)
- [IAM Roles](#iam-roles)
- [Least-Privilege Rules](#least-privilege-rules)
- [SQL Redaction Examples](#sql-redaction-examples)

---

## Principles

1. **Least Privilege**: Grant minimum permissions required
2. **Separation of Duties**: Different roles for different functions
3. **Defense in Depth**: Multiple layers of access control
4. **Audit Everything**: Log all data access

---

## IAM Roles

### AWS IAM Role Definitions

```json
{
  "Roles": [
    {
      "RoleName": "ml-inference-role",
      "Description": "Read-only access for inference services",
      "AssumeRolePolicyDocument": {
        "Version": "2012-10-17",
        "Statement": [{
          "Effect": "Allow",
          "Principal": {
            "Service": "eks.amazonaws.com"
          },
          "Action": "sts:AssumeRoleWithWebIdentity"
        }]
      }
    },
    {
      "RoleName": "ml-training-role",
      "Description": "Read/write for training pipelines",
      "AssumeRolePolicyDocument": {
        "Version": "2012-10-17",
        "Statement": [{
          "Effect": "Allow",
          "Principal": {
            "Service": "sagemaker.amazonaws.com"
          },
          "Action": "sts:AssumeRole"
        }]
      }
    },
    {
      "RoleName": "ml-admin-role",
      "Description": "Full access for platform admins",
      "AssumeRolePolicyDocument": {
        "Version": "2012-10-17",
        "Statement": [{
          "Effect": "Allow",
          "Principal": {
            "AWS": "arn:aws:iam::ACCOUNT:root"
          },
          "Action": "sts:AssumeRole",
          "Condition": {
            "Bool": {"aws:MultiFactorAuthPresent": "true"}
          }
        }]
      }
    }
  ]
}
```

### Terraform IAM Policies

```hcl
# Inference Role - Read-only
resource "aws_iam_role" "ml_inference" {
  name = "ml-inference-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = aws_iam_openid_connect_provider.eks.arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${aws_iam_openid_connect_provider.eks.url}:sub" = "system:serviceaccount:ml-services:predict-service"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "inference_s3_read" {
  name = "s3-read-models"
  role = aws_iam_role.ml_inference.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.mlflow_bucket}",
          "arn:aws:s3:::${var.mlflow_bucket}/*"
        ]
      }
    ]
  })
}

# Training Role - Read/Write
resource "aws_iam_role" "ml_training" {
  name = "ml-training-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = ["sagemaker.amazonaws.com", "airflow.amazonaws.com"]
      }
      Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "training_full" {
  name = "training-permissions"
  role = aws_iam_role.ml_training.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.mlflow_bucket}/*",
          "arn:aws:s3:::${var.training_bucket}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "glue:GetTable",
          "glue:GetDatabase",
          "glue:GetPartitions"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "glue:ResourceTag/project" = "enterprise-ai"
          }
        }
      }
    ]
  })
}
```

---

## Least-Privilege Rules

### Role Permission Matrix

| Role | S3 Models | S3 Data | RDS MLflow | Kafka | Secrets |
|------|-----------|---------|------------|-------|---------|
| `ml-inference` | Read | - | Read | Consume | Read |
| `ml-training` | Read/Write | Read | Read/Write | Produce | Read |
| `ml-admin` | Full | Full | Full | Full | Full |
| `data-engineer` | - | Read/Write | - | Full | - |
| `analyst` | - | Read | Read | - | - |

### Service Account Mappings

| Service | K8s ServiceAccount | AWS Role |
|---------|-------------------|----------|
| predict-api | predict-service | ml-inference-role |
| rag-api | rag-service | ml-inference-role |
| training-dag | airflow-worker | ml-training-role |
| agent-core | agent-service | ml-training-role |

### Network Policies

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: predict-service-policy
  namespace: ml-services
spec:
  podSelector:
    matchLabels:
      app: predict
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress
      ports:
        - port: 8000
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: ml-services
      ports:
        - port: 5432  # RDS
    - to:
        - ipBlock:
            cidr: 10.0.0.0/8
      ports:
        - port: 443  # S3
```

---

## SQL Redaction Examples

### PostgreSQL Row-Level Security

```sql
-- Enable RLS on sensitive tables
ALTER TABLE customers ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see non-PII or their own data
CREATE POLICY customer_access ON customers
FOR SELECT
USING (
  current_user = 'admin' 
  OR current_user = 'compliance'
  OR customer_id = current_setting('app.current_customer_id')::int
);

-- Create redacted view for analysts
CREATE VIEW customers_redacted AS
SELECT 
  customer_id,
  -- Redact email: show first char + domain
  CONCAT(LEFT(email, 1), '***@', SPLIT_PART(email, '@', 2)) AS email_masked,
  -- Redact SSN: show last 4 only
  CONCAT('***-**-', RIGHT(ssn, 4)) AS ssn_masked,
  -- Redact phone: show last 4
  CONCAT('***-***-', RIGHT(phone, 4)) AS phone_masked,
  -- Keep non-PII fields
  subscription_tier,
  created_at,
  -- Hash identifier for joins
  MD5(email || current_setting('app.pii_salt')) AS email_hash
FROM customers;

-- Grant access to redacted view only
GRANT SELECT ON customers_redacted TO analyst_role;
REVOKE ALL ON customers FROM analyst_role;
```

### Dynamic Data Masking

```sql
-- Create masking function
CREATE OR REPLACE FUNCTION mask_pii(
  value TEXT,
  data_type TEXT DEFAULT 'text'
) RETURNS TEXT AS $$
BEGIN
  -- Check if user has PII access
  IF current_user IN ('admin', 'compliance', 'data_protection') THEN
    RETURN value;
  END IF;
  
  CASE data_type
    WHEN 'email' THEN
      RETURN CONCAT(LEFT(value, 1), '***@', SPLIT_PART(value, '@', 2));
    WHEN 'ssn' THEN
      RETURN CONCAT('***-**-', RIGHT(value, 4));
    WHEN 'phone' THEN
      RETURN CONCAT('***-***-', RIGHT(value, 4));
    WHEN 'name' THEN
      RETURN CONCAT(LEFT(value, 1), '***');
    WHEN 'address' THEN
      RETURN '*** REDACTED ***';
    ELSE
      RETURN '***';
  END CASE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Usage in views
CREATE VIEW orders_safe AS
SELECT 
  order_id,
  mask_pii(customer_email, 'email') AS customer_email,
  mask_pii(shipping_address, 'address') AS shipping_address,
  order_total,
  order_date
FROM orders;
```

### Spark SQL Redaction

```sql
-- Create UDFs for masking
CREATE TEMPORARY FUNCTION mask_email AS 'com.company.udf.MaskEmail';
CREATE TEMPORARY FUNCTION mask_ssn AS 'com.company.udf.MaskSSN';
CREATE TEMPORARY FUNCTION hash_pii AS 'com.company.udf.HashPII';

-- Redacted export query
SELECT 
  user_id,
  mask_email(email) AS email_masked,
  mask_ssn(ssn) AS ssn_masked,
  hash_pii(email, 'salt123') AS email_hash,
  subscription_tier,
  signup_date
FROM users
WHERE region = 'US';
```

### Delta Lake Column-Level Access

```sql
-- Create fine-grained access control
CREATE TABLE customers_secured (
  customer_id BIGINT,
  email STRING MASK mask_pii,
  ssn STRING MASK mask_pii,
  name STRING,
  subscription_tier STRING
)
USING DELTA
TBLPROPERTIES (
  'delta.enableRowTracking' = 'true',
  'delta.columnMapping.mode' = 'name'
);

-- Grant access with column restrictions
GRANT SELECT (customer_id, name, subscription_tier) 
ON customers_secured 
TO analyst_group;

-- Full access for admins
GRANT SELECT ON customers_secured TO admin_group;
```

---

## Audit Queries

### Access Log Analysis

```sql
-- Who accessed PII in the last 24 hours?
SELECT 
  user_name,
  query_text,
  accessed_tables,
  timestamp
FROM query_audit_log
WHERE timestamp > NOW() - INTERVAL '24 hours'
AND accessed_columns && ARRAY['email', 'ssn', 'phone', 'address']
ORDER BY timestamp DESC;

-- Unusual access patterns
SELECT 
  user_name,
  COUNT(*) AS query_count,
  COUNT(DISTINCT accessed_tables) AS tables_accessed
FROM query_audit_log
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY user_name
HAVING COUNT(*) > 100
ORDER BY query_count DESC;
```

---

## Compliance Mapping

| Regulation | Requirement | Implementation |
|------------|-------------|---------------|
| GDPR Art. 25 | Data minimization | Column-level access |
| GDPR Art. 32 | Encryption | TLS + KMS |
| HIPAA | Access controls | RBAC + audit logs |
| SOC 2 | Monitoring | Audit logging |
| PCI DSS | Data masking | SQL redaction |
