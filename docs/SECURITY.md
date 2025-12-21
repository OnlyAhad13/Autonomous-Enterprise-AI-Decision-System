# Security Guide

Enterprise AI Decision System security architecture and best practices.

## Table of Contents

- [Encryption](#encryption)
- [Role-Based Access Control](#role-based-access-control)
- [PII Masking](#pii-masking)
- [Audit Logging](#audit-logging)

---

## Encryption

### At Rest

| Component | Encryption | Key Management |
|-----------|-----------|----------------|
| S3 (MLflow artifacts) | AES-256 (SSE-S3) | AWS KMS |
| RDS (MLflow backend) | AES-256 | AWS KMS CMK |
| EBS (K8s volumes) | AES-256 | AWS KMS |
| Delta Lake | AES-256 | AWS KMS |

**Configuration:**

```hcl
# Terraform - S3 encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.mlflow.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

# RDS encryption
resource "aws_db_instance" "mlflow" {
  storage_encrypted = true
  kms_key_id        = aws_kms_key.mlflow.arn
}
```

### In Transit

| Connection | Protocol | Certificate |
|------------|----------|-------------|
| API endpoints | TLS 1.3 | ACM/Let's Encrypt |
| Kafka | mTLS | Internal CA |
| Database | TLS | RDS CA |
| Inter-service | mTLS | Istio/Linkerd |

**FastAPI TLS:**

```python
# uvicorn with TLS
uvicorn.run(
    app,
    host="0.0.0.0",
    port=443,
    ssl_keyfile="/certs/key.pem",
    ssl_certfile="/certs/cert.pem",
)
```

**Kafka TLS:**

```properties
security.protocol=SSL
ssl.truststore.location=/certs/kafka.truststore.jks
ssl.keystore.location=/certs/kafka.keystore.jks
```

---

## Role-Based Access Control

### Feature Store RBAC

| Role | Permissions | Use Case |
|------|-------------|----------|
| `feature-reader` | Read features | Inference services |
| `feature-writer` | Read/Write features | Training pipelines |
| `feature-admin` | Full access | Platform team |

**Feast Configuration:**

```yaml
# feature_store.yaml
project: enterprise_ai
registry: s3://bucket/registry.pb
provider: aws

permissions:
  - name: readers
    types: [FeatureView, FeatureService]
    actions: [READ]
    policy:
      role: feature-reader
  
  - name: writers
    types: [FeatureView, FeatureService]
    actions: [READ, WRITE]
    policy:
      role: feature-writer
```

### MLflow RBAC

| Role | Experiments | Models | Runs |
|------|-------------|--------|------|
| `mlflow-viewer` | Read | Read | Read |
| `mlflow-editor` | Read/Write | Read/Write | Read/Write |
| `mlflow-admin` | Full | Full | Full |

**MLflow with Basic Auth:**

```yaml
# mlflow-auth.yaml
default_permission: READ
admin_username: admin
authorization:
  function: mlflow_auth.auth:authenticate
```

**Custom Authorization:**

```python
# mlflow_auth/auth.py
from mlflow.server.auth import AuthorizationFunction

def authenticate(request) -> bool:
    user = get_user_from_token(request.headers.get("Authorization"))
    resource = get_mlflow_resource(request.path)
    
    return check_permission(user, resource, request.method)
```

### Kubernetes RBAC

```yaml
# ml-services-role.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: ml-service-role
  namespace: ml-services
rules:
  - apiGroups: [""]
    resources: ["secrets"]
    resourceNames: ["mlflow-credentials", "kafka-credentials"]
    verbs: ["get"]
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: predict-service-binding
  namespace: ml-services
subjects:
  - kind: ServiceAccount
    name: predict-service
roleRef:
  kind: Role
  name: ml-service-role
  apiGroup: rbac.authorization.k8s.io
```

---

## PII Masking

### Strategies

| Strategy | Use Case | Reversible |
|----------|----------|------------|
| **Pseudonymization** | Analytics | Yes (with key) |
| **Tokenization** | Payment data | Yes (with vault) |
| **Hashing** | Identifiers | No |
| **Redaction** | Logs/exports | No |

### Pseudonymization Implementation

```python
# services/privacy/pseudonymizer.py
import hashlib
import hmac
from cryptography.fernet import Fernet

class Pseudonymizer:
    """Pseudonymize PII fields with reversible encryption."""
    
    def __init__(self, key: bytes, salt: bytes):
        self.fernet = Fernet(key)
        self.salt = salt
    
    def pseudonymize(self, value: str) -> str:
        """Encrypt value for pseudonymized storage."""
        return self.fernet.encrypt(value.encode()).decode()
    
    def depseudonymize(self, token: str) -> str:
        """Decrypt pseudonymized value (requires key)."""
        return self.fernet.decrypt(token.encode()).decode()
    
    def hash_identifier(self, value: str) -> str:
        """One-way hash for joining datasets without exposing PII."""
        return hmac.new(
            self.salt,
            value.encode(),
            hashlib.sha256
        ).hexdigest()

# Usage in pipeline
pseudonymizer = Pseudonymizer(key=os.environ["PII_KEY"], salt=os.environ["PII_SALT"])

df["email_pseudo"] = df["email"].apply(pseudonymizer.pseudonymize)
df["user_hash"] = df["user_id"].apply(pseudonymizer.hash_identifier)
df = df.drop(columns=["email", "user_id"])  # Remove raw PII
```

### Spark PII Masking

```python
# spark_jobs/pii_mask.py
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

def mask_email(email: str) -> str:
    if not email or "@" not in email:
        return "***@***.***"
    local, domain = email.split("@")
    return f"{local[0]}***@{domain}"

mask_email_udf = F.udf(mask_email, StringType())

df = df.withColumn("email_masked", mask_email_udf(F.col("email")))
df = df.withColumn("ssn_redacted", F.lit("***-**-****"))
df = df.withColumn("phone_masked", 
    F.concat(F.lit("***-***-"), F.substring("phone", -4, 4)))
```

### Delta Lake Column Masking

```sql
-- Create masking function
CREATE FUNCTION mask_pii(value STRING)
RETURNS STRING
RETURN CASE 
  WHEN current_user() IN ('admin', 'compliance') THEN value
  ELSE '***REDACTED***'
END;

-- Apply to table
ALTER TABLE customers 
ALTER COLUMN ssn SET MASK mask_pii;
```

---

## Audit Logging

### Agent Action Audit

```python
# services/audit/logger.py
import json
import logging
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class AuditEvent:
    """Audit event for agent actions."""
    timestamp: str
    event_type: str
    actor: str
    action: str
    resource: str
    resource_id: str
    status: str
    details: dict
    ip_address: str = None
    user_agent: str = None

class AuditLogger:
    """Structured audit logging for compliance."""
    
    def __init__(self, service_name: str):
        self.logger = logging.getLogger(f"audit.{service_name}")
        self.service = service_name
    
    def log_agent_action(
        self,
        actor: str,
        action: str,
        resource: str,
        resource_id: str,
        status: str,
        details: dict = None,
    ) -> None:
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type="AGENT_ACTION",
            actor=actor,
            action=action,
            resource=resource,
            resource_id=resource_id,
            status=status,
            details=details or {},
        )
        self.logger.info(json.dumps(asdict(event)))
    
    def log_data_access(
        self,
        actor: str,
        dataset: str,
        operation: str,
        row_count: int,
        columns_accessed: list,
    ) -> None:
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            event_type="DATA_ACCESS",
            actor=actor,
            action=operation,
            resource="dataset",
            resource_id=dataset,
            status="success",
            details={
                "row_count": row_count,
                "columns": columns_accessed,
            },
        )
        self.logger.info(json.dumps(asdict(event)))

# Usage
audit = AuditLogger("agent-core")

audit.log_agent_action(
    actor="agent-retraining",
    action="TRIGGER_RETRAIN",
    resource="dag",
    resource_id="auto_retrain_dag",
    status="success",
    details={"drift_score": 0.15, "model": "forecasting-v2"},
)
```

### Audit Log Schema

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "event_type": "AGENT_ACTION",
  "actor": "agent-retraining",
  "action": "TRIGGER_RETRAIN",
  "resource": "dag",
  "resource_id": "auto_retrain_dag",
  "status": "success",
  "details": {
    "drift_score": 0.15,
    "model": "forecasting-v2",
    "approved_by": "system"
  }
}
```

### Centralized Logging

```yaml
# fluentbit-config.yaml
[INPUT]
    Name   tail
    Path   /var/log/audit/*.log
    Tag    audit.*
    Parser json

[FILTER]
    Name   record_modifier
    Match  audit.*
    Record cluster enterprise-ai
    Record environment production

[OUTPUT]
    Name   opensearch
    Match  audit.*
    Host   opensearch.logging.svc
    Port   9200
    Index  audit-logs
    Type   _doc
```

---

## Compliance Checklist

| Requirement | Implementation | Status |
|-------------|---------------|--------|
| Encryption at rest | KMS + SSE | ✅ |
| Encryption in transit | TLS 1.3 | ✅ |
| Access control | RBAC | ✅ |
| PII protection | Pseudonymization | ✅ |
| Audit trails | Structured logging | ✅ |
| Data retention | 90-day policy | Configure |
| Incident response | Runbooks | Document |

---

## References

- [AWS Security Best Practices](https://aws.amazon.com/security/)
- [OWASP API Security](https://owasp.org/www-project-api-security/)
- [GDPR Guidelines](https://gdpr.eu/)
