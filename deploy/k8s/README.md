# Kubernetes Deployment

Kubernetes manifests for the Autonomous Enterprise AI Decision System.

> ⚠️ **PLACEHOLDER**: This directory contains Kubernetes templates to be customized for your environment.

## Overview

Deployment manifests for:
- All service deployments
- ConfigMaps and Secrets
- Ingress and Services
- HPA and resource limits
- RBAC and service accounts

## Structure

```
k8s/
├── base/                    # Base configurations
│   ├── namespace.yaml
│   ├── configmaps/
│   ├── secrets/
│   └── rbac/
├── services/
│   ├── data-ingestion/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── kustomization.yaml
│   ├── data-processing/
│   ├── feature-store/
│   ├── ml-platform/
│   ├── serving/
│   ├── vector-db/
│   └── agent/
├── overlays/
│   ├── dev/
│   ├── staging/
│   └── prod/
├── ingress/
│   └── ingress.yaml
└── kustomization.yaml
```

## Prerequisites

- Kubernetes cluster >= 1.28
- kubectl configured
- Kustomize >= 5.0 (or kubectl kustomize)
- Helm 3.x (optional, for Helm charts)

## Deployment

```bash
# Deploy to dev environment
kubectl apply -k overlays/dev

# Deploy to production
kubectl apply -k overlays/prod

# Check deployment status
kubectl get pods -n enterprise-ai

# View logs
kubectl logs -f deployment/serving -n enterprise-ai
```

## Resource Requirements

| Service | CPU Request | Memory Request | Replicas |
|---------|-------------|----------------|----------|
| data-ingestion | 500m | 1Gi | 3 |
| data-processing | 2000m | 4Gi | 2 |
| feature-store | 500m | 2Gi | 2 |
| ml-platform | 1000m | 2Gi | 1 |
| serving | 1000m | 2Gi | 3 |
| vector-db | 500m | 4Gi | 2 |
| agent | 500m | 1Gi | 2 |

## Scaling

```bash
# Manual scaling
kubectl scale deployment serving --replicas=5 -n enterprise-ai

# HPA scaling (auto)
kubectl autoscale deployment serving \
  --min=2 --max=10 --cpu-percent=70 \
  -n enterprise-ai
```

## Monitoring

- **Prometheus**: Metrics collection
- **Grafana**: Dashboards
- **Loki**: Log aggregation
- **Jaeger**: Distributed tracing

## Secrets Management

```bash
# Create secrets from .env file
kubectl create secret generic app-secrets \
  --from-env-file=.env \
  -n enterprise-ai

# Using external secrets operator (recommended)
# Configure ExternalSecret CRDs to pull from Vault/AWS Secrets Manager
```

## Troubleshooting

```bash
# Check pod status
kubectl describe pod <pod-name> -n enterprise-ai

# View events
kubectl get events -n enterprise-ai --sort-by='.lastTimestamp'

# Execute into pod
kubectl exec -it <pod-name> -n enterprise-ai -- /bin/bash
```
