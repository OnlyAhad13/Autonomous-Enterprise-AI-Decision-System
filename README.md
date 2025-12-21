<div align="center">

# 🚀 Autonomous Enterprise AI Decision System

[![CI/CD](https://github.com/OnlyAhad13/Autonomous-Enterprise-AI-Decision-System/actions/workflows/deploy.yml/badge.svg)](https://github.com/OnlyAhad13/Autonomous-Enterprise-AI-Decision-System/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Production-grade ML platform with autonomous agent orchestration, real-time inference, and self-healing capabilities.**

---

### 🛠️ Tech Stack

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/></a>
  <a href="https://kafka.apache.org/"><img src="https://img.shields.io/badge/Apache_Kafka-231F20?style=for-the-badge&logo=apache-kafka&logoColor=white" alt="Kafka"/></a>
  <a href="https://spark.apache.org/"><img src="https://img.shields.io/badge/Apache_Spark-E25A1C?style=for-the-badge&logo=apache-spark&logoColor=white" alt="Spark"/></a>
  <a href="https://airflow.apache.org/"><img src="https://img.shields.io/badge/Apache_Airflow-017CEE?style=for-the-badge&logo=apache-airflow&logoColor=white" alt="Airflow"/></a>
</p>

<p align="center">
  <a href="https://mlflow.org/"><img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="https://delta.io/"><img src="https://img.shields.io/badge/Delta_Lake-003366?style=for-the-badge&logo=delta&logoColor=white" alt="Delta Lake"/></a>
  <a href="https://feast.dev/"><img src="https://img.shields.io/badge/Feast-FF6F00?style=for-the-badge&logo=feast&logoColor=white" alt="Feast"/></a>
  <a href="https://openai.com/"><img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI"/></a>
</p>

<p align="center">
  <a href="https://prometheus.io/"><img src="https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white" alt="Prometheus"/></a>
  <a href="https://grafana.com/"><img src="https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=grafana&logoColor=white" alt="Grafana"/></a>
  <a href="https://kubernetes.io/"><img src="https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white" alt="Kubernetes"/></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/></a>
  <a href="https://www.terraform.io/"><img src="https://img.shields.io/badge/Terraform-7B42BC?style=for-the-badge&logo=terraform&logoColor=white" alt="Terraform"/></a>
</p>

<p align="center">
  <a href="https://helm.sh/"><img src="https://img.shields.io/badge/Helm-0F1689?style=for-the-badge&logo=helm&logoColor=white" alt="Helm"/></a>
  <a href="https://aws.amazon.com/"><img src="https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white" alt="AWS"/></a>
  <a href="https://github.com/features/actions"><img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white" alt="GitHub Actions"/></a>
  <a href="https://www.postgresql.org/"><img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL"/></a>
  <a href="https://redis.io/"><img src="https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white" alt="Redis"/></a>
</p>

---

[Quick Start](#-quick-start) • [Architecture](#-architecture) • [Features](#-features) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 📋 Overview

The **Autonomous Enterprise AI Decision System** is a comprehensive MLOps platform designed for production environments at scale. It combines real-time ML inference, autonomous agent-driven operations, and robust data pipelines into a unified system.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| 🤖 **Autonomous Agents** | LLM-orchestrated agents with ReAct reasoning, tool execution, and human-in-the-loop confirmation |
| 📊 **Real-time Inference** | FastAPI prediction service with P99 < 200ms latency at 1000+ RPS |
| 🔄 **Auto-Retraining** | Drift-detection driven retraining with Airflow DAGs and MLflow promotion |
| 📚 **RAG Pipeline** | Vector-based retrieval with FAISS/Milvus for contextual AI responses |
| 🌊 **Stream Processing** | Kafka + Spark Structured Streaming for real-time feature engineering |
| 📈 **Full Observability** | Prometheus metrics, Grafana dashboards, and structured audit logging |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ENTERPRISE AI PLATFORM                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐          │
│  │   Data Sources   │───▶│  Kafka Streams   │───▶│  Spark Streaming │          │
│  │  (APIs, DBs, S3) │    │  (events.raw.v1) │    │  (Transformation) │          │
│  └──────────────────┘    └──────────────────┘    └────────┬─────────┘          │
│                                                            │                     │
│                                                            ▼                     │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐          │
│  │   Feature Store  │◀───│    Delta Lake    │◀───│ Feature Pipeline │          │
│  │     (Feast)      │    │   (Bronze→Gold)  │    │ (Great Expectations)│       │
│  └────────┬─────────┘    └──────────────────┘    └──────────────────┘          │
│           │                                                                      │
│           ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │                        ML SERVICES                                 │          │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │          │
│  │  │   Predict   │  │     RAG     │  │   Explain   │              │          │
│  │  │   Service   │  │   Service   │  │   Service   │              │          │
│  │  │  (FastAPI)  │  │  (FAISS)    │  │   (SHAP)    │              │          │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                                    │                                             │
│                                    ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │                      AUTONOMOUS AGENT LAYER                       │          │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │          │
│  │  │  Agent Core │  │    Tools    │  │  Prompts    │              │          │
│  │  │ (ReAct Loop)│  │(MLflow,Kafka│  │ (Few-shot)  │              │          │
│  │  │             │  │ Airflow,etc)│  │             │              │          │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                                    │                                             │
│                                    ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │                    MLOPS & ORCHESTRATION                          │          │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │          │
│  │  │   MLflow    │  │   Airflow   │  │ Prometheus  │              │          │
│  │  │  (Registry) │  │   (DAGs)    │  │  (Metrics)  │              │          │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🤖 Autonomous Agent System

- **ReAct Reasoning Loop**: Think → Act → Observe cycle with LLM orchestration
- **Tool Integration**: MLflow, Airflow, Kafka, Prometheus, Slack
- **Human-in-the-Loop**: Confirmation protocol for destructive actions
- **Retry Policy**: Exponential backoff with jitter and circuit breaker patterns

```python
from agents.agent_core import AgentCore

agent = AgentCore(llm_client=openai_client)
result = await agent.run_drift_check_and_retrain(
    model_name="forecasting-model",
    drift_threshold=0.1,
)
```

### 📊 Prediction Service

- **High Performance**: P99 latency < 200ms, 1000+ RPS
- **Batch Processing**: CSV upload with async processing
- **Explainability**: SHAP/LIME feature importance
- **Auto-scaling**: HPA with CPU/memory-based scaling

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"age": 35, "income": 75000}}'

# Batch prediction
curl -X POST http://localhost:8000/batch_predict \
  -F "file=@data.csv"
```

### 🔄 Auto-Retraining Pipeline

- **Drift Detection**: Prometheus-based monitoring with configurable thresholds
- **Agent Decision**: LLM evaluates drift and recommends action
- **Conditional Training**: Spark or Python training based on data size
- **Auto-Promotion**: MLflow model staging with validation gates

```python
# Airflow DAG Flow
Drift Sensor → Agent Decision → Branch
                                  ├── Spark Training (large data)
                                  ├── Python Training (small data)
                                  └── Skip (no drift)
                                        ↓
                              Validation → MLflow Promotion → Slack Notification
```

### 📚 RAG Pipeline

- **Document Ingestion**: Markdown, PDF, code file support
- **Vector Store**: FAISS (local) or Milvus (distributed)
- **Semantic Search**: Sentence Transformers embeddings
- **Context Retrieval**: Top-k relevant chunks for LLM context

```python
from services.rag.retriever import RAGRetriever

retriever = RAGRetriever()
response = retriever.query(
    "How does the model handle missing features?",
    top_k=5,
)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- 8GB+ RAM

### Installation

```bash
# Clone repository
git clone https://github.com/OnlyAhad13/Autonomous-Enterprise-AI-Decision-System.git
cd Autonomous-Enterprise-AI-Decision-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -e ".[dev]"
```

### Run Services

```bash
# Start all services (prediction, RAG, monitoring)
docker-compose -f docker-compose.predict.yml up -d
docker-compose -f docker-compose.monitoring.yml up -d

# Or run prediction service locally
cd services/predict
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# Run tests
pytest tests/ -v --tb=short

# Run load tests
pip install locust
locust -f perf/locustfile.py --host=http://localhost:8000
```

---

## 📁 Project Structure

```
├── agents/                    # Autonomous agent system
│   ├── agent_core.py         # ReAct loop orchestration
│   ├── tools/                # MLflow, Kafka, Airflow, Prometheus, Slack
│   └── prompts/              # System prompts, few-shot examples
│
├── services/                  # Microservices
│   ├── predict/              # FastAPI prediction service
│   └── rag/                  # RAG retrieval service
│
├── ingest/                    # Data ingestion
│   └── dags/                 # Airflow DAGs (auto-retrain)
│
├── spark_jobs/               # Spark streaming & batch
│   └── streaming_to_delta.py
│
├── features/                  # Feature engineering
│   ├── feature_store.py      # Feast integration
│   └── transformers.py
│
├── models/                    # Model training
│   ├── train_forecast.py     # Prophet, LSTM, ETS
│   └── optuna_study.py       # Hyperparameter tuning
│
├── mlflow_utils/             # MLflow utilities
│   ├── auto_promote.py       # Model promotion
│   └── cli.py                # CLI tools
│
├── conf/                      # Configuration
│   ├── prometheus.yml        # Prometheus scrape config
│   ├── alerting_rules.yml    # Alert rules
│   └── alertmanager.yml      # Alertmanager routing
│
├── deploy/                    # Deployment
│   └── helm/                 # Helm charts (predict, rag)
│
├── infra/                     # Infrastructure
│   └── terraform/            # EKS, MSK, RDS
│
├── tests/                     # Test suites
│   ├── e2e/                  # End-to-end tests
│   └── chaos/                # Chaos engineering
│
├── perf/                      # Performance testing
│   └── locustfile.py         # Load tests
│
├── docs/                      # Documentation
│   ├── SECURITY.md           # Security guide
│   └── COST_ESTIMATE.md      # AWS/GCP costs
│
└── notebooks/                 # Jupyter notebooks
    ├── 01_EDA.ipynb
    ├── 02_baselines.ipynb
    └── 04_rag_demo.ipynb
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5000` |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka brokers | `localhost:9092` |
| `OPENAI_API_KEY` | OpenAI API key | Required for agents |
| `SLACK_WEBHOOK_URL` | Slack notifications | Optional |
| `MODEL_NAME` | Model to serve | `forecasting-model` |

### Agent Policy (`conf/agent_policy.json`)

```json
{
  "drift": {
    "threshold": 0.1,
    "metric_name": "model_drift_score"
  },
  "actions": {
    "allowed": ["alert", "retrain", "promote", "rollback"],
    "require_confirmation": ["rollback"]
  },
  "safety": {
    "max_retrains_per_day": 3,
    "dry_run_mode": false
  }
}
```

---

## 📈 Observability

### Metrics

Access Grafana dashboards at `http://localhost:3000` (admin/admin)

| Dashboard | Metrics |
|-----------|---------|
| ML Platform | P50/P95/P99 latency, error rate, throughput |
| Model Drift | Drift score, feature distributions |
| Agent Actions | Action counts, execution time |
| Infrastructure | CPU, memory, Kafka lag |

### Alerting

| Alert | Threshold | Severity |
|-------|-----------|----------|
| HighPredictionLatency | P99 > 500ms | Warning |
| ModelDriftDetected | drift > 0.1 | Warning |
| CriticalKafkaLag | lag > 100k | Critical |

---

## 🧪 Testing

```bash
# Unit tests
pytest tests/ -v --cov=. --cov-report=html

# E2E tests (requires Docker)
docker-compose -f tests/e2e/docker-compose.e2e.yml up -d
E2E_MODE=true pytest tests/e2e/ -v

# Chaos tests
pytest tests/chaos/ -v

# Load tests
locust -f perf/locustfile.py --host=http://localhost:8000 \
  --users=100 --spawn-rate=10 --run-time=5m --headless
```

---

## 🚢 Deployment

### Kubernetes (Helm)

```bash
# Deploy prediction service
helm upgrade --install predict ./deploy/helm/predict \
  --namespace ml-services \
  --set image.tag=latest \
  --set replicaCount=3

# Deploy RAG service
helm upgrade --install rag ./deploy/helm/rag \
  --namespace ml-services
```

### Terraform (AWS)

```bash
cd infra/terraform
terraform init
terraform plan -var="environment=production"
terraform apply
```

### CI/CD (GitHub Actions)

The pipeline automatically:
1. Runs unit tests
2. Builds Docker images
3. Pushes to GHCR
4. Deploys to staging
5. Deploys to production (with approval)

---

## 💰 Cost Estimates

| Environment | AWS | GCP |
|-------------|-----|-----|
| Staging | ~$320/month | ~$305/month |
| Production | ~$1,910/month | ~$1,590/month |

See [docs/COST_ESTIMATE.md](docs/COST_ESTIMATE.md) for detailed breakdown.

---

## 🔒 Security

- **Encryption**: TLS 1.3 in transit, AES-256 at rest (KMS)
- **RBAC**: Feature store, MLflow, Kubernetes access control
- **PII Masking**: Pseudonymization in pipelines
- **Audit Logging**: Structured logs for all agent actions

See [docs/SECURITY.md](docs/SECURITY.md) for security guidelines.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [SECURITY.md](docs/SECURITY.md) | Security architecture |
| [COST_ESTIMATE.md](docs/COST_ESTIMATE.md) | Infrastructure costs |
| [infra/README.md](infra/README.md) | Operator guide |
| [conf/MONITORING.md](conf/MONITORING.md) | Observability setup |

---

## 🛠 Development

### Setup

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run formatters
black .
isort .

# Run linters
flake8 .
mypy .
```

### Adding a New Tool

```python
# agents/tools/tool_custom.py
from agents.tools.base import BaseTool, ToolResult

class CustomTool(BaseTool):
    def execute(self, **kwargs) -> ToolResult:
        # Implementation
        return ToolResult(success=True, data=result)
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards

- **Style**: Black (line length 100), isort
- **Types**: Full type annotations
- **Tests**: >80% coverage required
- **Docs**: Docstrings for public APIs

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- [MLflow](https://mlflow.org/) - Model registry and tracking
- [Apache Kafka](https://kafka.apache.org/) - Event streaming
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
- [Prometheus](https://prometheus.io/) - Monitoring

---

<div align="center">

**Built with ❤️ for production ML at scale**

[⬆ Back to Top](#-autonomous-enterprise-ai-decision-system)

</div>
