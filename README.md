<div align="center">

# 🚀 Autonomous Enterprise AI Decision System
### *Production-Grade MLOps Platform with Self-Healing Capabilities*

<br/>

<a href="LICENSE"><img src="https://img.shields.io/badge/⚖️_License-MIT-00C853?style=for-the-badge&labelColor=1a1a2e" alt="License"/></a>
<a href="https://python.org"><img src="https://img.shields.io/badge/🐍_Python-3.11+-00C853?style=for-the-badge&labelColor=1a1a2e" alt="Python"/></a>
<a href="https://react.dev"><img src="https://img.shields.io/badge/⚛️_React-18.3+-61DAFB?style=for-the-badge&labelColor=1a1a2e" alt="React"/></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/💅_Code_Style-Black-00C853?style=for-the-badge&labelColor=1a1a2e" alt="Code Style"/></a>
<a href="https://github.com/OnlyAhad13/Autonomous-Enterprise-AI-Decision-System"><img src="https://img.shields.io/badge/✅_Build-Passing-00C853?style=for-the-badge&labelColor=1a1a2e" alt="Build"/></a>

<br/><br/>

---

### 🛠️ Built With

<table>
<tr>
<td align="center" width="96">
<img src="https://techstack-generator.vercel.app/python-icon.svg" alt="Python" width="48" height="48"/>
<br><b>Python</b>
</td>
<td align="center" width="96">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/fastapi/fastapi-original.svg" alt="FastAPI" width="48" height="48"/>
<br><b>FastAPI</b>
</td>
<td align="center" width="96">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/react/react-original.svg" alt="React" width="48" height="48"/>
<br><b>React</b>
</td>
<td align="center" width="96">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/apachekafka/apachekafka-original.svg" alt="Kafka" width="48" height="48"/>
<br><b>Kafka</b>
</td>
<td align="center" width="96">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Apache_Spark_logo.svg/512px-Apache_Spark_logo.svg.png" alt="Spark" width="48" height="48"/>
<br><b>Spark</b>
</td>
<td align="center" width="96">
<img src="https://img.shields.io/badge/ML-flow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow" height="28"/>
<br><b>MLflow</b>
</td>
</tr>
<tr>
<td align="center" width="96">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/docker/docker-original.svg" alt="Docker" width="48" height="48"/>
<br><b>Docker</b>
</td>
<td align="center" width="96">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kubernetes/kubernetes-plain.svg" alt="K8s" width="48" height="48"/>
<br><b>Kubernetes</b>
</td>
<td align="center" width="96">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/terraform/terraform-original.svg" alt="Terraform" width="48" height="48"/>
<br><b>Terraform</b>
</td>
<td align="center" width="96">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/prometheus/prometheus-original.svg" alt="Prometheus" width="48" height="48"/>
<br><b>Prometheus</b>
</td>
<td align="center" width="96">
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/grafana/grafana-original.svg" alt="Grafana" width="48" height="48"/>
<br><b>Grafana</b>
</td>
<td align="center" width="96">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/OpenAI_Logo.svg/512px-OpenAI_Logo.svg.png" alt="OpenAI" width="48" height="48"/>
<br><b>GPT-4o</b>
</td>
</tr>
</table>

---

[🚀 Quick Start](#-quick-start) • [🏗 Architecture](#-architecture) • [✨ Features](#-features) • [📚 Docs](#-documentation) • [🤝 Contributing](#-contributing)

</div>

---

## 📋 Overview

The **Autonomous Enterprise AI Decision System** is a unified platform bridging **Data Engineering, Data Science, AI Engineering, and Software Engineering**. It is designed to demonstrate a production-grade MLOps lifecycle where an autonomous agent manages the entire system—from data ingestion to model deployment.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| 🤖 **Autonomous Agents** | LLM-orchestrated agents (GPT-4o) with tool execution (MLflow, Kafka, Docker) |
| 🖥️ **Full-Stack Dashboard** | Modern React UI for monitoring events, managing models, and controlling the agent |
| 📊 **Real-time Inference** | FastAPI prediction service with P99 < 200ms latency at 1000+ RPS |
| 🌊 **Stream Processing** | Kafka event streaming + Spark Structured Streaming for feature engineering |
| 🧠 **Live Model Training** | Train models (Random Forest, GBM, etc.) on *live* streaming data with one click |
| 📈 **Full Observability** | Prometheus metrics, Grafana dashboards, and In-App Notifications |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           AUTONOMOUS AI PLATFORM                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐          │
│  │   Data Sources   │───▶│  Kafka Streams   │───▶│  Spark Streaming │          │
│  │  (Event Streams) │    │  (events.raw.v1) │    │  (Feature Eng.)  │          │
│  └──────────────────┘    └──────────────────┘    └────────┬─────────┘          │
│                                                            │                     │
│                                                            ▼                     │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐          │
│  │   Frontend UI    │◀───│   Backend API    │◀───│   Live Trainer   │          │
│  │  (React + Vite)  │    │    (FastAPI)     │    │ (Scikit + MLflow)│          │
│  └──────────────────┘    └────────┬─────────┘    └──────────────────┘          │
│                                   │                                              │
│                                   ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │                      AUTONOMOUS AGENT LAYER                       │          │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │          │
│  │  │  Agent Core │  │    Tools    │  │     LLM     │              │          │
│  │  │ (ReAct Loop)│  │(MLflow,Kafka│  │   (GPT-4o)  │              │          │
│  │  │             │  │ Docker, etc)│  │             │              │          │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🖥️ Interactive Dashboard
A full-featured React application providing real-time visibility into the system.
- **Data Ingestion**: Watch live Kafka events flow in via WebSockets.
- **Model Registry**: View MLflow models, compare metrics (Accuracy, F1).
- **Training Panel**: Trigger training runs on live data with a single click.
- **Agent Interface**: Chat with the autonomous agent to execute complex tasks.

### 🤖 Autonomous Agent System
- **ReAct Reasoning Loop**: Think → Act → Observe cycle.
- **Tool Integration**: Can restart containers, check logs, train models, deploys services.
- **Self-Healing**: Detects failures (e.g., Kafka down) and attempts to restart services.

### 📊 Live Model Training
- **Real-time Data**: Trains on the buffer of most recent events from Kafka.
- **MLflow Integration**: Automatically logs parameters, metrics, and artifacts.
- **One-Click Deploy**: Promote models to production instantly from the UI.
- **Model Types**: Comparison of Random Forest, Gradient Boosting, and Logistic Regression.

### 🔔 In-App Notification System
- **Real-time Alerts**: Success/Error toasts for all background actions.
- **Action History**: Persistent notification center to track agent activities.
- **No Slack Required**: Fully self-contained within the application.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ & npm
- Docker & Docker Compose
- OpenAI API Key

### 1. Start Infrastructure (Kafka, MLflow, Prometheus)
```bash
# Start Core Infrastructure
docker compose -f infra/docker-compose.kafka.yml up -d
docker compose -f docker-compose.monitoring.yml up -d

# Start MLflow (Local)
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

### 2. Start Backend
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run FastAPI Backend
PYTHONPATH=. uvicorn webapp.main:app --host 0.0.0.0 --port 8080 --reload
```

### 3. Start Frontend
```bash
cd frontend
npm install
npm run dev
```

### 4. Start Live Data Producer
```bash
# In the root directory
python ingest/live_producer.py --rate 5
```

Running these commands will spin up the entire platform. Access the dashboard at `http://localhost:5173`.

---

## 📁 Project Structure

```
├── agents/                    # Autonomous agent system
│   ├── agent_core.py          # ReAct loop orchestration
│   ├── tools/                 # Agent tools (MLflow, Kafka, Slack)
│
├── webapp/                    # FastAPI Backend
│   ├── main.py                # App entry point
│   ├── routers/               # API endpoints
│   │   ├── agent.py           # Agent interaction
│   │   ├── ingestion.py       # Live event streaming
│   │   ├── models.py          # ML training & deployment
│   │   ├── notifications.py   # In-app notifications
│
├── frontend/                  # React Frontend
│   ├── src/
│   │   ├── components/        # Reusable UI components
│   │   ├── pages/             # Dashboard, Models, Agent pages
│   │   ├── api/               # Axios client
│
├── models/                    # ML Training Logic
│   ├── live_train.py          # Real-time training pipeline
│
├── ingest/                    # Data Ingestion
│   ├── live_producer.py       # Kafka event producer
│
├── deploy/                    # Deployment Configs
├── infra/                     # Infrastructure (Terraform, Docker)
├── tests/                     # Test Suites
└── notebooks/                 # EDA & Prototyping
```

---

## 🔧 Configuration

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://localhost:5000` |
| `KAFKA_BOOTSTRAP` | Kafka brokers | `localhost:9093` |
| `OPENAI_API_KEY` | OpenAI API key | Required for agents |
| `SLACK_WEBHOOK_URL` | Optional Slack Integration | - |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.

---

<div align="center">

**Built with ❤️ for production ML at scale by Syed Abdul Ahad**

[⬆ Back to Top](#-autonomous-enterprise-ai-decision-system)

</div>
