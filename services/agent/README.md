# Agent Service

Autonomous AI agent orchestration powered by LangChain.

## Overview

This service handles:
- Autonomous decision orchestration
- Tool usage and action execution
- Memory and context management
- Multi-step reasoning chains

## Structure

```
agent/
├── agents/              # Agent definitions
│   ├── __init__.py
│   ├── decision_agent.py
│   └── research_agent.py
├── tools/               # Tool implementations
│   ├── __init__.py
│   ├── search_tool.py
│   ├── predict_tool.py
│   └── action_tool.py
├── chains/              # LangChain chains
│   ├── __init__.py
│   └── reasoning_chain.py
├── prompts/             # Prompt templates
│   ├── system.txt
│   └── decision.txt
├── memory/              # Memory configurations
│   └── memory_store.py
├── config.py
├── main.py
├── Dockerfile
└── README.md
```

## Agents

| Agent | Purpose | Tools |
|-------|---------|-------|
| `DecisionAgent` | Autonomous decision making | predict, search, action |
| `ResearchAgent` | Information gathering | search, summarize |
| `ReActAgent` | Reasoning & acting | all tools |

## Tools

| Tool | Description | API |
|------|-------------|-----|
| `SearchTool` | Vector similarity search | VectorDB |
| `PredictTool` | ML model predictions | Serving API |
| `ActionTool` | Execute business actions | Internal APIs |
| `SQLTool` | Query data warehouse | Delta Lake |

## Usage

```python
from agents import DecisionAgent

agent = DecisionAgent()

result = agent.run(
    query="Should we approve this transaction?",
    context={
        "transaction_id": "txn_123",
        "amount": 1500,
        "user_id": "user_456"
    }
)

print(result.decision)       # "APPROVE" | "REJECT" | "REVIEW"
print(result.confidence)     # 0.95
print(result.reasoning)      # Step-by-step explanation
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `LLM_MODEL` | Language model | `gpt-4-turbo` |
| `AGENT_MAX_ITERATIONS` | Max reasoning steps | `10` |
| `MEMORY_TYPE` | Memory backend | `redis` |

## Development

```bash
# Run agent locally
poetry run python main.py

# Run with tracing
LANGCHAIN_TRACING_V2=true poetry run python main.py

# Test agent
poetry run pytest tests/test_agents.py
```

## Observability

- **LangSmith** for tracing and debugging
- **Prometheus** metrics for performance
- **Structured logging** for audit trails
