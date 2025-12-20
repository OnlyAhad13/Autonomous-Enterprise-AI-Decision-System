"""
System Prompt for LLM Agent Tool Usage.

Provides the system prompt and tool descriptions for instructing
an LLM to use infrastructure tools safely.
"""

from typing import Dict, List


# ============================================================================
# Tool Descriptions
# ============================================================================


TOOL_DESCRIPTIONS = {
    "airflow": {
        "name": "Airflow",
        "description": "Apache Airflow workflow orchestration",
        "functions": [
            {
                "name": "trigger_dag",
                "description": "Trigger an Airflow DAG run with optional configuration",
                "parameters": {
                    "dag_id": "string (required) - The DAG identifier",
                    "conf": "object (optional) - Configuration to pass to the DAG",
                },
                "returns": "run_id, state, execution_date",
                "requires_confirmation": True,
                "risk_level": "medium",
            },
            {
                "name": "get_dag_status",
                "description": "Get the status of a DAG or specific run",
                "parameters": {
                    "dag_id": "string (required) - The DAG identifier",
                    "run_id": "string (optional) - Specific run ID",
                },
                "returns": "state, execution_date, start_date, end_date",
                "requires_confirmation": False,
                "risk_level": "low",
            },
            {
                "name": "list_dags",
                "description": "List available DAGs",
                "parameters": {
                    "only_active": "boolean (optional) - Filter to active DAGs only",
                },
                "returns": "List of DAG info",
                "requires_confirmation": False,
                "risk_level": "low",
            },
        ],
    },
    "mlflow": {
        "name": "MLflow",
        "description": "MLflow experiment tracking and model registry",
        "functions": [
            {
                "name": "get_latest_run_metrics",
                "description": "Fetch metrics from the latest run of an experiment",
                "parameters": {
                    "experiment_name": "string (required) - Name of the MLflow experiment",
                    "metric_keys": "array (optional) - Specific metrics to retrieve",
                },
                "returns": "run_id, status, metrics, params",
                "requires_confirmation": False,
                "risk_level": "low",
            },
            {
                "name": "register_model",
                "description": "Register a model from a run to the model registry",
                "parameters": {
                    "run_id": "string (required) - The run ID containing the model",
                    "model_name": "string (required) - Name for the registered model",
                    "artifact_path": "string (optional) - Path to model artifact, default 'model'",
                },
                "returns": "model name, version, source",
                "requires_confirmation": True,
                "risk_level": "medium",
            },
            {
                "name": "get_model_versions",
                "description": "List versions of a registered model",
                "parameters": {
                    "model_name": "string (required) - Name of the registered model",
                    "stages": "array (optional) - Filter by stages",
                },
                "returns": "List of model versions with stage and status",
                "requires_confirmation": False,
                "risk_level": "low",
            },
        ],
    },
    "kafka": {
        "name": "Kafka",
        "description": "Apache Kafka cluster monitoring",
        "functions": [
            {
                "name": "get_consumer_lag",
                "description": "Check consumer group lag for topics",
                "parameters": {
                    "group_id": "string (required) - Consumer group ID",
                    "topics": "array (optional) - Topics to check",
                },
                "returns": "total_lag, partition details",
                "requires_confirmation": False,
                "risk_level": "low",
            },
            {
                "name": "get_topic_stats",
                "description": "Get statistics for a Kafka topic",
                "parameters": {
                    "topic": "string (required) - Topic name",
                },
                "returns": "partition_count, total_messages, partition details",
                "requires_confirmation": False,
                "risk_level": "low",
            },
            {
                "name": "list_topics",
                "description": "List all Kafka topics",
                "parameters": {
                    "include_internal": "boolean (optional) - Include internal topics",
                },
                "returns": "List of topic names",
                "requires_confirmation": False,
                "risk_level": "low",
            },
        ],
    },
    "prometheus": {
        "name": "Prometheus",
        "description": "Prometheus metrics and alerting",
        "functions": [
            {
                "name": "query",
                "description": "Execute an instant PromQL query",
                "parameters": {
                    "promql": "string (required) - The PromQL query",
                    "time": "string (optional) - Evaluation timestamp",
                },
                "returns": "Query results with labels and values",
                "requires_confirmation": False,
                "risk_level": "low",
            },
            {
                "name": "query_range",
                "description": "Execute a range PromQL query for time series",
                "parameters": {
                    "promql": "string (required) - The PromQL query",
                    "start": "string (required) - Start time",
                    "end": "string (required) - End time",
                    "step": "string (optional) - Resolution step, default '1m'",
                },
                "returns": "Time series data",
                "requires_confirmation": False,
                "risk_level": "low",
            },
            {
                "name": "get_alerts",
                "description": "Get active Prometheus alerts",
                "parameters": {
                    "state": "string (optional) - Filter by state (firing, pending)",
                },
                "returns": "List of alerts with severity and summary",
                "requires_confirmation": False,
                "risk_level": "low",
            },
        ],
    },
}


# ============================================================================
# System Prompt
# ============================================================================


SYSTEM_PROMPT = """You are an AI assistant for managing enterprise infrastructure. You have access to the following tools:

## Available Tools

### 1. Airflow (Workflow Orchestration)
- `trigger_dag(dag_id, conf)` - Trigger a DAG run ⚠️ REQUIRES CONFIRMATION
- `get_dag_status(dag_id, run_id)` - Check DAG run status
- `list_dags()` - List available DAGs

### 2. MLflow (ML Platform)
- `get_latest_run_metrics(experiment_name)` - Get metrics from latest run
- `register_model(run_id, model_name)` - Register model to registry ⚠️ REQUIRES CONFIRMATION
- `get_model_versions(model_name)` - List model versions

### 3. Kafka (Event Streaming)
- `get_consumer_lag(group_id)` - Check consumer group lag
- `get_topic_stats(topic)` - Get topic statistics
- `list_topics()` - List available topics

### 4. Prometheus (Monitoring)
- `query(promql)` - Execute PromQL query
- `query_range(promql, start, end, step)` - Range query for time series
- `get_alerts()` - Get active alerts

## Safety Guidelines

1. **ALWAYS ASK FOR CONFIRMATION** before executing actions marked with ⚠️
2. **READ-ONLY FIRST**: When investigating issues, start with read-only operations
3. **EXPLAIN YOUR REASONING**: Before calling a tool, explain why you're using it
4. **VERIFY RESULTS**: After tool calls, summarize the results for the user
5. **ESCALATE UNCERTAINTIES**: If unsure, ask the user for clarification

## Response Format

When using tools, structure your response as:

1. **Understanding**: What the user is asking for
2. **Plan**: Which tools you'll use and why
3. **Tool Calls**: Execute the tools (with confirmation if needed)
4. **Summary**: Explain the results

## Confirmation Pattern

For actions requiring confirmation, use this format:

```
I'm about to [action description]. This will:
- [Effect 1]
- [Effect 2]

Do you want me to proceed? (yes/no)
```

Only proceed after explicit user confirmation.
"""


# ============================================================================
# Helper Functions
# ============================================================================


def get_tool_descriptions(tools: List[str] = None) -> str:
    """
    Get formatted tool descriptions.
    
    Args:
        tools: List of tool names to include. If None, includes all.
        
    Returns:
        Formatted string of tool descriptions.
    """
    if tools is None:
        tools = list(TOOL_DESCRIPTIONS.keys())
    
    output = []
    for tool_name in tools:
        if tool_name not in TOOL_DESCRIPTIONS:
            continue
        
        tool = TOOL_DESCRIPTIONS[tool_name]
        output.append(f"## {tool['name']}")
        output.append(f"{tool['description']}\n")
        
        for func in tool["functions"]:
            confirm = " ⚠️" if func.get("requires_confirmation") else ""
            output.append(f"### {func['name']}{confirm}")
            output.append(f"{func['description']}\n")
            output.append("**Parameters:**")
            for param, desc in func["parameters"].items():
                output.append(f"- `{param}`: {desc}")
            output.append(f"\n**Returns:** {func['returns']}\n")
    
    return "\n".join(output)


def get_full_system_prompt(tools: List[str] = None) -> str:
    """
    Get the full system prompt with tool descriptions.
    
    Args:
        tools: List of tool names to include. If None, includes all.
        
    Returns:
        Complete system prompt string.
    """
    return SYSTEM_PROMPT + "\n\n" + get_tool_descriptions(tools)
