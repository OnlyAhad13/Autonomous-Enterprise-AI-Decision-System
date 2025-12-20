"""
Few-Shot Examples for LLM Agent Tool Usage.

Provides example conversations demonstrating safe tool usage patterns.
"""

from typing import Dict, List, Optional


# ============================================================================
# Few-Shot Examples
# ============================================================================


FEW_SHOT_EXAMPLES = {
    "airflow": [
        {
            "description": "Checking DAG status (read-only)",
            "user": "What's the status of the ETL pipeline?",
            "assistant": """I'll check the status of the ETL pipeline.

**Understanding**: You want to know the current state of the ETL pipeline DAG.

**Plan**: I'll use `get_dag_status` to fetch the latest run status.

**Tool Call**:
```json
{
    "tool": "airflow.get_dag_status",
    "parameters": {"dag_id": "etl_pipeline"}
}
```

**Result**:
- DAG: etl_pipeline
- State: success
- Last Run: 2024-01-15T08:00:00Z
- Duration: 45 minutes

The ETL pipeline completed successfully in its latest run.""",
        },
        {
            "description": "Triggering a DAG (requires confirmation)",
            "user": "Can you run the data backfill pipeline for yesterday?",
            "assistant": """**Understanding**: You want to trigger the data backfill pipeline with yesterday's date.

**Plan**: I'll use `trigger_dag` with a date configuration. This requires your confirmation first.

⚠️ **Confirmation Required**

I'm about to trigger the `data_backfill` DAG. This will:
- Start a new DAG run immediately
- Process data for 2024-01-14
- May take 2-3 hours to complete

Do you want me to proceed? (yes/no)""",
        },
        {
            "description": "After user confirms DAG trigger",
            "user": "yes, go ahead",
            "assistant": """Proceeding with the DAG trigger.

**Tool Call**:
```json
{
    "tool": "airflow.trigger_dag",
    "parameters": {
        "dag_id": "data_backfill",
        "conf": {"date": "2024-01-14"}
    }
}
```

**Result**:
- Run ID: manual__2024-01-15T10:30:00+00:00
- State: running
- Triggered: 2024-01-15T10:30:00Z

✓ The data backfill pipeline has been triggered for 2024-01-14. You can check its progress with "What's the status of data_backfill?".""",
        },
    ],
    "mlflow": [
        {
            "description": "Checking experiment metrics",
            "user": "What were the metrics from the latest training run?",
            "assistant": """I'll fetch the metrics from the latest training run.

**Understanding**: You want to see the performance metrics from the most recent model training.

**Plan**: I'll use `get_latest_run_metrics` to retrieve the metrics.

**Tool Call**:
```json
{
    "tool": "mlflow.get_latest_run_metrics",
    "parameters": {"experiment_name": "forecasting-pipeline"}
}
```

**Result**:
- Run ID: a1b2c3d4
- Status: FINISHED
- Metrics:
  - prophet_mape: 8.5%
  - prophet_rmse: 1234.56
  - training_time: 120.5s

The latest model achieved a MAPE of 8.5%, which is within the target threshold of 10%.""",
        },
        {
            "description": "Registering a model (requires confirmation)",
            "user": "Register the model from that run as churn-predictor",
            "assistant": """**Understanding**: You want to register the model from run a1b2c3d4 to the model registry as "churn-predictor".

**Plan**: I'll use `register_model`. This requires your confirmation first.

⚠️ **Confirmation Required**

I'm about to register a new model version. This will:
- Create model "churn-predictor" if it doesn't exist
- Add a new version from run a1b2c3d4
- The model will be in "None" stage initially

Do you want me to proceed? (yes/no)""",
        },
    ],
    "kafka": [
        {
            "description": "Checking consumer lag",
            "user": "Is there any lag in the events consumer?",
            "assistant": """I'll check the consumer lag for the events consumer group.

**Understanding**: You want to know if the events consumer is falling behind.

**Plan**: I'll use `get_consumer_lag` to check the lag.

**Tool Call**:
```json
{
    "tool": "kafka.get_consumer_lag",
    "parameters": {"group_id": "events-processor"}
}
```

**Result**:
- Consumer Group: events-processor
- Total Lag: 1,500 messages
- Partitions:
  - events.raw.v1 [0]: 500 messages
  - events.raw.v1 [1]: 500 messages
  - events.raw.v1 [2]: 500 messages

⚠️ There is moderate lag (1,500 messages). This could indicate:
1. Increased event volume
2. Slower processing time
3. Consumer scaling needed

Would you like me to check the Prometheus metrics for more details?""",
        },
    ],
    "prometheus": [
        {
            "description": "Checking system alerts",
            "user": "Are there any active alerts?",
            "assistant": """I'll check for active Prometheus alerts.

**Understanding**: You want to see if there are any firing or pending alerts.

**Plan**: I'll use `get_alerts` to fetch active alerts.

**Tool Call**:
```json
{
    "tool": "prometheus.get_alerts",
    "parameters": {"state": "firing"}
}
```

**Result**:
- Firing Alerts: 2
- Pending Alerts: 1

**Active Alerts**:
1. 🔴 **HighCPUUsage** (critical)
   - Instance: worker-03
   - Summary: CPU usage above 90% for 5 minutes

2. 🟡 **KafkaConsumerLag** (warning)
   - Consumer: events-processor
   - Summary: Consumer lag exceeds 1000 messages

There are 2 firing alerts. The high CPU on worker-03 is critical and may need immediate attention. Would you like me to investigate the CPU metrics further?""",
        },
        {
            "description": "Querying specific metrics",
            "user": "What's the current request rate for the API?",
            "assistant": """I'll query the request rate metrics.

**Understanding**: You want to see the current API request throughput.

**Plan**: I'll use `query` with a PromQL expression for request rate.

**Tool Call**:
```json
{
    "tool": "prometheus.query",
    "parameters": {"promql": "sum(rate(http_requests_total{job='api'}[5m]))"}
}
```

**Result**:
- Query: `sum(rate(http_requests_total{job='api'}[5m]))`
- Value: 245.67

The API is currently handling approximately **246 requests per second** (averaged over 5 minutes). This is within normal operating range (typical: 200-400 rps).""",
        },
    ],
}


# ============================================================================
# Helper Functions
# ============================================================================


def get_examples_for_tool(tool_name: str) -> List[Dict]:
    """
    Get few-shot examples for a specific tool.
    
    Args:
        tool_name: Name of the tool (airflow, mlflow, kafka, prometheus).
        
    Returns:
        List of example conversations.
    """
    return FEW_SHOT_EXAMPLES.get(tool_name, [])


def format_examples_as_conversation(tool_name: str) -> str:
    """
    Format examples as a conversation string for prompts.
    
    Args:
        tool_name: Name of the tool.
        
    Returns:
        Formatted conversation examples.
    """
    examples = get_examples_for_tool(tool_name)
    if not examples:
        return ""
    
    output = [f"## {tool_name.title()} Tool Examples\n"]
    
    for i, example in enumerate(examples, 1):
        output.append(f"### Example {i}: {example['description']}\n")
        output.append(f"**User**: {example['user']}\n")
        output.append(f"**Assistant**: {example['assistant']}\n")
        output.append("---\n")
    
    return "\n".join(output)


def get_all_examples() -> str:
    """
    Get all few-shot examples formatted for a prompt.
    
    Returns:
        All examples as a formatted string.
    """
    output = ["# Few-Shot Examples for Tool Usage\n"]
    
    for tool_name in FEW_SHOT_EXAMPLES:
        output.append(format_examples_as_conversation(tool_name))
    
    return "\n".join(output)


def get_confirmation_examples() -> str:
    """
    Get examples specifically for confirmation patterns.
    
    Returns:
        Examples showing confirmation patterns.
    """
    return """## Confirmation Pattern Examples

### When to Confirm

Always ask for confirmation before:
- Triggering DAG runs
- Registering models
- Transitioning model stages
- Any action that modifies state

### Confirmation Format

```
⚠️ **Confirmation Required**

I'm about to [ACTION]. This will:
- [Effect 1]
- [Effect 2]

[Optional: Any risks or considerations]

Do you want me to proceed? (yes/no)
```

### After User Confirms

```
Proceeding with [ACTION].

[Tool Call]

**Result**:
[Summary of what happened]

✓ [Confirmation message]
```

### If User Declines

```
Understood. I won't [ACTION]. Let me know if you'd like to do something else instead.
```
"""
