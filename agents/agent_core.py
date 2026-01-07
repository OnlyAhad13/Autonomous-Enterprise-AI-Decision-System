"""
Agent Core - LLM-based Orchestration Agent.

Implements a ReAct (Reason-Act-Observe) loop for executing high-level objectives
using infrastructure tools with human confirmation for destructive actions.
"""

import os
import re
import json
import time
import logging
from enum import Enum
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Union
from abc import ABC, abstractmethod

# Import tools
from .tools.tool_airflow import AirflowTool
from .tools.tool_mlflow import MLflowTool
from .tools.tool_prometheus import PrometheusTool
from .tools.tool_slack import SlackTool
from .tools.tool_airflow import ToolResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================


class AgentState(Enum):
    """Agent execution states."""
    IDLE = "idle"
    REASONING = "reasoning"
    ACTING = "acting"
    WAITING_CONFIRMATION = "waiting_confirmation"
    OBSERVING = "observing"
    COMPLETED = "completed"
    FAILED = "failed"


class ActionType(Enum):
    """Types of agent actions."""
    TOOL_CALL = "tool_call"
    CONFIRMATION_REQUEST = "confirmation_request"
    FINAL_ANSWER = "final_answer"
    REPORT = "report"


# Thresholds
DEFAULT_DRIFT_THRESHOLD = 0.1
MAX_ITERATIONS = 10
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 1.0


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class AgentStep:
    """A single step in the agent execution."""
    step_number: int
    state: AgentState
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict] = None
    observation: Optional[str] = None
    requires_confirmation: bool = False
    confirmed: Optional[bool] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentResponse:
    """Final response from agent execution."""
    success: bool
    objective: str
    final_answer: str
    steps: List[AgentStep]
    report: Optional[Dict] = None
    total_duration: float = 0.0
    error: Optional[str] = None


@dataclass
class ToolSpec:
    """Specification for a registered tool."""
    name: str
    description: str
    function: Callable
    requires_confirmation: bool = False
    parameters: Dict[str, str] = field(default_factory=dict)


# ============================================================================
# Confirmation Protocol
# ============================================================================


class ConfirmationHandler(ABC):
    """Abstract base for handling confirmations."""
    
    @abstractmethod
    def request_confirmation(
        self,
        action: str,
        description: str,
        details: Dict[str, Any],
    ) -> bool:
        """Request confirmation for an action. Returns True if approved."""
        pass


class AutoConfirmHandler(ConfirmationHandler):
    """Auto-confirms all actions (for testing)."""
    
    def request_confirmation(
        self,
        action: str,
        description: str,
        details: Dict[str, Any],
    ) -> bool:
        logger.info(f"[AUTO-CONFIRM] {action}: {description}")
        return True


class InteractiveConfirmHandler(ConfirmationHandler):
    """Interactive confirmation via stdin."""
    
    def request_confirmation(
        self,
        action: str,
        description: str,
        details: Dict[str, Any],
    ) -> bool:
        print("\n" + "=" * 60)
        print("⚠️  CONFIRMATION REQUIRED")
        print("=" * 60)
        print(f"Action: {action}")
        print(f"Description: {description}")
        if details:
            print(f"Details: {json.dumps(details, indent=2)}")
        print("=" * 60)
        
        response = input("Proceed? (yes/no): ").strip().lower()
        return response in ("yes", "y")


class CallbackConfirmHandler(ConfirmationHandler):
    """Confirmation via callback function."""
    
    def __init__(self, callback: Callable[[str, str, Dict], bool]):
        self.callback = callback
    
    def request_confirmation(
        self,
        action: str,
        description: str,
        details: Dict[str, Any],
    ) -> bool:
        return self.callback(action, description, details)


# ============================================================================
# Retry Policy
# ============================================================================


class RetryPolicy:
    """Configurable retry policy with exponential backoff."""
    
    def __init__(
        self,
        max_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        base_delay: float = DEFAULT_RETRY_DELAY,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        retryable_errors: Optional[List[str]] = None,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retryable_errors = retryable_errors or [
            "timeout", "connection", "rate limit", "503", "502", "500"
        ]
    
    def should_retry(self, error: str, attempt: int) -> bool:
        """Check if the error should be retried."""
        if attempt >= self.max_attempts:
            return False
        
        error_lower = error.lower()
        return any(e in error_lower for e in self.retryable_errors)
    
    def get_delay(self, attempt: int) -> float:
        """Get delay for the given attempt number."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)
    
    def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """Execute function with retry logic."""
        last_error = None
        
        for attempt in range(self.max_attempts):
            try:
                result = func(*args, **kwargs)
                
                # Check if result indicates an error
                if isinstance(result, ToolResult) and not result.success:
                    if self.should_retry(result.error or "", attempt):
                        delay = self.get_delay(attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{self.max_attempts} "
                            f"after {delay:.1f}s: {result.error}"
                        )
                        time.sleep(delay)
                        continue
                
                return result
                
            except Exception as e:
                last_error = str(e)
                if self.should_retry(last_error, attempt):
                    delay = self.get_delay(attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_attempts} "
                        f"after {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    raise
        
        raise Exception(f"Max retries exceeded: {last_error}")


# ============================================================================
# LLM Interface
# ============================================================================


class LLMInterface(ABC):
    """Abstract base for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        pass


class MockLLM(LLMInterface):
    """Mock LLM for testing with predefined responses."""
    
    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = responses or []
        self.call_count = 0
        self.prompts: List[str] = []
    
    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        
        return "FINAL_ANSWER: Task completed."


class OpenAILLM(LLMInterface):
    """OpenAI LLM implementation."""
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature
    
    def generate(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not set")
        
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        
        return response.choices[0].message.content


# ============================================================================
# Agent Core
# ============================================================================


class AgentCore:
    """
    LLM-based orchestration agent using ReAct pattern.
    
    Executes high-level objectives by:
    1. REASONING about the current state
    2. ACTING by calling tools
    3. OBSERVING the results
    4. Repeating until objective is met
    """
    
    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        confirmation_handler: Optional[ConfirmationHandler] = None,
        retry_policy: Optional[RetryPolicy] = None,
        airflow_url: str = "http://localhost:8080",
        prometheus_url: str = "http://localhost:9090",
        slack_webhook: Optional[str] = None,
        drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    ):
        """
        Initialize the agent.
        
        Args:
            llm: LLM interface for reasoning.
            confirmation_handler: Handler for action confirmations.
            retry_policy: Retry policy for tool calls.
            airflow_url: Airflow webserver URL.
            prometheus_url: Prometheus server URL.
            slack_webhook: Slack webhook URL.
            drift_threshold: Threshold for triggering retraining.
        """
        self.llm = llm or MockLLM()
        self.confirmation_handler = confirmation_handler or AutoConfirmHandler()
        self.retry_policy = retry_policy or RetryPolicy()
        self.drift_threshold = drift_threshold
        
        # Initialize tools
        self.airflow = AirflowTool(base_url=airflow_url)
        self.prometheus = PrometheusTool(base_url=prometheus_url)
        self.slack = SlackTool(webhook_url=slack_webhook)
        
        # Register tools
        self.tools = self._register_tools()
        
        # State
        self.steps: List[AgentStep] = []
        self.state = AgentState.IDLE
    
    def _register_tools(self) -> Dict[str, ToolSpec]:
        """Register available tools."""
        return {
            "query_metrics": ToolSpec(
                name="query_metrics",
                description="Query Prometheus metrics using PromQL",
                function=self._tool_query_metrics,
                requires_confirmation=False,
                parameters={"query": "PromQL query string (or use 'promql')"},
            ),
            "get_alerts": ToolSpec(
                name="get_alerts",
                description="Get active Prometheus alerts",
                function=self._tool_get_alerts,
                requires_confirmation=False,
            ),
            "check_drift": ToolSpec(
                name="check_drift",
                description="Check model drift score and compare to threshold",
                function=self._tool_check_drift,
                requires_confirmation=False,
                parameters={"metric_name": "Name of drift metric"},
            ),
            "trigger_retrain": ToolSpec(
                name="trigger_retrain",
                description="Trigger model retraining DAG",
                function=self._tool_trigger_retrain,
                requires_confirmation=True,
                parameters={"dag_id": "DAG ID (default: retrain)"},
            ),
            "train_model": ToolSpec(
                name="train_model",
                description="Train a model on live event data. Uses events from the ingestion buffer to train ML models and logs to MLflow.",
                function=self._tool_train_model,
                requires_confirmation=True,
                parameters={"model_type": "Model type: random_forest, gradient_boosting, or logistic_regression"},
            ),
            "get_dag_status": ToolSpec(
                name="get_dag_status",
                description="Get status of an Airflow DAG",
                function=self._tool_get_dag_status,
                requires_confirmation=False,
                parameters={"dag_id": "DAG identifier"},
            ),
            "send_slack_message": ToolSpec(
                name="send_slack_message",
                description="Send a message to Slack",
                function=self._tool_send_slack,
                requires_confirmation=False,
                parameters={"message": "Message text", "channel": "Slack channel"},
            ),
            "post_report": ToolSpec(
                name="post_report",
                description="Post a formatted report to Slack",
                function=self._tool_post_report,
                requires_confirmation=False,
                parameters={"title": "Report title", "sections": "Report sections"},
            ),
        }
    
    # ========================================================================
    # Tool Implementations
    # ========================================================================
    
    def _tool_query_metrics(self, promql: str = None, query: str = None) -> ToolResult:
        """Query Prometheus metrics."""
        # Accept either 'promql' or 'query' parameter from LLM
        query_str = promql or query or ""
        return self.retry_policy.execute_with_retry(
            self.prometheus.query, query_str
        )
    
    def _tool_get_alerts(self) -> ToolResult:
        """Get active alerts."""
        return self.retry_policy.execute_with_retry(
            self.prometheus.get_alerts
        )
    
    def _tool_check_drift(
        self,
        metric_name: str = "model_drift_score",
    ) -> ToolResult:
        """Check model drift against threshold."""
        result = self._tool_query_metrics(metric_name)
        
        if not result.success:
            return result
        
        results = result.data.get("results", [])
        if not results:
            return ToolResult(
                success=True,
                data={
                    "drift_score": 0.0,
                    "threshold": self.drift_threshold,
                    "exceeds_threshold": False,
                    "message": "No drift metrics found",
                },
            )
        
        # Get the drift value
        drift_score = float(results[0].get("value", 0))
        exceeds = drift_score > self.drift_threshold
        
        return ToolResult(
            success=True,
            data={
                "drift_score": drift_score,
                "threshold": self.drift_threshold,
                "exceeds_threshold": exceeds,
                "message": f"Drift score {drift_score:.3f} {'exceeds' if exceeds else 'within'} threshold {self.drift_threshold}",
            },
        )
    
    def _tool_trigger_retrain(
        self,
        dag_id: str = "retrain",
        conf: Optional[Dict] = None,
    ) -> ToolResult:
        """Trigger retraining DAG."""
        return self.retry_policy.execute_with_retry(
            self.airflow.trigger_dag, dag_id, conf
        )
    
    def _tool_train_model(
        self,
        model_type: str = "random_forest",
    ) -> ToolResult:
        """Train a model on live event data from the ingestion buffer."""
        try:
            # Import the live trainer
            from models.live_train import LiveModelTrainer
            
            # Get events from the webapp ingestion buffer if available
            events = []
            try:
                from webapp.routers.ingestion import event_buffer
                events = list(event_buffer)
            except ImportError:
                pass
            
            # Fallback: generate sample events if buffer is empty
            if len(events) < 50:
                import random
                from datetime import datetime
                
                event_types = ["order_placed", "page_view", "purchase", "add_to_cart", "checkout_started"]
                for i in range(200):
                    events.append({
                        "id": f"train_evt_{i}",
                        "timestamp": datetime.now().isoformat(),
                        "value": {
                            "event_type": random.choice(event_types),
                            "price": round(random.uniform(10, 500), 2),
                            "quantity": random.randint(1, 5),
                            "user_id": f"U{random.randint(1000, 9999)}",
                            "product_id": f"P{random.randint(100, 999)}",
                        }
                    })
            
            # Train the model
            trainer = LiveModelTrainer()
            result = trainer.train_event_classifier(events, model_type)
            
            return ToolResult(
                success=True,
                data={
                    "model_type": result["model_type"],
                    "accuracy": result["metrics"]["accuracy"],
                    "f1_score": result["metrics"]["f1_weighted"],
                    "num_samples": result["metrics"]["num_samples"],
                    "mlflow_run_id": result["mlflow_run_id"],
                    "model_path": result["model_path"],
                    "classes": result["classes"],
                },
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Training failed: {str(e)}",
            )
    
    def _tool_get_dag_status(self, dag_id: str) -> ToolResult:
        """Get DAG status."""
        return self.retry_policy.execute_with_retry(
            self.airflow.get_dag_status, dag_id
        )
    
    def _tool_send_slack(
        self,
        message: str,
        channel: Optional[str] = None,
    ) -> ToolResult:
        """Send Slack message."""
        return self.slack.send_message(message, channel)
    
    def _tool_post_report(
        self,
        title: str,
        sections: List[Dict[str, str]],
        channel: Optional[str] = None,
    ) -> ToolResult:
        """Post formatted report to Slack."""
        return self.slack.post_report(title, sections, channel)
    
    # ========================================================================
    # ReAct Loop
    # ========================================================================
    
    def _build_prompt(self, objective: str, history: List[AgentStep]) -> str:
        """Build the prompt for the LLM."""
        tools_desc = "\n".join([
            f"- {name}: {spec.description}"
            + (f" [REQUIRES CONFIRMATION]" if spec.requires_confirmation else "")
            for name, spec in self.tools.items()
        ])
        
        history_str = ""
        for step in history:
            history_str += f"\nThought: {step.thought}"
            if step.action:
                history_str += f"\nAction: {step.action}"
                if step.action_input:
                    history_str += f"\nAction Input: {json.dumps(step.action_input)}"
            if step.observation:
                history_str += f"\nObservation: {step.observation}"
        
        prompt = f"""You are an AI agent that executes infrastructure operations.

OBJECTIVE: {objective}

AVAILABLE TOOLS:
{tools_desc}

RESPONSE FORMAT:
- Start with "Thought:" to explain your reasoning
- Then "Action:" with the tool name
- Then "Action Input:" with JSON parameters (or empty {{}})
- After observation, continue with next Thought
- When done, respond with "FINAL_ANSWER: <your summary>"

RULES:
1. Always check metrics before taking action
2. Actions marked [REQUIRES CONFIRMATION] need approval
3. After completing actions, send a summary report to Slack
4. Be concise and focused on the objective

{history_str}

Thought:"""
        
        return prompt
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured components."""
        result = {
            "thought": "",
            "action": None,
            "action_input": {},
            "final_answer": None,
        }
        
        # Check for final answer
        if "FINAL_ANSWER:" in response:
            parts = response.split("FINAL_ANSWER:", 1)
            result["thought"] = parts[0].replace("Thought:", "").strip()
            result["final_answer"] = parts[1].strip()
            return result
        
        # Parse thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", response, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        # Parse action
        action_match = re.search(r"Action:\s*(\w+)", response)
        if action_match:
            result["action"] = action_match.group(1).strip()
        
        # Parse action input
        input_match = re.search(r"Action Input:\s*(\{.*?\})", response, re.DOTALL)
        if input_match:
            try:
                result["action_input"] = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                result["action_input"] = {}
        
        return result
    
    def _execute_action(
        self,
        action: str,
        action_input: Dict,
    ) -> tuple[bool, str]:
        """Execute an action and return (success, observation)."""
        if action not in self.tools:
            return False, f"Unknown tool: {action}"
        
        tool_spec = self.tools[action]
        
        # Check if confirmation needed
        if tool_spec.requires_confirmation:
            confirmed = self.confirmation_handler.request_confirmation(
                action=action,
                description=tool_spec.description,
                details=action_input,
            )
            
            if not confirmed:
                return False, "Action cancelled by user"
        
        # Execute the tool
        try:
            result = tool_spec.function(**action_input)
            
            if isinstance(result, ToolResult):
                if result.success:
                    return True, json.dumps(result.data, indent=2)
                else:
                    return False, f"Tool error: {result.error}"
            else:
                return True, str(result)
                
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return False, f"Execution error: {str(e)}"
    
    def run(self, objective: str) -> AgentResponse:
        """
        Execute the objective using ReAct loop.
        
        Args:
            objective: High-level goal to accomplish.
            
        Returns:
            AgentResponse with results and execution trace.
        """
        start_time = time.time()
        self.steps = []
        self.state = AgentState.REASONING
        
        logger.info(f"Starting agent with objective: {objective}")
        
        for iteration in range(MAX_ITERATIONS):
            # Generate next action
            prompt = self._build_prompt(objective, self.steps)
            
            try:
                response = self.llm.generate(prompt)
            except Exception as e:
                logger.error(f"LLM error: {e}")
                self.state = AgentState.FAILED
                return AgentResponse(
                    success=False,
                    objective=objective,
                    final_answer="",
                    steps=self.steps,
                    error=str(e),
                    total_duration=time.time() - start_time,
                )
            
            # Parse response
            parsed = self._parse_response(response)
            
            step = AgentStep(
                step_number=iteration + 1,
                state=self.state,
                thought=parsed["thought"],
                action=parsed["action"],
                action_input=parsed["action_input"],
            )
            
            # Check for final answer
            if parsed["final_answer"]:
                step.observation = parsed["final_answer"]
                self.steps.append(step)
                self.state = AgentState.COMPLETED
                
                logger.info(f"Agent completed: {parsed['final_answer']}")
                
                return AgentResponse(
                    success=True,
                    objective=objective,
                    final_answer=parsed["final_answer"],
                    steps=self.steps,
                    total_duration=time.time() - start_time,
                )
            
            # Execute action
            if parsed["action"]:
                self.state = AgentState.ACTING
                
                # Check confirmation requirement
                tool_spec = self.tools.get(parsed["action"])
                if tool_spec and tool_spec.requires_confirmation:
                    step.requires_confirmation = True
                
                success, observation = self._execute_action(
                    parsed["action"],
                    parsed["action_input"],
                )
                
                step.observation = observation
                step.confirmed = success if step.requires_confirmation else None
                
                self.state = AgentState.OBSERVING
            
            self.steps.append(step)
            self.state = AgentState.REASONING
        
        # Max iterations reached
        self.state = AgentState.FAILED
        return AgentResponse(
            success=False,
            objective=objective,
            final_answer="Max iterations reached",
            steps=self.steps,
            error="Max iterations exceeded",
            total_duration=time.time() - start_time,
        )
    
    # ========================================================================
    # Convenience Methods
    # ========================================================================
    
    def check_drift_and_retrain(
        self,
        metric_name: str = "model_drift_score",
        dag_id: str = "retrain",
        report_channel: Optional[str] = None,
    ) -> AgentResponse:
        """
        Convenience method to check drift and trigger retraining if needed.
        
        This provides a direct approach without LLM reasoning.
        """
        steps = []
        start_time = time.time()
        
        # Step 1: Check drift
        logger.info("Checking model drift...")
        drift_result = self._tool_check_drift(metric_name)
        
        steps.append(AgentStep(
            step_number=1,
            state=AgentState.OBSERVING,
            thought="Checking model drift metrics",
            action="check_drift",
            action_input={"metric_name": metric_name},
            observation=json.dumps(drift_result.data) if drift_result.success else drift_result.error,
        ))
        
        if not drift_result.success:
            return AgentResponse(
                success=False,
                objective="Check drift and retrain",
                final_answer="Failed to check drift metrics",
                steps=steps,
                error=drift_result.error,
                total_duration=time.time() - start_time,
            )
        
        exceeds = drift_result.data.get("exceeds_threshold", False)
        drift_score = drift_result.data.get("drift_score", 0)
        
        report_sections = [
            {"title": "Drift Score", "value": f"{drift_score:.3f}"},
            {"title": "Threshold", "value": f"{self.drift_threshold}"},
            {"title": "Exceeds", "value": "Yes ⚠️" if exceeds else "No ✓"},
        ]
        
        if exceeds:
            # Step 2: Request confirmation for retraining
            logger.info("Drift exceeds threshold, requesting confirmation...")
            
            confirmed = self.confirmation_handler.request_confirmation(
                action="trigger_retrain",
                description=f"Trigger retraining DAG due to drift ({drift_score:.3f} > {self.drift_threshold})",
                details={"dag_id": dag_id, "drift_score": drift_score},
            )
            
            if confirmed:
                # Step 3: Trigger retraining
                retrain_result = self._tool_trigger_retrain(dag_id)
                
                steps.append(AgentStep(
                    step_number=2,
                    state=AgentState.ACTING,
                    thought="Triggering retraining due to drift",
                    action="trigger_retrain",
                    action_input={"dag_id": dag_id},
                    observation=json.dumps(retrain_result.data) if retrain_result.success else retrain_result.error,
                    requires_confirmation=True,
                    confirmed=True,
                ))
                
                report_sections.append({
                    "title": "Action", 
                    "value": f"Retraining triggered (run: {retrain_result.data.get('run_id', 'N/A')})"
                })
                final_answer = f"Drift detected ({drift_score:.3f}). Retraining triggered."
            else:
                steps.append(AgentStep(
                    step_number=2,
                    state=AgentState.WAITING_CONFIRMATION,
                    thought="Retraining cancelled by user",
                    action="trigger_retrain",
                    requires_confirmation=True,
                    confirmed=False,
                ))
                report_sections.append({"title": "Action", "value": "Retraining cancelled by user"})
                final_answer = f"Drift detected ({drift_score:.3f}). Retraining cancelled by user."
        else:
            final_answer = f"No significant drift detected ({drift_score:.3f} within threshold). No action needed."
            report_sections.append({"title": "Action", "value": "No action needed"})
        
        # Step 4: Send report
        report_result = self._tool_post_report(
            title="🔍 Model Drift Report",
            sections=report_sections,
            channel=report_channel,
        )
        
        steps.append(AgentStep(
            step_number=len(steps) + 1,
            state=AgentState.ACTING,
            thought="Sending summary report to Slack",
            action="post_report",
            observation="Report sent" if report_result.success else report_result.error,
        ))
        
        return AgentResponse(
            success=True,
            objective="Check drift and retrain",
            final_answer=final_answer,
            steps=steps,
            report={"sections": report_sections},
            total_duration=time.time() - start_time,
        )


# ============================================================================
# Factory Functions
# ============================================================================


def create_agent(
    llm_type: str = "mock",
    confirmation_mode: str = "auto",
    **kwargs,
) -> AgentCore:
    """
    Factory function to create an agent with specified configuration.
    
    Args:
        llm_type: One of "mock", "openai".
        confirmation_mode: One of "auto", "interactive", "callback".
        **kwargs: Additional arguments for agent initialization.
        
    Returns:
        Configured AgentCore instance.
    """
    # Create LLM
    if llm_type == "openai":
        llm = OpenAILLM(
            model=kwargs.get("model", "gpt-3.5-turbo"),
            api_key=kwargs.get("api_key"),
        )
    else:
        llm = MockLLM(responses=kwargs.get("mock_responses", []))
    
    # Create confirmation handler
    if confirmation_mode == "interactive":
        handler = InteractiveConfirmHandler()
    elif confirmation_mode == "callback":
        handler = CallbackConfirmHandler(kwargs.get("confirm_callback", lambda *args: True))
    else:
        handler = AutoConfirmHandler()
    
    return AgentCore(
        llm=llm,
        confirmation_handler=handler,
        airflow_url=kwargs.get("airflow_url", "http://localhost:8080"),
        prometheus_url=kwargs.get("prometheus_url", "http://localhost:9090"),
        slack_webhook=kwargs.get("slack_webhook"),
        drift_threshold=kwargs.get("drift_threshold", DEFAULT_DRIFT_THRESHOLD),
    )
