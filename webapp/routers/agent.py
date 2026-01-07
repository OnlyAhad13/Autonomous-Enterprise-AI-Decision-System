"""
Agent Router - AI Agent Control with Real LLM Integration.
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️ python-dotenv not installed, using existing environment variables")

# Try to import agent core
AGENT_AVAILABLE = False
AgentCore = None
AutoConfirmHandler = None
OpenAILLM = None

try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from agents.agent_core import AgentCore as AC, AutoConfirmHandler as ACH, OpenAILLM as OAILLM
    AgentCore = AC
    AutoConfirmHandler = ACH
    OpenAILLM = OAILLM
    AGENT_AVAILABLE = True
    
    # Check if OpenAI API key is available
    if os.environ.get("OPENAI_API_KEY"):
        print("✅ Agent Core loaded with OpenAI LLM (API key found)")
    else:
        print("⚠️ Agent Core loaded but OPENAI_API_KEY not set - will use MockLLM")
except ImportError as e:
    print(f"⚠️ Agent not available: {e}")
except Exception as e:
    print(f"⚠️ Agent error: {e}")


router = APIRouter()


# Store for executions
executions_store: List[Dict] = []
current_execution: Optional[Dict] = None


# ============================================================================
# Response Models
# ============================================================================

class AgentStep(BaseModel):
    """Single agent execution step."""
    step_number: int
    state: str
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    timestamp: str


class AgentExecution(BaseModel):
    """Complete agent execution record."""
    execution_id: str
    objective: str
    status: str
    started_at: str
    ended_at: Optional[str] = None
    steps: List[AgentStep]
    final_answer: Optional[str] = None
    duration_seconds: Optional[float] = None


class AgentStatus(BaseModel):
    """Current agent status."""
    state: str
    current_objective: Optional[str] = None
    current_step: Optional[int] = None
    available_tools: List[str]
    last_execution_id: Optional[str] = None
    agent_available: bool


class AgentTool(BaseModel):
    """Agent tool specification."""
    name: str
    description: str
    requires_confirmation: bool
    parameters: Dict[str, str]


# ============================================================================
# Background Execution
# ============================================================================

async def run_agent_execution(execution_id: str, objective: str):
    """Run agent in background."""
    global current_execution
    
    execution = {
        "execution_id": execution_id,
        "objective": objective,
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "ended_at": None,
        "steps": [],
        "final_answer": None,
        "duration_seconds": None,
    }
    
    current_execution = execution
    start_time = datetime.now()
    
    if AGENT_AVAILABLE and AgentCore is not None:
        try:
            # Create LLM - use OpenAI if API key available, otherwise MockLLM
            llm = None
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key and OpenAILLM is not None:
                llm = OpenAILLM(api_key=api_key, model="gpt-4o")
                print(f"🤖 Using OpenAI GPT-4o for execution: {execution_id}")
            else:
                print(f"🔧 Using MockLLM for execution: {execution_id}")
            
            # Create agent with LLM and auto-confirm handler
            agent = AgentCore(
                llm=llm,
                confirmation_handler=AutoConfirmHandler(),
            )
            
            # Execute using the correct method name
            response = agent.run(objective)
            
            # Convert steps
            for step in response.steps:
                execution["steps"].append({
                    "step_number": step.step_number,
                    "state": step.state.value if hasattr(step.state, 'value') else str(step.state),
                    "thought": step.thought,
                    "action": step.action,
                    "action_input": step.action_input,
                    "observation": step.observation,
                    "timestamp": step.timestamp,
                })
            
            execution["status"] = "completed" if response.success else "failed"
            execution["final_answer"] = response.final_answer
            
        except Exception as e:
            print(f"Agent execution error: {e}")
            execution["status"] = "failed"
            execution["final_answer"] = f"Error: {str(e)}"
    else:
        # Mock execution when agent not available
        import asyncio
        
        mock_steps = [
            {
                "step_number": 1,
                "state": "reasoning",
                "thought": f"Analyzing objective: {objective}",
                "action": "query_metrics",
                "action_input": {"metric_name": "system_health"},
                "observation": "CPU: 45%, Memory: 62%, Disk: 38%",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "step_number": 2,
                "state": "reasoning",
                "thought": "System metrics look healthy. Checking for drift...",
                "action": "check_drift",
                "action_input": {"threshold": 0.1},
                "observation": "Drift score: 0.05 (within threshold)",
                "timestamp": (datetime.now() + timedelta(seconds=2)).isoformat(),
            },
            {
                "step_number": 3,
                "state": "completed",
                "thought": "All checks passed. System is operating normally.",
                "action": None,
                "action_input": None,
                "observation": None,
                "timestamp": (datetime.now() + timedelta(seconds=4)).isoformat(),
            },
        ]
        
        for step in mock_steps:
            await asyncio.sleep(1)  # Simulate work
            execution["steps"].append(step)
        
        execution["status"] = "completed"
        execution["final_answer"] = "Health check complete. All systems operating within normal parameters."
    
    execution["ended_at"] = datetime.now().isoformat()
    execution["duration_seconds"] = (datetime.now() - start_time).total_seconds()
    
    # Store execution
    executions_store.insert(0, execution)
    if len(executions_store) > 50:
        executions_store.pop()
    
    current_execution = None


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/status", response_model=AgentStatus)
async def get_agent_status():
    """Get current agent status."""
    tools = ["query_metrics", "check_drift", "trigger_retrain", "send_slack_message", "get_dag_status", "post_report"]
    
    if AGENT_AVAILABLE and AgentCore is not None:
        try:
            agent = AgentCore(confirmation_handler=AutoConfirmHandler())
            tools = list(agent.tools.keys())
        except Exception as e:
            print(f"Error getting tools: {e}")
    
    return AgentStatus(
        state="running" if current_execution else "idle",
        current_objective=current_execution["objective"] if current_execution else None,
        current_step=len(current_execution["steps"]) if current_execution else None,
        available_tools=tools,
        last_execution_id=executions_store[0]["execution_id"] if executions_store else None,
        agent_available=AGENT_AVAILABLE,
    )


@router.get("/tools")
async def get_agent_tools():
    """Get available agent tools."""
    tools = []
    
    if AGENT_AVAILABLE and AgentCore is not None:
        try:
            agent = AgentCore(confirmation_handler=AutoConfirmHandler())
            for name, spec in agent.tools.items():
                tools.append(AgentTool(
                    name=name,
                    description=spec.description,
                    requires_confirmation=spec.requires_confirmation,
                    parameters=spec.parameters,
                ))
        except Exception as e:
            print(f"Error getting tools: {e}")
    
    if not tools:
        # Default tools
        tools = [
            AgentTool(name="query_metrics", description="Query Prometheus metrics", requires_confirmation=False, parameters={"metric_name": "string"}),
            AgentTool(name="check_drift", description="Check for data/model drift", requires_confirmation=False, parameters={"threshold": "float"}),
            AgentTool(name="trigger_retrain", description="Trigger model retraining", requires_confirmation=True, parameters={"model_type": "string"}),
            AgentTool(name="send_slack_message", description="Send Slack notification", requires_confirmation=True, parameters={"message": "string"}),
            AgentTool(name="get_dag_status", description="Get Airflow DAG status", requires_confirmation=False, parameters={"dag_id": "string"}),
            AgentTool(name="post_report", description="Generate analysis report", requires_confirmation=False, parameters={"report_type": "string"}),
        ]
    
    return {"tools": tools}


@router.get("/executions")
async def get_agent_executions(limit: int = 10):
    """Get recent agent executions."""
    return {
        "executions": executions_store[:limit],
        "total": len(executions_store),
    }


@router.post("/execute")
async def execute_agent(
    background_tasks: BackgroundTasks,
    objective: str = "Monitor system health and detect anomalies",
):
    """Start a new agent execution."""
    if current_execution:
        raise HTTPException(status_code=409, detail="An execution is already in progress")
    
    execution_id = f"exec_{int(datetime.now().timestamp() * 1000)}"
    
    # Start background execution
    background_tasks.add_task(run_agent_execution, execution_id, objective)
    
    return {
        "status": "started",
        "execution_id": execution_id,
        "objective": objective,
        "message": f"Agent execution {execution_id} started",
        "timestamp": datetime.now().isoformat(),
        "agent_available": AGENT_AVAILABLE,
    }


@router.get("/executions/{execution_id}")
async def get_execution_details(execution_id: str):
    """Get details of a specific execution."""
    # Check current execution
    if current_execution and current_execution["execution_id"] == execution_id:
        return current_execution
    
    # Check stored executions
    for exec in executions_store:
        if exec["execution_id"] == execution_id:
            return exec
    
    raise HTTPException(status_code=404, detail="Execution not found")


@router.get("/executions/{execution_id}/stream")
async def stream_execution(execution_id: str):
    """Stream execution progress via SSE."""
    from fastapi.responses import StreamingResponse
    import asyncio
    
    async def execution_generator():
        last_step_count = 0
        max_wait = 60  # 60 second timeout
        waited = 0
        
        while waited < max_wait:
            if current_execution and current_execution["execution_id"] == execution_id:
                # Send new steps
                if len(current_execution["steps"]) > last_step_count:
                    for step in current_execution["steps"][last_step_count:]:
                        yield f"data: {json.dumps({'type': 'step', 'data': step})}\n\n"
                    last_step_count = len(current_execution["steps"])
                
                if current_execution["status"] != "running":
                    yield f"data: {json.dumps({'type': 'complete', 'data': current_execution})}\n\n"
                    break
            else:
                # Check if completed in store
                for exec in executions_store:
                    if exec["execution_id"] == execution_id:
                        yield f"data: {json.dumps({'type': 'complete', 'data': exec})}\n\n"
                        return
            
            await asyncio.sleep(0.5)
            waited += 0.5
    
    return StreamingResponse(
        execution_generator(),
        media_type="text/event-stream",
    )
