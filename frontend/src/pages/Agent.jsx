import { useState, useEffect, useRef } from 'react';
import { Bot, Wrench, Clock, Play, CheckCircle, AlertTriangle, Zap, Loader } from 'lucide-react';
import { agentApi } from '../api/client';
import './Agent.css';

function Agent() {
    const [status, setStatus] = useState(null);
    const [tools, setTools] = useState([]);
    const [executions, setExecutions] = useState([]);
    const [selectedExecution, setSelectedExecution] = useState(null);
    const [objective, setObjective] = useState('Monitor system health and detect anomalies');
    const [activeTab, setActiveTab] = useState('status');
    const [isExecuting, setIsExecuting] = useState(false);
    const [liveSteps, setLiveSteps] = useState([]);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [statusRes, toolsRes, execRes] = await Promise.all([
                    agentApi.getStatus(),
                    agentApi.getTools(),
                    agentApi.getExecutions(10),
                ]);
                setStatus(statusRes.data);
                setTools(toolsRes.data.tools || []);
                setExecutions(execRes.data.executions || []);
            } catch (error) {
                console.error('Failed to fetch agent data:', error);
            }
        };
        fetchData();
        const interval = setInterval(fetchData, 5000);
        return () => clearInterval(interval);
    }, []);

    const handleExecute = async () => {
        setIsExecuting(true);
        setLiveSteps([]);

        try {
            const res = await agentApi.execute(objective);
            const executionId = res.data.execution_id;

            // Connect to SSE stream for live updates
            const eventSource = new EventSource(`http://localhost:8080/api/agent/executions/${executionId}/stream`);

            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'step') {
                    setLiveSteps(prev => [...prev, data.data]);
                } else if (data.type === 'complete') {
                    setSelectedExecution(data.data);
                    setIsExecuting(false);
                    eventSource.close();

                    // Refresh executions list
                    agentApi.getExecutions(10).then(res => {
                        setExecutions(res.data.executions || []);
                    });
                }
            };

            eventSource.onerror = () => {
                setIsExecuting(false);
                eventSource.close();
                // Fallback: poll for completion
                setTimeout(async () => {
                    try {
                        const execRes = await agentApi.getExecution(executionId);
                        setSelectedExecution(execRes.data);
                        const listRes = await agentApi.getExecutions(10);
                        setExecutions(listRes.data.executions || []);
                    } catch (e) {
                        console.error('Failed to fetch execution:', e);
                    }
                }, 2000);
            };

        } catch (error) {
            console.error('Failed to start agent:', error);
            setIsExecuting(false);
            alert(`Failed to start agent: ${error.message}`);
        }
    };

    return (
        <div className="page-container">
            <div className="tabs">
                <button className={`tab ${activeTab === 'status' ? 'active' : ''}`} onClick={() => setActiveTab('status')}>Status</button>
                <button className={`tab ${activeTab === 'tools' ? 'active' : ''}`} onClick={() => setActiveTab('tools')}>Tools</button>
                <button className={`tab ${activeTab === 'history' ? 'active' : ''}`} onClick={() => setActiveTab('history')}>Execution History</button>
            </div>

            {activeTab === 'status' && (
                <>
                    <div className="agent-status-card">
                        <div className="agent-icon">
                            <Bot size={48} />
                        </div>
                        <div className="agent-info">
                            <h2>Agent Core</h2>
                            <div className="status-row">
                                <span className={`status-badge ${status?.state || 'idle'}`}>
                                    {status?.state || 'Unknown'}
                                </span>
                                {status?.agent_available ? (
                                    <span className="status-badge success">LLM Connected</span>
                                ) : (
                                    <span className="status-badge warning">Mock Mode</span>
                                )}
                            </div>
                        </div>
                        <div className="agent-stats">
                            <div className="agent-stat">
                                <span className="stat-value">{tools.length}</span>
                                <span className="stat-label">Tools</span>
                            </div>
                            <div className="agent-stat">
                                <span className="stat-value">{executions.length}</span>
                                <span className="stat-label">Runs</span>
                            </div>
                        </div>
                    </div>

                    <div className="card">
                        <div className="card-header">
                            <h3 className="card-title"><Zap size={18} /> Execute Agent</h3>
                        </div>
                        <div className="form-group">
                            <label>Objective</label>
                            <textarea
                                value={objective}
                                onChange={e => setObjective(e.target.value)}
                                placeholder="Enter the objective for the agent..."
                                rows={3}
                            />
                        </div>
                        <button
                            className="btn btn-primary"
                            onClick={handleExecute}
                            disabled={isExecuting}
                        >
                            {isExecuting ? (
                                <><Loader size={16} className="spinning" /> Executing...</>
                            ) : (
                                <><Play size={16} /> Start Execution</>
                            )}
                        </button>
                    </div>

                    {/* Live Execution View */}
                    {(isExecuting || liveSteps.length > 0) && (
                        <div className="card live-execution">
                            <div className="card-header">
                                <h3 className="card-title">
                                    {isExecuting ? <><Loader size={18} className="spinning" /> Live Execution</> : 'Execution Complete'}
                                </h3>
                            </div>
                            <div className="execution-steps">
                                {liveSteps.map((step, idx) => (
                                    <div key={idx} className={`step-item ${step.state}`}>
                                        <div className="step-number">{step.step_number}</div>
                                        <div className="step-content">
                                            <div className="step-thought">{step.thought}</div>
                                            {step.action && (
                                                <div className="step-action">
                                                    <code>{step.action}</code>
                                                    {step.action_input && (
                                                        <span className="action-input">
                                                            ({JSON.stringify(step.action_input)})
                                                        </span>
                                                    )}
                                                </div>
                                            )}
                                            {step.observation && (
                                                <div className="step-observation">{step.observation}</div>
                                            )}
                                        </div>
                                        <div className={`step-state ${step.state}`}>
                                            {step.state === 'completed' ? <CheckCircle size={16} /> : <Zap size={16} />}
                                        </div>
                                    </div>
                                ))}
                                {isExecuting && (
                                    <div className="step-item pending">
                                        <div className="step-number">?</div>
                                        <div className="step-content">
                                            <div className="step-thought">Thinking...</div>
                                        </div>
                                        <Loader size={16} className="spinning" />
                                    </div>
                                )}
                            </div>
                            {selectedExecution?.final_answer && !isExecuting && (
                                <div className="final-answer">
                                    <strong>Result:</strong> {selectedExecution.final_answer}
                                </div>
                            )}
                        </div>
                    )}
                </>
            )}

            {activeTab === 'tools' && (
                <div className="tools-grid">
                    {tools.map((tool) => (
                        <div key={tool.name} className="tool-card">
                            <div className="tool-header">
                                <Wrench size={20} className="tool-icon" />
                                <h4 className="tool-name">{tool.name}</h4>
                            </div>
                            <p className="tool-desc">{tool.description}</p>
                            {tool.requires_confirmation && (
                                <div className="tool-warning">
                                    <AlertTriangle size={14} /> Requires confirmation
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            )}

            {activeTab === 'history' && (
                <div className="execution-list">
                    {executions.map((exec) => (
                        <div
                            key={exec.execution_id}
                            className={`execution-card ${selectedExecution?.execution_id === exec.execution_id ? 'selected' : ''}`}
                            onClick={() => setSelectedExecution(selectedExecution?.execution_id === exec.execution_id ? null : exec)}
                        >
                            <div className="execution-header">
                                <span className="execution-id">{exec.execution_id}</span>
                                <span className={`status-badge ${exec.status}`}>{exec.status}</span>
                            </div>
                            <div className="execution-objective">{exec.objective}</div>
                            <div className="execution-meta">
                                <Clock size={14} /> {exec.duration_seconds?.toFixed(1) || '?'}s
                                <span className="execution-time">
                                    {new Date(exec.started_at).toLocaleString()}
                                </span>
                            </div>

                            {selectedExecution?.execution_id === exec.execution_id && (
                                <div className="execution-steps">
                                    <h4>Steps ({exec.steps?.length || 0})</h4>
                                    {exec.steps?.map((step, idx) => (
                                        <div key={idx} className={`step-item ${step.state}`}>
                                            <div className="step-number">{step.step_number}</div>
                                            <div className="step-content">
                                                <div className="step-thought">{step.thought}</div>
                                                {step.action && <div className="step-action"><code>{step.action}</code></div>}
                                                {step.observation && <div className="step-observation">{step.observation}</div>}
                                            </div>
                                        </div>
                                    ))}
                                    {exec.final_answer && (
                                        <div className="final-answer">
                                            <strong>Result:</strong> {exec.final_answer}
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    ))}
                    {executions.length === 0 && (
                        <div className="no-executions">
                            No executions yet. Start one from the Status tab!
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export default Agent;
