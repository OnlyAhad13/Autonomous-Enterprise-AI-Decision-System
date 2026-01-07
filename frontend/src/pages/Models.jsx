import { useState, useEffect, useCallback } from 'react';
import { Cpu, RefreshCw, Rocket, Clock, Check, AlertCircle, Play, Settings } from 'lucide-react';
import { modelsApi } from '../api/client';
import './Models.css';

function Models() {
    const [models, setModels] = useState([]);
    const [runs, setRuns] = useState([]);
    const [activeTab, setActiveTab] = useState('registry');
    const [training, setTraining] = useState(false);
    const [trainingStatus, setTrainingStatus] = useState(null);
    const [deployedModel, setDeployedModel] = useState(null);
    const [selectedModelType, setSelectedModelType] = useState('random_forest');
    const [deploying, setDeploying] = useState({});

    const fetchData = useCallback(async () => {
        try {
            const [modelsRes, runsRes, deployedRes] = await Promise.all([
                modelsApi.getRegistry(),
                modelsApi.getRuns(10),
                modelsApi.getDeployed(),
            ]);
            setModels(modelsRes.data.models || []);
            setRuns(runsRes.data.runs || []);
            if (deployedRes.data.deployed) {
                setDeployedModel(deployedRes.data);
            }
        } catch (error) {
            console.error('Failed to fetch data:', error);
        }
    }, []);

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 10000);
        return () => clearInterval(interval);
    }, [fetchData]);

    // Poll training status when training
    useEffect(() => {
        if (!training) return;

        const pollStatus = async () => {
            try {
                const res = await modelsApi.getTrainingStatus();
                setTrainingStatus(res.data);

                if (!res.data.is_training && res.data.progress >= 100) {
                    setTraining(false);
                    fetchData();
                }
            } catch (e) {
                console.error('Status poll failed:', e);
            }
        };

        pollStatus();
        const interval = setInterval(pollStatus, 2000);
        return () => clearInterval(interval);
    }, [training, fetchData]);

    const handleTrain = async () => {
        setTraining(true);
        setTrainingStatus({ progress: 0, message: 'Starting...' });
        try {
            await modelsApi.train(selectedModelType);
        } catch (error) {
            console.error('Train failed:', error);
            setTraining(false);
            setTrainingStatus({ progress: 0, message: `Failed: ${error.message}` });
        }
    };

    const handleDeploy = async (modelName, version) => {
        setDeploying(prev => ({ ...prev, [modelName]: true }));
        try {
            const res = await modelsApi.deploy(modelName, version || 'latest');
            setDeployedModel({
                deployed: true,
                model_name: res.data.model_name,
                version: res.data.version,
                loaded_at: res.data.loaded_at,
            });
            fetchData();
        } catch (error) {
            console.error('Deploy failed:', error);
            alert(`Deploy failed: ${error.response?.data?.detail || error.message}`);
        }
        setDeploying(prev => ({ ...prev, [modelName]: false }));
    };

    const formatMetricValue = (value) => {
        if (value === undefined || value === null) return '-';
        if (typeof value === 'number') {
            return value < 1 ? (value * 100).toFixed(1) + '%' : value.toFixed(2);
        }
        return value;
    };

    return (
        <div className="page-container">
            {/* Training Panel */}
            <div className="card training-panel">
                <div className="card-header">
                    <h3 className="card-title"><Settings size={18} /> Train New Model</h3>
                </div>
                <div className="training-controls">
                    <div className="model-type-selector">
                        <label>Model Type:</label>
                        <select
                            value={selectedModelType}
                            onChange={(e) => setSelectedModelType(e.target.value)}
                            disabled={training}
                        >
                            <option value="random_forest">Random Forest</option>
                            <option value="gradient_boosting">Gradient Boosting</option>
                            <option value="logistic_regression">Logistic Regression</option>
                        </select>
                    </div>
                    <button
                        className="btn btn-primary train-btn"
                        onClick={handleTrain}
                        disabled={training}
                    >
                        {training ? (
                            <>
                                <RefreshCw size={16} className="spinning" />
                                Training...
                            </>
                        ) : (
                            <>
                                <Play size={16} />
                                Train on Live Data
                            </>
                        )}
                    </button>
                </div>

                {trainingStatus && (training || trainingStatus.progress > 0) && (
                    <div className="training-progress">
                        <div className="progress-bar">
                            <div
                                className="progress-fill"
                                style={{ width: `${trainingStatus.progress || 0}%` }}
                            />
                        </div>
                        <span className="progress-message">{trainingStatus.message}</span>
                    </div>
                )}
            </div>

            {/* Deployed Model Info */}
            {deployedModel && deployedModel.deployed && (
                <div className="card deployed-info">
                    <div className="deployed-badge">
                        <Check size={16} />
                        <span>Currently Deployed</span>
                    </div>
                    <div className="deployed-details">
                        <strong>{deployedModel.model_name}</strong>
                        <span>Version: {deployedModel.version}</span>
                        <span>Since: {new Date(deployedModel.loaded_at).toLocaleString()}</span>
                    </div>
                </div>
            )}

            {/* Tabs */}
            <div className="tabs">
                <button className={`tab ${activeTab === 'registry' ? 'active' : ''}`} onClick={() => setActiveTab('registry')}>
                    Model Registry
                </button>
                <button className={`tab ${activeTab === 'runs' ? 'active' : ''}`} onClick={() => setActiveTab('runs')}>
                    Training Runs
                </button>
            </div>

            {activeTab === 'registry' && (
                <div className="models-grid">
                    {models.length === 0 ? (
                        <div className="empty-state">
                            <Cpu size={48} />
                            <h3>No models found</h3>
                            <p>Train a model to get started</p>
                        </div>
                    ) : (
                        models.map((model) => (
                            <div key={model.name} className="model-card">
                                <div className="model-header">
                                    <Cpu size={24} className="model-icon" />
                                    <div>
                                        <h3 className="model-name">{model.name}</h3>
                                        <p className="model-desc">{model.description}</p>
                                    </div>
                                </div>

                                <div className="model-tags">
                                    {Object.entries(model.tags || {}).map(([key, val]) => (
                                        <span key={key} className="tag">{key}: {val}</span>
                                    ))}
                                </div>

                                <div className="model-versions">
                                    <h4>Versions</h4>
                                    {(model.latest_versions || []).map((v) => (
                                        <div key={v.version} className="version-row">
                                            <span className="version-num">v{v.version}</span>
                                            <span className={`status-badge ${(v.stage || 'none').toLowerCase()}`}>
                                                {v.stage || 'None'}
                                            </span>
                                            <span className="version-metric">
                                                {v.metrics?.accuracy
                                                    ? `Acc: ${formatMetricValue(v.metrics.accuracy)}`
                                                    : v.metrics?.mape
                                                        ? `MAPE: ${formatMetricValue(v.metrics.mape)}`
                                                        : 'No metrics'
                                                }
                                            </span>
                                        </div>
                                    ))}
                                </div>

                                <div className="model-actions">
                                    <button
                                        className="btn btn-success"
                                        onClick={() => handleDeploy(
                                            model.name,
                                            model.latest_versions?.[0]?.version
                                        )}
                                        disabled={deploying[model.name]}
                                    >
                                        {deploying[model.name] ? (
                                            <RefreshCw size={16} className="spinning" />
                                        ) : (
                                            <Rocket size={16} />
                                        )}
                                        {deployedModel?.model_name === model.name ? 'Redeploy' : 'Deploy'}
                                    </button>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            )}

            {activeTab === 'runs' && (
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title"><Clock size={18} /> Training Runs from MLflow</h3>
                        <button className="btn btn-sm" onClick={fetchData}>
                            <RefreshCw size={14} /> Refresh
                        </button>
                    </div>
                    {runs.length === 0 ? (
                        <div className="empty-state">
                            <Clock size={48} />
                            <h3>No training runs</h3>
                            <p>Train a model to see runs here</p>
                        </div>
                    ) : (
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Run ID</th>
                                    <th>Model Type</th>
                                    <th>Status</th>
                                    <th>Accuracy</th>
                                    <th>F1 Score</th>
                                    <th>Started</th>
                                </tr>
                            </thead>
                            <tbody>
                                {runs.map((run) => (
                                    <tr key={run.run_id}>
                                        <td><code>{run.run_id?.slice(0, 8)}...</code></td>
                                        <td className="model-type">{run.model_type || 'unknown'}</td>
                                        <td>
                                            <span className={`status-badge ${run.status}`}>
                                                {run.status === 'finished' && <Check size={12} />}
                                                {run.status === 'failed' && <AlertCircle size={12} />}
                                                {run.status}
                                            </span>
                                        </td>
                                        <td>{formatMetricValue(run.metrics?.accuracy)}</td>
                                        <td>{formatMetricValue(run.metrics?.f1_weighted)}</td>
                                        <td>{run.start_time ? new Date(run.start_time).toLocaleString() : '-'}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </div>
            )}
        </div>
    );
}

export default Models;
