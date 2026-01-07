import { useState, useEffect } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import { AlertTriangle, CheckCircle, Info, Bell, Server } from 'lucide-react';
import { monitoringApi } from '../api/client';
import './Monitoring.css';

function Monitoring() {
    const [data, setData] = useState(null);
    const [activeTab, setActiveTab] = useState('overview');

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await monitoringApi.getOverview();
                setData(response.data);
            } catch (error) {
                // Mock data
                setData({
                    system_metrics: [
                        { name: 'CPU Usage', value: 45.2, unit: '%', status: 'healthy' },
                        { name: 'Memory', value: 62.8, unit: '%', status: 'healthy' },
                        { name: 'Disk', value: 38.5, unit: '%', status: 'healthy' },
                        { name: 'Network In', value: 245.3, unit: 'MB/s', status: 'healthy' },
                        { name: 'Network Out', value: 128.7, unit: 'MB/s', status: 'healthy' },
                        { name: 'GPU', value: 15.2, unit: '%', status: 'healthy' },
                    ],
                    services: [
                        { name: 'Kafka Broker', status: 'healthy', uptime: '7d 12h 34m' },
                        { name: 'Spark Streaming', status: 'healthy', uptime: '7d 12h 30m' },
                        { name: 'Delta Lake', status: 'healthy', uptime: '7d 12h 34m' },
                        { name: 'Feast (Online)', status: 'healthy', uptime: '7d 10h 15m' },
                        { name: 'MLflow', status: 'healthy', uptime: '1h 30m' },
                        { name: 'Prediction API', status: 'healthy', uptime: '45m' },
                    ],
                    alerts: [
                        { id: '001', severity: 'info', title: 'Model Retrained', message: 'Prophet model retrained with MAPE 4.2%', timestamp: new Date().toISOString(), resolved: true },
                        { id: '002', severity: 'warning', title: 'Kafka Consumer Lag', message: 'Consumer lag reached 500 on partition 2', timestamp: new Date().toISOString(), resolved: true },
                    ],
                    metrics_history: {
                        cpu: Array.from({ length: 24 }, (_, i) => ({ time: `${i}:00`, value: 30 + Math.random() * 30 })),
                        memory: Array.from({ length: 24 }, (_, i) => ({ time: `${i}:00`, value: 50 + Math.random() * 20 })),
                    },
                });
            }
        };
        fetchData();
    }, []);

    const severityIcons = {
        info: <Info size={16} />,
        warning: <AlertTriangle size={16} />,
        critical: <AlertTriangle size={16} />,
    };

    if (!data) return <div className="page-container"><div className="skeleton" style={{ height: 400 }}></div></div>;

    return (
        <div className="page-container">
            {/* Tabs */}
            <div className="tabs">
                <button className={`tab ${activeTab === 'overview' ? 'active' : ''}`} onClick={() => setActiveTab('overview')}>Overview</button>
                <button className={`tab ${activeTab === 'services' ? 'active' : ''}`} onClick={() => setActiveTab('services')}>Services</button>
                <button className={`tab ${activeTab === 'alerts' ? 'active' : ''}`} onClick={() => setActiveTab('alerts')}>Alerts</button>
            </div>

            {activeTab === 'overview' && (
                <>
                    {/* System Metrics */}
                    <div className="metrics-grid">
                        {data.system_metrics.map((metric) => (
                            <div key={metric.name} className="metric-card">
                                <div className="metric-header">
                                    <span className="metric-name">{metric.name}</span>
                                    <span className={`status-badge ${metric.status}`}>{metric.status}</span>
                                </div>
                                <div className="metric-value">{metric.value}{metric.unit}</div>
                                <div className="metric-bar">
                                    <div className="metric-fill" style={{ width: `${Math.min(metric.value, 100)}%` }}></div>
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* CPU Chart */}
                    <div className="chart-container">
                        <div className="card-header">
                            <h3 className="card-title">CPU Usage (24h)</h3>
                        </div>
                        <ResponsiveContainer width="100%" height={300}>
                            <LineChart data={data.metrics_history.cpu}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                <XAxis dataKey="time" stroke="#64748b" fontSize={12} />
                                <YAxis stroke="#64748b" fontSize={12} domain={[0, 100]} tickFormatter={v => `${v}%`} />
                                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
                                <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} dot={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </>
            )}

            {activeTab === 'services' && (
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title"><Server size={18} /> Service Health</h3>
                    </div>
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Service</th>
                                <th>Status</th>
                                <th>Uptime</th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.services.map((service) => (
                                <tr key={service.name}>
                                    <td>{service.name}</td>
                                    <td><span className={`status-badge ${service.status}`}>{service.status}</span></td>
                                    <td>{service.uptime}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {activeTab === 'alerts' && (
                <div className="card">
                    <div className="card-header">
                        <h3 className="card-title"><Bell size={18} /> Alerts</h3>
                    </div>
                    <div className="alerts-list">
                        {data.alerts.map((alert) => (
                            <div key={alert.id} className={`alert-item ${alert.severity}`}>
                                <div className="alert-icon">{severityIcons[alert.severity]}</div>
                                <div className="alert-content">
                                    <div className="alert-title">{alert.title}</div>
                                    <div className="alert-message">{alert.message}</div>
                                </div>
                                <div className="alert-meta">
                                    <span className={`status-badge ${alert.resolved ? 'success' : alert.severity}`}>
                                        {alert.resolved ? 'Resolved' : 'Active'}
                                    </span>
                                    <span className="alert-time">{new Date(alert.timestamp).toLocaleString()}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

export default Monitoring;
