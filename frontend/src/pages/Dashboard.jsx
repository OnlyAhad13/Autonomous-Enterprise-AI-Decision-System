import { useState, useEffect } from 'react';
import {
    LineChart, Line, AreaChart, Area,
    XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import { TrendingUp, TrendingDown, Minus, Activity, Cpu, Target, Clock, CheckCircle, Server, RefreshCw } from 'lucide-react';
import { useStream } from '../context/StreamContext';
import { dashboardApi } from '../api/client';
import './Dashboard.css';

const iconMap = {
    activity: Activity,
    cpu: Cpu,
    target: Target,
    clock: Clock,
    'check-circle': CheckCircle,
    server: Server,
};

function KPICard({ title, value, change, trend, icon }) {
    const Icon = iconMap[icon] || Activity;
    const TrendIcon = trend === 'up' ? TrendingUp : trend === 'down' ? TrendingDown : Minus;

    return (
        <div className="kpi-card">
            <div className="kpi-header">
                <Icon size={24} className="kpi-icon" />
                <span className={`kpi-change ${trend}`}>
                    <TrendIcon size={14} />
                    {Math.abs(change)}%
                </span>
            </div>
            <div className="kpi-value">{value}</div>
            <div className="kpi-label">{title}</div>
        </div>
    );
}

function StatusBadge({ status }) {
    return <span className={`status-badge ${status}`}>{status}</span>;
}

function Dashboard() {
    const { events, isStreaming, isConnected } = useStream();
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [lastUpdate, setLastUpdate] = useState(null);

    // Fetch data with auto-refresh
    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await dashboardApi.getStats();
                setStats(response.data);
                setLastUpdate(new Date());
            } catch (error) {
                console.error('Failed to fetch dashboard stats:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
        // Auto-refresh every 5 seconds
        const interval = setInterval(fetchData, 5000);
        return () => clearInterval(interval);
    }, []);

    // Calculate dynamic KPIs from stream events
    const getStreamKPIs = () => {
        if (!events || events.length === 0) return null;

        const prices = events
            .filter(e => e.value?.price || e.value?.total)
            .map(e => e.value.total || e.value.price)
            .slice(0, 100);

        return {
            eventCount: events.length,
            avgPrice: prices.length > 0 ? prices.reduce((a, b) => a + b, 0) / prices.length : 0,
        };
    };

    const streamKPIs = getStreamKPIs();

    if (loading && !stats) {
        return (
            <div className="page-container">
                <div className="kpi-grid">
                    {[...Array(6)].map((_, i) => (
                        <div key={i} className="kpi-card skeleton" style={{ height: 120 }}></div>
                    ))}
                </div>
            </div>
        );
    }

    // Merge stream data into recent events
    const recentEvents = events.length > 0
        ? events.slice(0, 10).map(e => ({
            id: e.id?.substring(0, 12) || 'unknown',
            type: e.value?.event_type || 'unknown',
            user_id: e.value?.user_id || e.key || 'unknown',
            timestamp: e.timestamp || new Date().toISOString(),
            value: e.value?.total || e.value?.price || 0,
        }))
        : stats?.recent_events || [];

    return (
        <div className="page-container">
            {/* Status Bar */}
            <div className="dashboard-status-bar">
                <div className="status-indicator">
                    <div className={`status-dot ${isConnected ? 'connected' : ''}`}></div>
                    <span>Stream: {isConnected ? 'Live' : isStreaming ? 'Connecting...' : 'Disconnected'}</span>
                </div>
                <div className="last-update">
                    <RefreshCw size={14} />
                    Last update: {lastUpdate?.toLocaleTimeString() || 'Never'}
                </div>
            </div>

            {/* KPI Cards */}
            <div className="kpi-grid">
                {stats?.kpis.map((kpi, idx) => (
                    <KPICard key={idx} {...kpi} />
                ))}
            </div>

            {/* Charts */}
            <div className="chart-grid">
                <div className="chart-container">
                    <div className="card-header">
                        <h3 className="card-title">Event Throughput (24h)</h3>
                        {isConnected && <div className="realtime-dot"></div>}
                    </div>
                    <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={stats?.event_throughput}>
                            <defs>
                                <linearGradient id="colorEvents" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                            <XAxis dataKey="timestamp" stroke="#64748b" fontSize={12} />
                            <YAxis stroke="#64748b" fontSize={12} tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`} />
                            <Tooltip
                                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                                labelStyle={{ color: '#f8fafc' }}
                            />
                            <Area type="monotone" dataKey="value" stroke="#3b82f6" fillOpacity={1} fill="url(#colorEvents)" />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>

                <div className="chart-container">
                    <div className="card-header">
                        <h3 className="card-title">Model Accuracy (24h)</h3>
                    </div>
                    <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={stats?.model_accuracy}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                            <XAxis dataKey="timestamp" stroke="#64748b" fontSize={12} />
                            <YAxis stroke="#64748b" fontSize={12} domain={[90, 100]} tickFormatter={(v) => `${v}%`} />
                            <Tooltip
                                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                                labelStyle={{ color: '#f8fafc' }}
                                formatter={(value) => [`${Number(value).toFixed(2)}%`, 'Accuracy']}
                            />
                            <Line type="monotone" dataKey="value" stroke="#10b981" strokeWidth={2} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Bottom Row */}
            <div className="dashboard-bottom">
                <div className="card recent-events">
                    <div className="card-header">
                        <h3 className="card-title">Recent Events {events.length > 0 && `(${events.length} in buffer)`}</h3>
                    </div>
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Event ID</th>
                                <th>Type</th>
                                <th>User</th>
                                <th>Value</th>
                                <th>Time</th>
                            </tr>
                        </thead>
                        <tbody>
                            {recentEvents.map((event, idx) => (
                                <tr key={event.id + idx}>
                                    <td><code>{event.id}</code></td>
                                    <td><StatusBadge status={event.type} /></td>
                                    <td>{event.user_id}</td>
                                    <td>${Number(event.value || 0).toFixed(2)}</td>
                                    <td>{new Date(event.timestamp).toLocaleTimeString()}</td>
                                </tr>
                            ))}
                            {recentEvents.length === 0 && (
                                <tr>
                                    <td colSpan={5} style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                                        No events yet. Start the stream to see live data.
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>

                <div className="card system-status">
                    <div className="card-header">
                        <h3 className="card-title">System Status</h3>
                    </div>
                    <div className="status-list">
                        {Object.entries(stats?.system_status || {}).map(([name, status]) => (
                            <div key={name} className="status-item">
                                <span className="status-name">{name.replace('_', ' ')}</span>
                                <StatusBadge status={status} />
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Dashboard;
