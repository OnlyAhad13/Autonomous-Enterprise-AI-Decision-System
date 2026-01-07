import { useState, useEffect } from 'react';
import { Radio, Pause, Play, RefreshCw, Wifi, WifiOff } from 'lucide-react';
import { useStream } from '../context/StreamContext';
import { ingestionApi } from '../api/client';
import './DataIngestion.css';

function DataIngestion() {
    const { events, isStreaming, isConnected, toggleStream, clearEvents } = useStream();
    const [stats, setStats] = useState(null);

    useEffect(() => {
        // Fetch initial stats
        const fetchStats = async () => {
            try {
                const response = await ingestionApi.getStats();
                setStats(response.data);
            } catch (error) {
                console.error('Failed to fetch stats:', error);
            }
        };
        fetchStats();
        const interval = setInterval(fetchStats, 5000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="page-container">
            {/* Stats Row */}
            <div className="ingestion-stats">
                <div className="stat-card">
                    <div className="stat-value">{stats?.events_per_second?.toFixed(1) || events.length / 60 || '—'}</div>
                    <div className="stat-label">Events/Second</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value">{events.length}</div>
                    <div className="stat-label">Events in Buffer</div>
                </div>
                <div className="stat-card">
                    <div className="stat-value">{stats?.partitions || '3'}</div>
                    <div className="stat-label">Partitions</div>
                </div>
                <div className={`stat-card ${isConnected ? 'connected' : 'disconnected'}`}>
                    <div className="stat-value">
                        {isConnected ? <Wifi size={24} /> : <WifiOff size={24} />}
                    </div>
                    <div className="stat-label">Stream {isConnected ? 'Connected' : 'Disconnected'}</div>
                </div>
            </div>

            {/* Stream Controls */}
            <div className="card stream-card">
                <div className="card-header">
                    <div className="stream-header">
                        <h3 className="card-title">
                            <Radio size={18} />
                            Live Event Stream
                        </h3>
                        {isConnected && <div className="realtime-dot"></div>}
                        {isStreaming && !isConnected && <span className="connecting">Connecting...</span>}
                    </div>
                    <div className="stream-controls">
                        <button
                            className={`btn ${isStreaming ? 'btn-danger' : 'btn-success'}`}
                            onClick={toggleStream}
                        >
                            {isStreaming ? <><Pause size={16} /> Stop</> : <><Play size={16} /> Start Stream</>}
                        </button>
                        <button className="btn btn-secondary" onClick={clearEvents}>
                            <RefreshCw size={16} /> Clear
                        </button>
                    </div>
                </div>

                <div className="event-stream">
                    {events.slice(0, 50).map((event, idx) => (
                        <div key={event.id || idx} className="event-item">
                            <div className="event-meta">
                                <span className="event-id">{(event.id || 'N/A').substring(0, 12)}...</span>
                                <span className="event-time">
                                    {event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : 'N/A'}
                                </span>
                            </div>
                            <div className="event-details">
                                <span className={`status-badge ${event.value?.event_type || 'unknown'}`}>
                                    {event.value?.event_type || 'unknown'}
                                </span>
                                <span className="event-user">{event.value?.user_id || event.key || 'N/A'}</span>
                                <span className="event-product">{event.value?.product_id || 'N/A'}</span>
                                <span className="event-price">
                                    ${event.value?.total || event.value?.price || 0}
                                </span>
                            </div>
                            <div className="event-partition">
                                P{event.partition} @ {event.offset}
                            </div>
                        </div>
                    ))}
                    {events.length === 0 && (
                        <div className="no-events">
                            {isStreaming ? 'Waiting for events from Kafka...' : 'Click "Start Stream" to begin receiving events'}
                        </div>
                    )}
                </div>

                {events.length > 50 && (
                    <div className="buffer-info">
                        Showing 50 of {events.length} events in buffer
                    </div>
                )}
            </div>

            {/* Topics */}
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">Kafka Topics</h3>
                </div>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Topic Name</th>
                            <th>Partitions</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><code>events.raw.v1</code></td>
                            <td>3</td>
                            <td><span className={`status-badge ${isConnected ? 'healthy' : 'warning'}`}>{isConnected ? 'Active' : 'Unknown'}</span></td>
                        </tr>
                        <tr>
                            <td><code>events.canonical.v1</code></td>
                            <td>3</td>
                            <td><span className="status-badge healthy">Active</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    );
}

export default DataIngestion;
