import { useState, useEffect, useRef } from 'react';
import { Bell, X, Check, Trash2, AlertCircle, Info, CheckCircle, FileText } from 'lucide-react';
import { notificationsApi } from '../../api/client';
import './NotificationsPanel.css';

function NotificationsPanel() {
    const [isOpen, setIsOpen] = useState(false);
    const [notifications, setNotifications] = useState([]);
    const [unreadCount, setUnreadCount] = useState(0);
    const panelRef = useRef(null);

    // Fetch notifications
    const fetchNotifications = async () => {
        try {
            const response = await notificationsApi.getAll(false, 50);
            setNotifications(response.data.notifications || []);
            setUnreadCount(response.data.unread_count || 0);
        } catch (error) {
            console.error('Failed to fetch notifications:', error);
        }
    };

    useEffect(() => {
        fetchNotifications();
        // Poll for new notifications every 5 seconds
        const interval = setInterval(fetchNotifications, 5000);
        return () => clearInterval(interval);
    }, []);

    // Close panel when clicking outside
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (panelRef.current && !panelRef.current.contains(event.target)) {
                setIsOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const handleMarkRead = async (id) => {
        await notificationsApi.markRead(id);
        fetchNotifications();
    };

    const handleMarkAllRead = async () => {
        await notificationsApi.markAllRead();
        fetchNotifications();
    };

    const handleDelete = async (id) => {
        await notificationsApi.delete(id);
        fetchNotifications();
    };

    const handleClearAll = async () => {
        await notificationsApi.clearAll();
        fetchNotifications();
    };

    const getIcon = (type) => {
        switch (type) {
            case 'success': return <CheckCircle size={16} className="icon-success" />;
            case 'warning': return <AlertCircle size={16} className="icon-warning" />;
            case 'error': return <AlertCircle size={16} className="icon-error" />;
            case 'report': return <FileText size={16} className="icon-report" />;
            default: return <Info size={16} className="icon-info" />;
        }
    };

    return (
        <div className="notifications-container" ref={panelRef}>
            <button
                className="notification-bell"
                onClick={() => setIsOpen(!isOpen)}
            >
                <Bell size={20} />
                {unreadCount > 0 && (
                    <span className="notification-badge">{unreadCount > 9 ? '9+' : unreadCount}</span>
                )}
            </button>

            {isOpen && (
                <div className="notifications-panel">
                    <div className="panel-header">
                        <h3>Notifications</h3>
                        <div className="panel-actions">
                            {unreadCount > 0 && (
                                <button onClick={handleMarkAllRead} title="Mark all read">
                                    <Check size={14} />
                                </button>
                            )}
                            {notifications.length > 0 && (
                                <button onClick={handleClearAll} title="Clear all">
                                    <Trash2 size={14} />
                                </button>
                            )}
                        </div>
                    </div>

                    <div className="notifications-list">
                        {notifications.length === 0 ? (
                            <div className="no-notifications">
                                <Bell size={32} />
                                <p>No notifications yet</p>
                                <span>Agent reports will appear here</span>
                            </div>
                        ) : (
                            notifications.map((notif) => (
                                <div
                                    key={notif.id}
                                    className={`notification-item ${notif.type} ${notif.read ? 'read' : 'unread'}`}
                                >
                                    <div className="notification-icon">
                                        {getIcon(notif.type)}
                                    </div>
                                    <div className="notification-content">
                                        <div className="notification-title">{notif.title}</div>
                                        <div className="notification-message">{notif.message}</div>
                                        <div className="notification-time">
                                            {new Date(notif.timestamp).toLocaleString()}
                                        </div>
                                    </div>
                                    <div className="notification-actions">
                                        {!notif.read && (
                                            <button onClick={() => handleMarkRead(notif.id)} title="Mark read">
                                                <Check size={14} />
                                            </button>
                                        )}
                                        <button onClick={() => handleDelete(notif.id)} title="Delete">
                                            <X size={14} />
                                        </button>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}

export default NotificationsPanel;
