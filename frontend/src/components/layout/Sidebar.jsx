import { NavLink } from 'react-router-dom';
import {
    LayoutDashboard,
    Activity,
    LineChart,
    Brain,
    Target,
    Bot,
    Settings,
    Zap,
} from 'lucide-react';
import './Sidebar.css';

const navItems = [
    { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/ingestion', icon: Activity, label: 'Data Ingestion' },
    { path: '/monitoring', icon: LineChart, label: 'Monitoring' },
    { path: '/models', icon: Brain, label: 'Models' },
    { path: '/predictions', icon: Target, label: 'Predictions' },
    { path: '/agent', icon: Bot, label: 'Agent' },
];

function Sidebar() {
    return (
        <aside className="sidebar">
            <div className="sidebar-header">
                <div className="logo">
                    <Zap className="logo-icon" />
                    <span className="logo-text">Enterprise AI</span>
                </div>
            </div>

            <nav className="sidebar-nav">
                {navItems.map(({ path, icon: Icon, label }) => (
                    <NavLink
                        key={path}
                        to={path}
                        className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
                    >
                        <Icon size={20} />
                        <span>{label}</span>
                    </NavLink>
                ))}
            </nav>

            <div className="sidebar-footer">
                <NavLink to="/settings" className="nav-item">
                    <Settings size={20} />
                    <span>Settings</span>
                </NavLink>
                <div className="version">v1.0.0</div>
            </div>
        </aside>
    );
}

export default Sidebar;
