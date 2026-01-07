import { Search, User } from 'lucide-react';
import NotificationsPanel from './NotificationsPanel';
import './Header.css';

function Header({ title }) {
    return (
        <header className="header">
            <div className="header-left">
                <h1 className="header-title">{title}</h1>
            </div>

            <div className="header-right">
                <div className="search-box">
                    <Search size={18} />
                    <input type="text" placeholder="Search..." />
                </div>

                <NotificationsPanel />

                <div className="user-menu">
                    <div className="user-avatar">
                        <User size={20} />
                    </div>
                    <span className="user-name">Admin</span>
                </div>
            </div>
        </header>
    );
}

export default Header;
