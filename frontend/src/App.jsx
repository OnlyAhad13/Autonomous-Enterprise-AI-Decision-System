import { Routes, Route } from 'react-router-dom';
import { StreamProvider } from './context/StreamContext';
import Sidebar from './components/layout/Sidebar';
import Header from './components/layout/Header';
import Dashboard from './pages/Dashboard';
import DataIngestion from './pages/DataIngestion';
import Monitoring from './pages/Monitoring';
import Models from './pages/Models';
import Predictions from './pages/Predictions';
import Agent from './pages/Agent';

function App() {
  return (
    <StreamProvider>
      <div className="app-layout">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<><Header title="Dashboard" /><Dashboard /></>} />
            <Route path="/ingestion" element={<><Header title="Data Ingestion" /><DataIngestion /></>} />
            <Route path="/monitoring" element={<><Header title="Monitoring" /><Monitoring /></>} />
            <Route path="/models" element={<><Header title="Models" /><Models /></>} />
            <Route path="/predictions" element={<><Header title="Predictions" /><Predictions /></>} />
            <Route path="/agent" element={<><Header title="Agent" /><Agent /></>} />
          </Routes>
        </main>
      </div>
    </StreamProvider>
  );
}

export default App;
