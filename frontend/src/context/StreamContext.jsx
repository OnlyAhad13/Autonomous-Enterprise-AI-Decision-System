import { createContext, useContext, useState, useEffect, useRef } from 'react';

// Create context
const StreamContext = createContext(null);

// Event stream provider
export function StreamProvider({ children }) {
    const [events, setEvents] = useState([]);
    const [isStreaming, setIsStreaming] = useState(false);
    const [isConnected, setIsConnected] = useState(false);
    const eventSourceRef = useRef(null);

    // Connect to stream
    const startStream = () => {
        if (eventSourceRef.current) return; // Already connected

        setIsStreaming(true);

        const eventSource = new EventSource('http://localhost:8080/api/ingestion/stream');
        eventSourceRef.current = eventSource;

        eventSource.onopen = () => {
            console.log('🟢 SSE connected');
            setIsConnected(true);
        };

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                setEvents(prev => [data, ...prev].slice(0, 500)); // Keep 500 events
            } catch (e) {
                console.error('Failed to parse event:', e);
            }
        };

        eventSource.onerror = (error) => {
            console.error('🔴 SSE error:', error);
            setIsConnected(false);
            // Try to reconnect after 3 seconds
            setTimeout(() => {
                if (isStreaming && !eventSourceRef.current) {
                    startStream();
                }
            }, 3000);
        };
    };

    // Disconnect from stream
    const stopStream = () => {
        setIsStreaming(false);
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
        }
        setIsConnected(false);
    };

    // Toggle stream
    const toggleStream = () => {
        if (isStreaming) {
            stopStream();
        } else {
            startStream();
        }
    };

    // Clear events
    const clearEvents = () => {
        setEvents([]);
    };

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
            }
        };
    }, []);

    const value = {
        events,
        isStreaming,
        isConnected,
        startStream,
        stopStream,
        toggleStream,
        clearEvents,
    };

    return (
        <StreamContext.Provider value={value}>
            {children}
        </StreamContext.Provider>
    );
}

// Hook to use stream
export function useStream() {
    const context = useContext(StreamContext);
    if (!context) {
        throw new Error('useStream must be used within a StreamProvider');
    }
    return context;
}

export default StreamContext;
