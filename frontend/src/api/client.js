import axios from 'axios';

const API_BASE_URL = 'http://localhost:8080/api';

const client = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Dashboard API
export const dashboardApi = {
    getStats: () => client.get('/dashboard/stats'),
    getKpis: () => client.get('/dashboard/kpis'),
};

// Ingestion API
export const ingestionApi = {
    getStats: () => client.get('/ingestion/stats'),
    getEvents: (page = 1, pageSize = 20) =>
        client.get(`/ingestion/events?page=${page}&page_size=${pageSize}`),
    getTopics: () => client.get('/ingestion/topics'),
};

// Monitoring API
export const monitoringApi = {
    getOverview: () => client.get('/monitoring/overview'),
    getServices: () => client.get('/monitoring/services'),
    getAlerts: (resolved = null) => {
        const params = resolved !== null ? `?resolved=${resolved}` : '';
        return client.get(`/monitoring/alerts${params}`);
    },
};

// Models API
export const modelsApi = {
    getRegistry: () => client.get('/models/registry'),
    getRuns: (limit = 10) => client.get(`/models/runs?limit=${limit}`),
    train: (modelType = 'random_forest') =>
        client.post(`/models/train?model_type=${modelType}`),
    deploy: (modelName, version = 'latest', stage = 'Production') =>
        client.post(`/models/deploy/${modelName}?version=${version}&stage=${stage}`),
    getDeployed: () => client.get('/models/deployed'),
    predict: (features) => client.post('/models/predict', features),
    getTrainingStatus: () => client.get('/models/training-status'),
    retrain: (modelType = 'random_forest') =>
        client.post(`/models/retrain?model_type=${modelType}`),
    getMetrics: (modelName) => client.get(`/models/${modelName}/metrics`),
};

// Predictions API
export const predictionsApi = {
    predict: (features, modelType = 'prophet') =>
        client.post('/predictions/single', { features, model_type: modelType }),
    getForecast: (startDate, endDate, modelType = 'prophet') =>
        client.get(`/predictions/forecast?start_date=${startDate}&end_date=${endDate}&model_type=${modelType}`),
    explain: (features, modelType = 'prophet') =>
        client.post('/predictions/explain', { features, model_type: modelType }),
    getHistory: (limit = 50) => client.get(`/predictions/history?limit=${limit}`),
};

// Agent API
export const agentApi = {
    getStatus: () => client.get('/agent/status'),
    getTools: () => client.get('/agent/tools'),
    getExecutions: (limit = 10) => client.get(`/agent/executions?limit=${limit}`),
    execute: (objective) => client.post('/agent/execute', null, { params: { objective } }),
    getExecution: (executionId) => client.get(`/agent/executions/${executionId}`),
};

// Notifications API
export const notificationsApi = {
    getAll: (unreadOnly = false, limit = 50) =>
        client.get(`/notifications?unread_only=${unreadOnly}&limit=${limit}`),
    markRead: (notificationId) => client.put(`/notifications/${notificationId}/read`),
    markAllRead: () => client.put('/notifications/read-all'),
    delete: (notificationId) => client.delete(`/notifications/${notificationId}`),
    clearAll: () => client.delete('/notifications'),
};

export default client;
