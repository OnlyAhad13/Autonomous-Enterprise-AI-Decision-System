"""
Web Application Backend - Main FastAPI Application.

Unified API for the Autonomous Enterprise AI System dashboard.
"""

import os
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Web Application Backend...")
    yield
    logger.info("Shutting down Web Application Backend...")


# Create FastAPI app
app = FastAPI(
    title="Enterprise AI Dashboard API",
    description="Backend API for the Autonomous Enterprise AI System",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health Check
# ============================================================================

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }


# ============================================================================
# Import and Include Routers
# ============================================================================

from webapp.routers import dashboard, ingestion, monitoring, models, predictions, agent, notifications

app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(ingestion.router, prefix="/api/ingestion", tags=["Ingestion"])
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["Monitoring"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(agent.router, prefix="/api/agent", tags=["Agent"])
app.include_router(notifications.router, prefix="/api/notifications", tags=["Notifications"])


# ============================================================================
# Root
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Enterprise AI Dashboard API",
        "docs": "/api/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
