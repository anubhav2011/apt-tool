"""
Health Check Endpoints
System health and status endpoints
"""
from fastapi import APIRouter, Depends

from app.schemas.models import HealthResponse
from app.core.proctoring_config import ProctoringConfig
from app.core.dependencies import get_config

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(config: ProctoringConfig = Depends(get_config)):
    """
    Health check endpoint
    """
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        components={
            "api": "operational",
            "video_processor": "operational",
            "database": f"{config.MYSQL_DATABASE}_operational",
            "mediapipe": "operational",
            "opencv": "operational"
        }
    )


@router.get("/", response_model=HealthResponse)
async def root(config: ProctoringConfig = Depends(get_config)):
    """
    Root endpoint
    """
    return HealthResponse(
        status="operational",
        version="2.0.0",
        components={
            "api": "healthy",
            "video_processor": "ready",
            "database": f"{config.MYSQL_DATABASE}_configured",
            "detection_engine": "count_based_ready"
        }
    )
