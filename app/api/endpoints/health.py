from fastapi import APIRouter, Depends
from datetime import datetime
import structlog

from app.core.config import settings
from app.core.security import verify_api_key
from app.models.schemas import HealthResponse
from app.core.container import get_retrieval_service

logger = structlog.get_logger(__name__)
router = APIRouter()

# No need to initialize service here - it's managed by the container

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Public health check endpoint.
    
    Returns:
        Basic health status without requiring authentication
    """
    try:
        # Basic health check
        retrieval_service = get_retrieval_service()
        health_data = await retrieval_service.health_check()
        
        # Determine overall status
        unhealthy_services = [k for k, v in health_data.items() if v == "unhealthy"]
        if unhealthy_services:
            status = "degraded"
        else:
            status = "healthy"
        
        # Format dependencies for response
        dependencies = health_data
        
        return HealthResponse(
            status=status,
            timestamp=datetime.utcnow().isoformat(),
            version=settings.api_version,
            dependencies=dependencies
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            version=settings.api_version,
            dependencies={"error": str(e)}
        )

@router.get("/health/detailed")
async def detailed_health_check(api_key: str = Depends(verify_api_key)) -> dict:
    """
    Detailed health check with authentication required.
    
    Args:
        api_key: API key for authentication
    
    Returns:
        Detailed health information including service status and configuration
    """
    try:
        logger.info("Detailed health check requested")
        
        # Get service health status
        retrieval_service = get_retrieval_service()
        service_health = await retrieval_service.health_check()
        
        # Configuration status (without sensitive values)
        config_status = {
            "openai_configured": bool(settings.openai_api_key),
            "database_configured": bool(settings.database_url),
            "embedding_model": settings.openai_embedding_model,
            "llm_model": settings.openai_model,
            "chunk_size": settings.chunk_size,
            "max_tokens": settings.max_tokens
        }
        
        # System information
        system_info = {
            "api_version": settings.api_version,
            "debug_mode": settings.debug,
            "max_file_size": settings.max_file_size,
            "max_concurrent_requests": settings.max_concurrent_requests,
            "request_timeout": settings.request_timeout
        }
        
        return {
            "status": "healthy" if all(v != "unhealthy" for v in service_health.values()) else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "services": service_health,
            "configuration": config_status,
            "system": system_info
        }
        
    except Exception as e:
        logger.error("Detailed health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@router.get("/metrics")
async def get_metrics(api_key: str = Depends(verify_api_key)) -> dict:
    """
    Get system metrics and performance data.
    
    Args:
        api_key: API key for authentication
    
    Returns:
        System metrics and performance statistics
    """
    try:
        # This would typically connect to a metrics system like Prometheus
        # For now, return basic placeholder metrics
        
        metrics = {
            "requests_total": 0,  # Would be tracked in a real system
            "requests_successful": 0,
            "requests_failed": 0,
            "avg_response_time": 0.0,
            "documents_processed": 0,
            "questions_answered": 0,
            "tokens_consumed": 0,
            "cache_hit_rate": 0.0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return metrics
        
    except Exception as e:
        logger.error("Metrics retrieval failed", error=str(e))
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }