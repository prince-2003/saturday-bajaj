from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog
import uvicorn
import asyncio

# Set up logging early
from app.core.logging import setup_logging
setup_logging()
logger = structlog.get_logger()

# --- Everything above this line runs immediately ---

try:
    logger.info("Loading application settings...")
    from app.core.config import settings
    logger.info("Settings loaded successfully.")
    
    logger.info("Setting up API routers...")
    from app.api.endpoints import hackrx, health
    logger.info("Routers imported successfully.")

except Exception as e:
    logger.error("Failed to import modules or load settings", error=str(e), traceback=True)
    # Re-raise the exception to crash the application,
    # making the error visible in the logs
    raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting up Intelligent Query-Retrieval System lifespan context...")
        
        # Initialize the centralized service container
        logger.info("Initializing service container...")
        from app.core.container import get_service_container
        container = get_service_container()
        
        # Verify all services are healthy
        health_status = await container.health_check()
        unhealthy_services = [name for name, status in health_status.items() if status == "unhealthy"]
        
        if unhealthy_services:
            logger.warning("Some services are unhealthy during startup", unhealthy_services=unhealthy_services)
        else:
            logger.info("All services initialized and healthy")
        
        # This is where your code should be.
        # Everything after this runs on shutdown.
        yield
        
        logger.info("Shutting down services...")
        await container.shutdown()
        logger.info("Shutdown complete")
        
    except Exception as e:
        logger.error("Lifespan startup failed", error=str(e), traceback=True)
        # Re-raise to ensure the app fails to start
        raise e

app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Advanced document analysis and query system for insurance, legal, HR, and compliance domains",
    version="1.0.0",
    lifespan=lifespan
)

# Configure longer timeouts to prevent 499 errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add timeout middleware
@app.middleware("http")
async def timeout_middleware(request, call_next):
    try:
        # Set a reasonable timeout for all requests
        response = await asyncio.wait_for(call_next(request), timeout=300.0)  # 5 minutes
        return response
    except asyncio.TimeoutError:
        logger.error("Request timeout", path=request.url.path)
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=504,
            content={"error": "Request timeout - processing took longer than 5 minutes"}
        )
    except Exception as e:
        logger.error("Middleware error", error=str(e))
        raise

app.include_router(hackrx.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "LLM-Powered Intelligent Query-Retrieval System",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health():
    """Top-level health check endpoint for container orchestrators and Render."""
    return {
        "status": "healthy",
        "service": "intelligent-query-retrieval",
        "version": "1.0.0"
    }

