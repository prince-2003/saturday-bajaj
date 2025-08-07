from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog
import uvicorn

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(hackrx.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "LLM-Powered Intelligent Query-Retrieval System",
        "version": "1.0.0",
        "status": "operational"
    }

