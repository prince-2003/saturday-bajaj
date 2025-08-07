"""
Centralized Service Container for Dependency Injection
This module provides a singleton pattern for managing all service instances.
"""
from typing import Optional, Dict, Any, List, TYPE_CHECKING
import structlog
from threading import Lock

if TYPE_CHECKING:
    from app.services.retrieval_service import RetrievalService

from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.optimized_llm_service import OptimizedLLMService
from app.services.cache_service import IntelligentCacheService
from app.services.database_service import DatabaseService
from app.services.qdrant_service import QdrantService

logger = structlog.get_logger(__name__)

class ServiceContainer:
    """
    Singleton service container that manages all service instances.
    Ensures services are initialized once and reused throughout the application.
    """
    
    _instance: Optional['ServiceContainer'] = None
    _lock = Lock()
    _initialized = False
    
    def __new__(cls) -> 'ServiceContainer':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ServiceContainer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Prevent re-initialization
        if self._initialized:
            return
            
        logger.info("Initializing ServiceContainer...")
        
        # Initialize services in dependency order
        self._services: Dict[str, Any] = {}
        self._initialize_services()
        
        ServiceContainer._initialized = True
        logger.info("ServiceContainer initialized successfully")
    
    def _initialize_services(self):
        """Initialize all services and wire up circular dependencies."""
        try:
            logger.info("Initializing core services...")
            
            # --- Phase 1: Initialize all services ---
            self._services['document_processor'] = DocumentProcessor()
            self._services['qdrant_service'] = QdrantService()
            self._services['database_service'] = DatabaseService()
            self._services['cache_service'] = IntelligentCacheService()
            self._services['embedding_service'] = EmbeddingService(
                qdrant_service=self._services['qdrant_service'],
                cache_service=self._services['cache_service']
            )
            self._services['llm_service'] = OptimizedLLMService(self._services['cache_service'])
            
            # --- Phase 2: Inject circular dependencies (THE FIX) ---
            # Now that embedding_service is created, give it to the cache_service
            # This is the crucial missing step.
            self._services['cache_service'].set_embedding_service(self._services['embedding_service'])
            logger.info("Circular dependency wired: CacheService -> EmbeddingService")

            # --- Phase 3: Initialize the top-level retrieval service ---
            from app.services.retrieval_service import RetrievalService
            self._services['retrieval_service'] = RetrievalService(
                document_processor=self._services['document_processor'],
                embedding_service=self._services['embedding_service'],
                llm_service=self._services['llm_service'],
                cache_service=self._services['cache_service'],
                database_service=self._services['database_service']
            )
            
            logger.info("All services initialized and wired successfully")
                       
        except Exception as e:
            logger.error("Failed to initialize services", error=str(e))
            raise
    
    # Service getters - provide easy access to services
    @property
    def document_processor(self) -> DocumentProcessor:
        """Get the document processor service instance."""
        return self._services['document_processor']
    
    @property
    def embedding_service(self) -> EmbeddingService:
        """Get the embedding service instance."""
        return self._services['embedding_service']
    
    @property
    def llm_service(self) -> OptimizedLLMService:
        """Get the LLM service instance."""
        return self._services['llm_service']
    
    @property
    def cache_service(self) -> IntelligentCacheService:
        """Get the cache service instance."""
        return self._services['cache_service']
    
    @property
    def database_service(self) -> DatabaseService:
        """Get the database service instance."""
        return self._services['database_service']
    
    @property
    def qdrant_service(self) -> QdrantService:
        """Get the Qdrant service instance."""
        return self._services['qdrant_service']
    
    @property
    def retrieval_service(self) -> 'RetrievalService':
        """Get the retrieval service instance."""
        return self._services['retrieval_service']
    
    def get_service(self, service_name: str) -> Any:
        """
        Get a service by name.
        
        Args:
            service_name: Name of the service to retrieve
            
        Returns:
            The service instance
            
        Raises:
            KeyError: If service not found
        """
        if service_name not in self._services:
            available_services = list(self._services.keys())
            raise KeyError(f"Service '{service_name}' not found. Available services: {available_services}")
        
        return self._services[service_name]
    
    def list_services(self) -> List[str]:
        """Get a list of all available service names."""
        return list(self._services.keys())
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health checks on all services.
        
        Returns:
            Dictionary with health status of each service
        """
        health_status = {
            'container_status': 'healthy',
            'services': {}
        }
        
        for service_name, service in self._services.items():
            try:
                # Check if service has a health_check method
                if hasattr(service, 'health_check'):
                    health_status['services'][service_name] = await service.health_check()
                else:
                    # Basic check - service exists and is initialized
                    health_status['services'][service_name] = {
                        'status': 'healthy',
                        'initialized': True,
                        'type': type(service).__name__
                    }
            except Exception as e:
                health_status['services'][service_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_status['container_status'] = 'degraded'
        
        return health_status
    
    async def shutdown(self):
        """Gracefully shutdown all services."""
        logger.info("Shutting down ServiceContainer...")
        
        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'shutdown'):
                    await service.shutdown()
                    logger.info(f"Service {service_name} shut down successfully")
            except Exception as e:
                logger.error(f"Error shutting down service {service_name}", error=str(e))
        
        logger.info("ServiceContainer shutdown complete")


# Global service container instance
_service_container: Optional[ServiceContainer] = None
_container_lock = Lock()

def get_service_container() -> ServiceContainer:
    """
    Get the global service container instance.
    Thread-safe singleton access.
    
    Returns:
        ServiceContainer instance
    """
    global _service_container
    
    if _service_container is None:
        with _container_lock:
            if _service_container is None:
                _service_container = ServiceContainer()
    
    return _service_container

def get_service(service_name: str) -> Any:
    """
    Convenience function to get a service directly.
    
    Args:
        service_name: Name of the service
        
    Returns:
        The service instance
    """
    container = get_service_container()
    return container.get_service(service_name)

# Convenience functions for commonly used services
def get_retrieval_service() -> 'RetrievalService':
    """Get the retrieval service instance."""
    return get_service_container().retrieval_service

def get_document_processor() -> DocumentProcessor:
    """Get the document processor service instance."""
    return get_service_container().document_processor

def get_embedding_service() -> EmbeddingService:
    """Get the embedding service instance."""
    return get_service_container().embedding_service

def get_llm_service() -> OptimizedLLMService:
    """Get the LLM service instance."""
    return get_service_container().llm_service

def get_cache_service() -> IntelligentCacheService:
    """Get the cache service instance."""
    return get_service_container().cache_service

def get_database_service() -> DatabaseService:
    """Get the database service instance."""
    return get_service_container().database_service

def get_qdrant_service() -> QdrantService:
    """Get the Qdrant service instance."""
    return get_service_container().qdrant_service
