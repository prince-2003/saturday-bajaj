import asyncio
from typing import List, Dict, Any, Optional
import structlog
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from app.core.config import settings
from app.models.schemas import EmbeddingChunk

logger = structlog.get_logger(__name__)

class QdrantService:
    """Service for managing Qdrant vector database operations."""
    
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Qdrant client."""
        try:
            if settings.qdrant_url and settings.qdrant_api_key:
                # Use cloud Qdrant with URL and API key
                self.client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key,
                )
                logger.info("Qdrant cloud client initialized", url=settings.qdrant_url)
            elif settings.qdrant_host:
                # Use local Qdrant instance
                self.client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                )
                logger.info("Qdrant local client initialized", host=settings.qdrant_host, port=settings.qdrant_port)
            else:
                logger.warning("Qdrant configuration not provided")
                
        except Exception as e:
            logger.error("Failed to initialize Qdrant client", error=str(e))
    
    async def create_collection(self, collection_name: str = None) -> bool:
        """Create collection if it doesn't exist."""
        if not self.client:
            logger.error("Qdrant client not initialized")
            return False
            
        collection_name = collection_name or settings.qdrant_collection_name
        
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if collection_name not in existing_collections:
                # Create collection
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Created Qdrant collection", collection=collection_name)
            else:
                logger.info("Qdrant collection already exists", collection=collection_name)
            
            return True
            
        except Exception as e:
            logger.error("Failed to create collection", error=str(e), collection=collection_name)
            return False
    
    async def store_embeddings(self, chunks: List[EmbeddingChunk], collection_name: str = None) -> bool:
        """Store embeddings in Qdrant."""
        if not self.client:
            logger.error("Qdrant client not initialized")
            return False
            
        collection_name = collection_name or settings.qdrant_collection_name
        
        try:
            # Ensure collection exists
            await self.create_collection(collection_name)
            
            # Prepare points for insertion
            points = []
            for i, chunk in enumerate(chunks):
                if chunk.embedding:
                    point = PointStruct(
                        id=i,  # You might want to use a UUID or hash here
                        vector=chunk.embedding,
                        payload={
                            "text": chunk.text,
                            "document_id": chunk.document_id,
                            "chunk_id": chunk.chunk_id,
                            "metadata": chunk.metadata or {}
                        }
                    )
                    points.append(point)
            
            if points:
                # Insert points
                self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                logger.info("Stored embeddings in Qdrant", count=len(points), collection=collection_name)
                return True
            else:
                logger.warning("No valid embeddings to store")
                return False
                
        except Exception as e:
            logger.error("Failed to store embeddings", error=str(e), collection=collection_name)
            return False
    
    async def search_similar(self, query_embedding: List[float], top_k: int = 10, 
                           document_id: Optional[str] = None, collection_name: str = None) -> List[Dict[str, Any]]:
        """Search for similar embeddings in Qdrant."""
        if not self.client:
            logger.error("Qdrant client not initialized")
            return []
            
        collection_name = collection_name or settings.qdrant_collection_name
        
        try:
            # Prepare filter if document_id is provided
            search_filter = None
            if document_id:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            
            # Perform search
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "text": result.payload.get("text", ""),
                    "document_id": result.payload.get("document_id"),
                    "chunk_id": result.payload.get("chunk_id"),
                    "score": result.score,
                    "metadata": result.payload.get("metadata", {})
                })
            
            logger.info("Search completed", results_count=len(results), collection=collection_name)
            return results
            
        except Exception as e:
            logger.error("Failed to search similar embeddings", error=str(e), collection=collection_name)
            return []
    
    async def delete_document(self, document_id: str, collection_name: str = None) -> bool:
        """Delete all chunks for a specific document."""
        if not self.client:
            logger.error("Qdrant client not initialized")
            return False
            
        collection_name = collection_name or settings.qdrant_collection_name
        
        try:
            # Delete points with matching document_id
            delete_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
            
            self.client.delete(
                collection_name=collection_name,
                points_selector=delete_filter
            )
            
            logger.info("Deleted document from Qdrant", document_id=document_id, collection=collection_name)
            return True
            
        except Exception as e:
            logger.error("Failed to delete document", error=str(e), document_id=document_id)
            return False
    
    async def health_check(self) -> str:
        """Check Qdrant service health."""
        if not self.client:
            return "unhealthy"
        
        try:
            # Try to get collections info
            collections = self.client.get_collections()
            return "healthy"
        except Exception as e:
            logger.error("Qdrant health check failed", error=str(e))
            return "unhealthy"
        
    async def create_payload_index_if_missing(self, collection_name: str, field_name: str, field_schema: str = "keyword"):
        """Ensure an index exists for the given payload field."""
        try:
            collection_info = await self.client.get_collection(collection_name)
            existing_indexes = collection_info.payload_schema or {}

            if field_name not in existing_indexes:
                await self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema
                )
        except Exception as e:
            logger.warning(f"Could not ensure index for {field_name}: {str(e)}")

    def reset_qdrant(self):
        
        self.client.delete_collection(collection_name="documents")
        print("Collection deleted.")
