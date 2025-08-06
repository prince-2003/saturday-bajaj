import asyncio
import uuid # Import uuid
from typing import List, Dict, Any, Optional
import structlog
from qdrant_client import AsyncQdrantClient, QdrantClient # Import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, ScoredPoint, PayloadSchemaType

from app.core.config import settings
from app.models.schemas import EmbeddingChunk

logger = structlog.get_logger(__name__)

class QdrantService:
    """Service for managing Qdrant vector database operations asynchronously."""
    
    def __init__(self):
        # The client will be initialized as AsyncQdrantClient
        self.client: Optional[AsyncQdrantClient] = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize AsyncQdrantClient."""
        try:
            if settings.qdrant_url and settings.qdrant_api_key:
                # Use cloud Qdrant with URL and API key
                self.client = AsyncQdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key,
                )
                logger.info("Qdrant async cloud client initialized", url=settings.qdrant_url)
            elif settings.qdrant_host:
                # Use local Qdrant instance
                self.client = AsyncQdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                )
                logger.info("Qdrant async local client initialized", host=settings.qdrant_host, port=settings.qdrant_port)
            else:
                logger.warning("Qdrant configuration not provided. Service will be inactive.")
                
        except Exception as e:
            logger.error("Failed to initialize Qdrant async client", error=str(e))
    
    # In your QdrantService class

    async def create_collection(self, collection_name: str = None) -> bool:
        """Create collection if it doesn't exist AND ensure indexes are set."""
        if not self.client:
            logger.error("Qdrant client not initialized")
            return False
            
        collection_name = collection_name or settings.qdrant_collection_name

        try:
            collections_response = await self.client.get_collections()
            existing_collections = [col.name for col in collections_response.collections]
            
            if collection_name not in existing_collections:
                # Step 1: Create the collection
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Created Qdrant collection", collection=collection_name)
                
                # Step 2: IMMEDIATELY create the necessary index after creating the collection
                logger.info("Creating payload index for 'document_id' on new collection.")
                await self.create_payload_index(collection_name, "document_id", "keyword")

            # Optional: You could even add a check here to ensure the index exists on existing collections,
            # but the primary fix is to create it with the collection.
            
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
            
            points = []
            for i, chunk in enumerate(chunks):
                logger.info("Processing chunk for storage", 
                           chunk_index=i, 
                           chunk_id=getattr(chunk, 'chunk_id', 'unknown'),
                           has_embedding=hasattr(chunk, 'embedding') and chunk.embedding is not None,
                           has_metadata=hasattr(chunk, 'metadata'))
                
                if chunk.embedding:
                    # FIX: Use a stable, unique ID for each point. UUID is a great choice.
                    point_id = str(uuid.uuid4())
                    point = PointStruct(
                        id=point_id,
                        vector=chunk.embedding,
                        payload={
                            "text": chunk.text,
                            "document_id": chunk.document_id,
                            "chunk_id": chunk.chunk_id,
                            "metadata": chunk.metadata or {}
                        }
                    )
                    points.append(point)
                    logger.info("Created point for chunk", point_id=point_id, chunk_id=chunk.chunk_id)
                else:
                    logger.warning("Chunk has no embedding", chunk_id=getattr(chunk, 'chunk_id', 'unknown'))
            
            if points:
                logger.info("Upserting points to Qdrant", point_count=len(points), collection=collection_name)
                await self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True # wait for the operation to complete
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
                logger.info("Using document filter", document_id=document_id)
            else:
                logger.info("Searching without document filter")
            
            # Perform search with await and the corrected filter parameter
            search_results: List[ScoredPoint] = await self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=search_filter, # FIX: Pass the filter here
                limit=top_k,
                with_payload=True
            )
            
            logger.info("Raw search results", raw_count=len(search_results))
            
            results = [
                {
                    "text": result.payload.get("text", ""),
                    "document_id": result.payload.get("document_id"),
                    "chunk_id": result.payload.get("chunk_id"),
                    "score": result.score,
                    "metadata": result.payload.get("metadata", {})
                } for result in search_results
            ]
            
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
            delete_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
            
            await self.client.delete(
                collection_name=collection_name,
                points_selector=delete_filter
            )
            
            logger.info("Deleted document from Qdrant", document_id=document_id, collection=collection_name)
            return True
            
        except Exception as e:
            logger.error("Failed to delete document", error=str(e), document_id=document_id)
            return False
            
   # In qdrant_service.py

    async def create_payload_index(self, collection_name: str, field_name: str, field_schema: str = "keyword"):
        """Create an index for a payload field if it doesn't exist."""
        if not self.client:
            logger.error("Qdrant client not initialized")
            return
            
        try:
            collection_info = await self.client.get_collection(collection_name=collection_name)
            # Check if the index already exists in the payload schema
            if field_name not in (collection_info.payload_schema or {}):
                await self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    #
                    # === THE FIX IS HERE ===
                    #
                    # BEFORE (WRONG): field_schema=PayloadSchemaType(field_schema.upper())
                    #
                    # AFTER (CORRECT): Pass the string directly
                    field_schema=field_schema
                )
                logger.info("Created payload index", field=field_name, collection=collection_name)
        except Exception as e:
            logger.warning(f"Could not ensure payload index for '{field_name}'", error=str(e))
            
    async def health_check(self) -> str:
        """Check if Qdrant service is healthy."""
        if not self.client:
            return "unhealthy - client not initialized"
        
        try:
            # Try to get collections to test connection
            collections_response = await self.client.get_collections()
            logger.info("Qdrant health check passed", collections_count=len(collections_response.collections))
            return "healthy"
        except Exception as e:
            logger.error("Qdrant health check failed", error=str(e))
            return f"unhealthy - {str(e)}"

    async def reset_qdrant(self, collection_name: str = None) -> bool:
        """Reset Qdrant collection by deleting and recreating it."""
        if not self.client:
            logger.error("Qdrant client not initialized")
            return False
            
        collection_name = collection_name or settings.qdrant_collection_name
        
        try:
            # Check if collection exists and delete it
            collections_response = await self.client.get_collections()
            existing_collections = [col.name for col in collections_response.collections]
            
            if collection_name in existing_collections:
                await self.client.delete_collection(collection_name=collection_name)
                logger.info("Deleted existing collection", collection=collection_name)
            
            # Recreate the collection
            success = await self.create_collection(collection_name)
            if success:
                logger.info("Reset Qdrant collection successfully", collection=collection_name)
            return success
            
        except Exception as e:
            logger.error("Failed to reset Qdrant collection", error=str(e), collection=collection_name)
            return False
        
  

    async def document_exists(self, document_id: str, collection_name: str = None) -> bool:
        """
        Check if a document with the given ID already exists in the collection.
        """
        if not self.client:
            logger.error("Qdrant client not initialized")
            return False

        collection_name = collection_name or settings.qdrant_collection_name

        try:
            # The count API is the most efficient way to check for existence
            count_result = await self.client.count(
                collection_name=collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(key="document_id", match=MatchValue(value=document_id))
                    ]
                ),
                exact=False  # Use `exact=False` for a faster, approximate count on large datasets
            )

            exists = count_result.count > 0
            logger.info("Checked document existence in Qdrant", document_id=document_id, exists=exists)
            return exists

        except Exception as e:
            # This can happen if the collection doesn't exist yet, which is fine on the first run.
            logger.warning("Could not check document existence, assuming it does not exist.", error=str(e))
            return False

  