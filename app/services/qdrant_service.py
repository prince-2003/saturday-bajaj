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

    async def collection_exists(self, collection_name: str = None) -> bool:
        """Check if collection exists."""
        if not self.client:
            logger.error("Qdrant client not initialized")
            return False
            
        collection_name = collection_name or settings.qdrant_collection_name

        try:
            collections_response = await self.client.get_collections()
            existing_collections = [col.name for col in collections_response.collections]
            exists = collection_name in existing_collections
            
            logger.debug("Collection existence check", 
                        collection=collection_name, 
                        exists=exists)
            return exists
            
        except Exception as e:
            logger.error("Failed to check collection existence", 
                        error=str(e), 
                        collection=collection_name)
            return False

    async def recreate_collection_with_correct_dimensions(self, collection_name: str = None) -> bool:
        """Recreate collection with correct embedding dimensions (3072 for text-embedding-3-large)."""
        if not self.client:
            logger.error("Qdrant client not initialized")
            return False
            
        collection_name = collection_name or settings.qdrant_collection_name
        
        try:
            # Delete existing collection if it exists
            collections_response = await self.client.get_collections()
            existing_collections = [col.name for col in collections_response.collections]
            
            if collection_name in existing_collections:
                logger.info("Deleting existing collection with wrong dimensions", collection=collection_name)
                await self.client.delete_collection(collection_name=collection_name)
            
            # Create new collection with correct dimensions
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,  # Should now be 3072
                    distance=Distance.COSINE
                )
            )
            logger.info("Recreated Qdrant collection with correct dimensions", 
                       collection=collection_name, 
                       dimension=settings.embedding_dimension)
            
            # Create payload index
            await self.create_payload_index(collection_name, "document_id", "keyword")
            
            return True
            
        except Exception as e:
            logger.error("Failed to recreate collection", error=str(e), collection=collection_name)
            return False

    async def store_embeddings(self, chunks: List[EmbeddingChunk], collection_name: str = None) -> bool:
        """Store embeddings in Qdrant with batch splitting and retry logic."""
        if not self.client:
            logger.error("Qdrant client not initialized")
            return False
            
        collection_name = collection_name or settings.qdrant_collection_name
        
        try:
            # Check if collection exists and has correct dimensions
            collections_response = await self.client.get_collections()
            existing_collections = [col.name for col in collections_response.collections]
            
            if collection_name in existing_collections:
                # Check collection info to verify dimensions
                collection_info = await self.client.get_collection(collection_name)
                current_dimension = collection_info.config.params.vectors.size
                expected_dimension = settings.embedding_dimension
                
                if current_dimension != expected_dimension:
                    logger.warning("Collection dimension mismatch", 
                                 current=current_dimension, 
                                 expected=expected_dimension,
                                 collection=collection_name)
                    # Recreate collection with correct dimensions
                    await self.recreate_collection_with_correct_dimensions(collection_name)
                else:
                    logger.info("Collection dimensions correct", dimension=current_dimension)
            else:
                # Create new collection
                await self.create_collection(collection_name)
            
            # Convert chunks to points
            points = []
            for i, chunk in enumerate(chunks):
                logger.debug("Processing chunk for storage", 
                           chunk_index=i, 
                           chunk_id=getattr(chunk, 'chunk_id', 'unknown'),
                           has_embedding=hasattr(chunk, 'embedding') and chunk.embedding is not None,
                           embedding_length=len(chunk.embedding) if hasattr(chunk, 'embedding') and chunk.embedding else 0,
                           has_metadata=hasattr(chunk, 'metadata'))
                
                if chunk.embedding and len(chunk.embedding) > 0:
                    # Use a stable, unique ID for each point
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
                    logger.debug("Created point for chunk", point_id=point_id, chunk_id=chunk.chunk_id)
                else:
                    logger.warning("Chunk has no valid embedding", 
                                 chunk_id=getattr(chunk, 'chunk_id', 'unknown'),
                                 embedding_status=f"embedding={'None' if not hasattr(chunk, 'embedding') else 'empty' if not chunk.embedding else 'length=' + str(len(chunk.embedding))}")

            if not points:
                logger.error("No valid embeddings to store", 
                           total_chunks=len(chunks),
                           chunks_without_embeddings=sum(1 for c in chunks if not hasattr(c, 'embedding') or not c.embedding or len(c.embedding) == 0))
                return False

            # Split large batches into smaller ones for better reliability
            max_batch_size = 100  # Qdrant handles smaller batches better
            total_stored = 0
            
            for i in range(0, len(points), max_batch_size):
                batch_points = points[i:i + max_batch_size]
                batch_num = i // max_batch_size + 1
                total_batches = (len(points) + max_batch_size - 1) // max_batch_size
                
                logger.info(f"Storing batch {batch_num}/{total_batches}", 
                           batch_size=len(batch_points),
                           collection=collection_name)
                
                # Try to store this batch with retries
                if await self._store_batch_with_retry(batch_points, collection_name):
                    total_stored += len(batch_points)
                    logger.info(f"Batch {batch_num} stored successfully", stored_count=len(batch_points))
                else:
                    logger.error(f"Failed to store batch {batch_num}")
                    # Continue with other batches instead of failing completely
            
            success_rate = (total_stored / len(points)) * 100 if points else 0
            logger.info("Batch storage completed", 
                       total_points=len(points),
                       stored_points=total_stored,
                       success_rate=f"{success_rate:.1f}%")
            
            return total_stored > 0  # Return True if at least some points were stored
                
        except Exception as e:
            logger.error("Failed to store embeddings - general error", 
                        error=str(e), 
                        error_type=type(e).__name__,
                        collection=collection_name,
                        chunk_count=len(chunks))
            import traceback
            logger.error("Full traceback", traceback=traceback.format_exc())
            return False
    
    async def _store_batch_with_retry(self, points: List[PointStruct], collection_name: str) -> bool:
        """Store a batch of points with retry logic."""
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                result = await self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True
                )
                logger.debug("Batch upsert successful", 
                           result=str(result)[:100], 
                           collection=collection_name,
                           attempt=attempt + 1,
                           point_count=len(points))
                return True
                
            except Exception as upsert_error:
                logger.warning("Batch upsert failed", 
                             error=str(upsert_error)[:300], 
                             error_type=type(upsert_error).__name__,
                             attempt=attempt + 1,
                             max_retries=max_retries,
                             point_count=len(points))
                
                if attempt == max_retries - 1:
                    # Last attempt failed
                    logger.error("All retry attempts failed for batch")
                    return False
                else:
                    # Wait before retry with exponential backoff
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5
        
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
            # Import the correct PayloadSchemaType from qdrant_client
            from qdrant_client.models import PayloadSchemaType
            
            collection_info = await self.client.get_collection(collection_name=collection_name)
            
            # Check if the index already exists in the payload schema
            existing_indexes = collection_info.payload_schema or {}
            if field_name not in existing_indexes:
                # Use the correct PayloadSchemaType enum
                schema_type = PayloadSchemaType.KEYWORD if field_schema.lower() == "keyword" else PayloadSchemaType.INTEGER
                
                await self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema_type
                )
                logger.info("Created payload index", field=field_name, collection=collection_name, schema=field_schema)
            else:
                logger.info("Payload index already exists", field=field_name, collection=collection_name)
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
            # Ensure collection and index exist before checking
            await self.create_collection(collection_name)
            
            # Add a small delay to ensure index is ready
            import asyncio
            await asyncio.sleep(0.1)
            
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
            logger.info("Checked document existence in Qdrant", document_id=document_id, exists=exists, count=count_result.count)
            return exists

        except Exception as e:
            # If index doesn't exist yet, the document definitely doesn't exist
            if "Index required" in str(e) or "not found" in str(e):
                logger.info("Index not ready yet, assuming document does not exist", document_id=document_id)
                return False
            
            # For other errors, log and assume document doesn't exist
            logger.warning("Could not check document existence, assuming it does not exist.", error=str(e))
            return False

  