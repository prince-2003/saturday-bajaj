import asyncio
import json
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import structlog
import redis.asyncio as redis
import numpy as np
from cachetools import TTLCache, LRUCache

from app.core.config import settings
from app.models.schemas import EmbeddingChunk

if TYPE_CHECKING:
    from app.services.embedding_service import EmbeddingService

logger = structlog.get_logger(__name__)

class IntelligentCacheService:
    """Multi-layer intelligent caching system for documents, embeddings, and Q&A."""
    
    def __init__(self):
        self.redis_client = None
        self.memory_cache = LRUCache(maxsize=settings.max_memory_cache_size)
        self.document_cache = TTLCache(maxsize=100, ttl=settings.document_cache_ttl)
        self.embedding_cache = TTLCache(maxsize=500, ttl=settings.embedding_cache_ttl)
        self.qa_cache = TTLCache(maxsize=200, ttl=settings.qa_cache_ttl)
        self.embedding_service: Optional["EmbeddingService"] = None
        
        self._initialize_cache()
    
    def set_embedding_service(self, embedding_service: "EmbeddingService"):
        """Set the embedding service after initialization to avoid circular imports."""
        self.embedding_service = embedding_service
    
    def _initialize_cache(self):
        """Initialize Redis connection."""
        try:
            # Check if using Upstash REST API
            if settings.upstash_redis_rest_url and settings.upstash_redis_rest_token:
                logger.info("Using Upstash Redis REST API")
                # For Upstash, we'll use HTTP client instead of Redis client
                import httpx
                self.upstash_client = httpx.AsyncClient(
                    base_url=settings.upstash_redis_rest_url,
                    headers={"Authorization": f"Bearer {settings.upstash_redis_rest_token}"}
                )
                self.redis_client = None  # Use HTTP instead
                self.use_upstash = True
            elif settings.redis_cluster_mode:
                from rediscluster import RedisCluster
                self.redis_client = RedisCluster(
                    startup_nodes=[{"host": settings.redis_url.split("://")[1].split(":")[0], 
                                   "port": int(settings.redis_url.split(":")[-1])}],
                    decode_responses=True,
                    password=settings.redis_password
                )
                self.use_upstash = False
            else:
                self.redis_client = redis.from_url(
                    settings.redis_url,
                    password=settings.redis_password,
                    db=settings.redis_db,
                    decode_responses=True
                )
                self.use_upstash = False
            
            logger.info("Cache service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize cache service", error=str(e))
            self.redis_client = None
            self.use_upstash = False
    
    async def _upstash_get(self, key: str) -> Optional[str]:
        """Get value from Upstash using REST API."""
        try:
            if not hasattr(self, 'upstash_client'):
                return None
            response = await self.upstash_client.post("/get", json=[key])
            if response.status_code == 200:
                result = response.json()
                return result.get("result")
            return None
        except Exception as e:
            logger.error("Upstash GET failed", key=key, error=str(e))
            return None
    
    async def _upstash_set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in Upstash using REST API."""
        try:
            if not hasattr(self, 'upstash_client'):
                return False
            
            if ttl:
                # Set with expiration
                response = await self.upstash_client.post("/setex", json=[key, ttl, value])
            else:
                response = await self.upstash_client.post("/set", json=[key, value])
            
            return response.status_code == 200
        except Exception as e:
            logger.error("Upstash SET failed", key=key, error=str(e))
            return False
    
    def _generate_cache_key(self, prefix: str, *args) -> str:
        """Generate consistent cache key."""
        key_data = "|".join(str(arg) for arg in args)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    async def get_document_cache(self, document_url: str) -> Optional[Tuple[Any, List[EmbeddingChunk]]]:
        """Get cached document processing results."""
        try:
            cache_key = self._generate_cache_key("doc", document_url)
            
            # Check memory cache first
            if cache_key in self.document_cache:
                logger.info("Document cache hit (memory)", url=document_url)
                return self.document_cache[cache_key]
            
            # Check external cache (Redis or Upstash)
            cached_data = None
            if hasattr(self, 'use_upstash') and self.use_upstash:
                cached_data = await self._upstash_get(cache_key)
            elif self.redis_client:
                cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                result = (data["metadata"], [EmbeddingChunk(**chunk) for chunk in data["chunks"]])
                self.document_cache[cache_key] = result
                cache_type = "Upstash" if hasattr(self, 'use_upstash') and self.use_upstash else "Redis"
                logger.info(f"Document cache hit ({cache_type})", url=document_url)
                return result
            
            logger.info("Document cache miss", url=document_url)
            return None
            
        except Exception as e:
            logger.error("Error accessing document cache", error=str(e))
            return None
    
    async def set_document_cache(self, document_url: str, metadata: Any, chunks: List[EmbeddingChunk]):
        """Cache document processing results."""
        try:
            cache_key = self._generate_cache_key("doc", document_url)
            
            # Prepare data for caching
            cache_data = {
                "metadata": metadata.dict() if hasattr(metadata, 'dict') else metadata,
                "chunks": [chunk.dict() for chunk in chunks],
                "timestamp": time.time()
            }
            
            # Store in memory cache
            self.document_cache[cache_key] = (metadata, chunks)
            
            # Store in external cache (Redis or Upstash)
            if hasattr(self, 'use_upstash') and self.use_upstash:
                await self._upstash_set(
                    cache_key,
                    json.dumps(cache_data, default=str),
                    settings.document_cache_ttl
                )
            elif self.redis_client:
                await self.redis_client.setex(
                    cache_key,
                    settings.document_cache_ttl,
                    json.dumps(cache_data, default=str)
                )
            
            logger.info("Document cached successfully", url=document_url, chunks=len(chunks))
            
        except Exception as e:
            logger.error("Error caching document", error=str(e))
    
    async def get_embedding_cache(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text."""
        try:
            cache_key = self._generate_cache_key("emb", text)
            
            # Check memory cache first
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]
            
            # Check Redis cache
            if self.redis_client:
                cached_embedding = await self.redis_client.get(cache_key)
                if cached_embedding:
                    embedding = json.loads(cached_embedding)
                    self.embedding_cache[cache_key] = embedding
                    return embedding
            
            return None
            
        except Exception as e:
            logger.error("Error accessing embedding cache", error=str(e))
            return None
    
    async def set_embedding_cache(self, text: str, embedding: List[float]):
        """Cache embedding for text."""
        try:
            cache_key = self._generate_cache_key("emb", text)
            
            # Store in memory cache
            self.embedding_cache[cache_key] = embedding
            
            # Store in Redis cache
            if self.redis_client:
                await self.redis_client.setex(
                    cache_key,
                    settings.embedding_cache_ttl,
                    json.dumps(embedding)
                )
            
        except Exception as e:
            logger.error("Error caching embedding", error=str(e))
    
    async def get_qa_cache(self, question: str, document_id: str) -> Optional[Dict[str, Any]]:
        """Get cached Q&A result with exact and semantic matching, using a two-layer cache."""
        try:
            cache_key = self._generate_cache_key("qa", question, document_id)
            
            # --- FIX: Check in-memory cache first ---
            if cache_key in self.qa_cache:
                logger.info("Q&A cache hit (exact in-memory)", question=question[:50])
                return self.qa_cache[cache_key]

            # 2. Exact match check in Redis
            if self.redis_client:
                cached_answer = await self.redis_client.get(cache_key)
                if cached_answer:
                    logger.info("Q&A cache hit (exact Redis)", question=question[:50])
                    answer = json.loads(cached_answer)
                    # Populate the in-memory cache for the next request
                    self.qa_cache[cache_key] = answer
                    return answer

            # 3. Semantic similarity check (if no exact match is found)
            if self.embedding_service:
                similar_answer = await self._check_semantic_cache(question, document_id)
                if similar_answer:
                    logger.info("Q&A cache hit (semantic)", question=question[:50])
                    # Populate both caches for the next time this exact question is asked
                    await self.set_qa_cache(question, document_id, similar_answer)
                    return similar_answer
            
            logger.info("Q&A cache miss", question=question[:50])
            return None
        except Exception as e:
            logger.error("Error accessing Q&A cache", error=str(e))
            return None
    
    async def set_qa_cache(self, question: str, document_id: str, answer_data: Dict[str, Any]):
        """Cache Q&A result for both exact and semantic lookups in both cache layers."""
        try:
            cache_key = self._generate_cache_key("qa", question, document_id)
            
            # --- FIX: Set in-memory cache ---
            self.qa_cache[cache_key] = answer_data

            # 2. Set exact match in Redis
            if self.redis_client:
                await self.redis_client.setex(
                    cache_key,
                    settings.qa_cache_ttl,
                    json.dumps(answer_data, default=str)
                )
            
            # 3. Add to semantic cache (this part is correct)
            if self.embedding_service:
                await self._add_to_semantic_cache(question, document_id, answer_data)
            
            logger.info("Q&A cached successfully", question=question[:50])
        except Exception as e:
            logger.error("Error caching Q&A", error=str(e))
    
    async def get_vector_cache(self, query_vector: List[float], document_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached vector search results."""
        try:
            # Create a hash of the query vector for caching
            vector_hash = hashlib.md5(
                json.dumps(query_vector, sort_keys=True).encode()
            ).hexdigest()
            
            cache_key = self._generate_cache_key("vec", vector_hash, document_id)
            
            if self.redis_client:
                cached_results = await self.redis_client.get(cache_key)
                if cached_results:
                    return json.loads(cached_results)
            
            return None
            
        except Exception as e:
            logger.error("Error accessing vector cache", error=str(e))
            return None
    
    async def set_vector_cache(self, query_vector: List[float], document_id: str, results: List[Dict[str, Any]]):
        """Cache vector search results."""
        try:
            vector_hash = hashlib.md5(
                json.dumps(query_vector, sort_keys=True).encode()
            ).hexdigest()
            
            cache_key = self._generate_cache_key("vec", vector_hash, document_id)
            
            if self.redis_client:
                await self.redis_client.setex(
                    cache_key,
                    settings.cache_ttl,
                    json.dumps(results, default=str)
                )
            
        except Exception as e:
            logger.error("Error caching vector results", error=str(e))
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        stats = {
            "memory_cache_size": len(self.memory_cache),
            "document_cache_size": len(self.document_cache),
            "embedding_cache_size": len(self.embedding_cache),
            "qa_cache_size": len(self.qa_cache),
            "redis_connected": self.redis_client is not None
        }
        
        if self.redis_client:
            try:
                info = await self.redis_client.info()
                stats["redis_memory_usage"] = info.get("used_memory_human", "Unknown")
                stats["redis_connected_clients"] = info.get("connected_clients", 0)
            except Exception as e:
                logger.error("Error getting Redis stats", error=str(e))
        
        return stats
    
    async def clear_cache(self, cache_type: Optional[str] = None):
        """Clear specific or all caches."""
        try:
            if cache_type == "memory" or cache_type is None:
                self.memory_cache.clear()
                self.document_cache.clear()
                self.embedding_cache.clear()
                self.qa_cache.clear()
            
            if cache_type == "redis" or cache_type is None:
                if self.redis_client:
                    await self.redis_client.flushdb()
            
            logger.info("Cache cleared", cache_type=cache_type or "all")
            
        except Exception as e:
            logger.error("Error clearing cache", error=str(e))
    
    async def warm_cache(self, documents: List[str]):
        """Pre-warm cache with frequently accessed documents."""
        logger.info("Starting cache warm-up", documents=len(documents))
        
        # This would be implemented to pre-process common documents
        # and store them in cache for faster access
        
        for doc_url in documents:
            try:
                # Check if document is already cached
                cached = await self.get_document_cache(doc_url)
                if not cached:
                    # Process and cache the document
                    # This would integrate with the document processor
                    logger.info("Pre-processing document for cache", url=doc_url)
                    # Implementation would go here
                    
            except Exception as e:
                logger.error("Error warming cache for document", url=doc_url, error=str(e))
        
        logger.info("Cache warm-up completed")

    async def _check_semantic_cache(self, question: str, document_id: str) -> Optional[Dict[str, Any]]:
        """Check for semantically similar cached questions using OpenAI embeddings."""
        try:
            # Get question embedding from OpenAI via the EmbeddingService
            embeddings = await self.embedding_service.generate_embeddings([question])
            if not embeddings: return None
            question_embedding = np.array(embeddings[0])
            
            semantic_key = f"semantic:{document_id}"
            if self.redis_client:
                cached_questions = await self.redis_client.hgetall(semantic_key)
                for cached_q, cached_data in cached_questions.items():
                    cached_info = json.loads(cached_data)
                    cached_embedding = np.array(cached_info["embedding"])
                    
                    similarity = np.dot(question_embedding, cached_embedding) / (np.linalg.norm(question_embedding) * np.linalg.norm(cached_embedding))
                    
                    if similarity >= settings.semantic_cache_threshold:
                        logger.info("Semantic cache match found", similarity=round(similarity, 2), original_question=cached_q[:50])
                        return cached_info["answer"]
            return None
        except Exception as e:
            logger.error("Error checking semantic cache", error=str(e))
            return None

    async def _add_to_semantic_cache(self, question: str, document_id: str, answer_data: Dict[str, Any]):
        """Add question-answer pair to semantic cache using OpenAI embeddings."""
        try:
            # Generate question embedding from OpenAI via the EmbeddingService
            embeddings = await self.embedding_service.generate_embeddings([question])
            if not embeddings: return
            question_embedding = embeddings[0]
            
            semantic_data = {"embedding": question_embedding, "answer": answer_data}
            
            semantic_key = f"semantic:{document_id}"
            if self.redis_client:
                await self.redis_client.hset(semantic_key, question, json.dumps(semantic_data, default=str))
                await self.redis_client.expire(semantic_key, settings.qa_cache_ttl)
        except Exception as e:
            logger.error("Error adding to semantic cache", error=str(e))