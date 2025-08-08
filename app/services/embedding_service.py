import asyncio
from typing import List, Dict, Any, Optional, Tuple
import structlog
import numpy as np
from openai import AsyncOpenAI

from app.core.config import settings
from app.models.schemas import EmbeddingChunk
from app.services.qdrant_service import QdrantService
from app.services.cache_service import IntelligentCacheService

logger = structlog.get_logger(__name__)

class EmbeddingService:
    def _get_embedding_token_limit(self):
        # Set token limits based on model name
        model = settings.openai_embedding_model
        if "large" in model:
            return 32768
        return 8192

    def _truncate_text_to_token_limit(self, text: str, max_tokens: int) -> str:
        # Simple whitespace split for now; can use tiktoken for more accuracy
        tokens = text.split()
        if len(tokens) > max_tokens:
            return " ".join(tokens[:max_tokens])
        return text
    """Manages embeddings operations using OpenAI."""
    
    def __init__(self, qdrant_service: QdrantService = None, cache_service: IntelligentCacheService = None):
        self.openai_client = None
        self.qdrant_service = qdrant_service or QdrantService()
        self.cache_service = cache_service or IntelligentCacheService()
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize OpenAI client."""
        try:
            # Initialize OpenAI client
            if settings.openai_api_key:
                self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
                logger.info("OpenAI client initialized")
            else:
                logger.warning("OpenAI API key not provided")
        
        except Exception as e:
            logger.error("Failed to initialize clients", error=str(e))
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts, with caching."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        try:
            embeddings_map = {}
            texts_to_embed = []
            token_limit = self._get_embedding_token_limit()
            # First, check the cache for each text
            for text in texts:
                cached_embedding = await self.cache_service.get_embedding_cache(text)
                if cached_embedding:
                    embeddings_map[text] = cached_embedding
                else:
                    # Truncate text to fit token limit
                    safe_text = self._truncate_text_to_token_limit(text, token_limit)
                    texts_to_embed.append(safe_text)
            logger.info("Embedding cache check", hits=len(embeddings_map), misses=len(texts_to_embed))
            # If there are any texts that were not in the cache, embed them in a batch
            if texts_to_embed:
                # ⚡ ULTRA-FAST: Remove rate limiting delays, rely on OpenAI's built-in handling
                try:
                    logger.info("Generating embeddings at max speed", count=len(texts_to_embed))
                    response = await self.openai_client.embeddings.create(
                        model=settings.openai_embedding_model,
                        input=texts_to_embed
                    )
                    new_embeddings = [embedding.embedding for embedding in response.data]
                    
                    # Add new embeddings to the map and set them in the cache
                    for orig_text, embedding in zip(texts_to_embed, new_embeddings):
                        embeddings_map[orig_text] = embedding
                        await self.cache_service.set_embedding_cache(orig_text, embedding)
                    
                    logger.info("Generated embeddings at max speed", count=len(texts_to_embed))
                    
                except Exception as api_error:
                    error_str = str(api_error).lower()
                    if "rate limit" in error_str or "too many requests" in error_str:
                        logger.warning("Rate limit hit, using exponential backoff")
                        # Only retry on rate limits with minimal delay
                        await asyncio.sleep(2.0)  # 2 second delay only
                        response = await self.openai_client.embeddings.create(
                            model=settings.openai_embedding_model,
                            input=texts_to_embed
                        )
                        new_embeddings = [embedding.embedding for embedding in response.data]
                        for orig_text, embedding in zip(texts_to_embed, new_embeddings):
                            embeddings_map[orig_text] = embedding
                            await self.cache_service.set_embedding_cache(orig_text, embedding)
                        logger.info("Generated embeddings after rate limit", count=len(texts_to_embed))
                    else:
                        logger.error("Embedding API error", error=str(api_error))
                        raise
            # Return the embeddings in the original order
            final_embeddings = [embeddings_map[self._truncate_text_to_token_limit(text, token_limit)] for text in texts]
            return final_embeddings
        except Exception as e:
            logger.error("Failed to generate embeddings", error=str(e))
            raise
    
    async def store_embeddings(self, chunks: List[EmbeddingChunk]) -> bool:
        """Store embeddings in Qdrant."""
        try:
            logger.info("Processing embeddings for storage", chunk_count=len(chunks))
            
            # Extract texts from chunks
            texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings
            embeddings = await self.generate_embeddings(texts)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            # Store in Qdrant
            success = await self.qdrant_service.store_embeddings(chunks)
            
            if success:
                logger.info("Embeddings stored successfully in Qdrant")
            else:
                logger.error("Failed to store embeddings in Qdrant")
            
            return success
            
        except Exception as e:
            logger.error("Failed to process embeddings", error=str(e))
            return False

    
    async def search_similar(self, query: str, top_k: int = 10, 
                           document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar content using Qdrant."""
        try:
            logger.info("Searching for similar content", query=query[:100], top_k=top_k)
            
            # Generate query embedding
            query_embeddings = await self.generate_embeddings([query])
            query_embedding = query_embeddings[0]
            
            # Search in Qdrant
            results = await self.qdrant_service.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                document_id=document_id
            )
            
            logger.info("Similar content search completed", results_count=len(results))
            return results
            
        except Exception as e:
            logger.error("Failed to search similar content", error=str(e))
            return []