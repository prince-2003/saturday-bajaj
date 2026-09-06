import asyncio
import os
import tempfile
from typing import List, Dict, Any, Optional
import structlog
import numpy as np

from app.core.config import settings
from app.models.schemas import EmbeddingChunk
from app.services.qdrant_service import QdrantService
from app.services.cache_service import IntelligentCacheService

logger = structlog.get_logger(__name__)

class EmbeddingService:
    """
    High-Performance Embedding Service using local CPU FastEmbed (ONNX)
    with OpenAI API fallback.
    
    Benefits:
    - 100% local, offline, and zero credit/quota dependency (never 429).
    - Sub-20ms latency per batch.
    - 384 dimensions (BAAI/bge-small-en-v1.5) for fast vector comparison.
    """
    
    def __init__(self, qdrant_service: QdrantService = None, cache_service: IntelligentCacheService = None):
        self.fastembed_model = None
        self.openai_client = None
        self.qdrant_service = qdrant_service or QdrantService()
        self.cache_service = cache_service or IntelligentCacheService()
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize local FastEmbed model or OpenAI client fallback."""
        # 1. Initialize FastEmbed (Primary local engine)
        if getattr(settings, "use_local_embeddings", True):
            try:
                from fastembed import TextEmbedding
                cache_dir = os.getenv("FASTEMBED_CACHE_PATH") or os.path.join(tempfile.gettempdir(), "fastembed_cache")
                model_name = getattr(settings, "fastembed_model", "BAAI/bge-small-en-v1.5")
                
                logger.info("Initializing local FastEmbed engine...", model=model_name, cache_dir=cache_dir)
                self.fastembed_model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
                logger.info("FastEmbed engine initialized successfully (Zero API cost, zero rate limits)")
            except Exception as e:
                logger.warning("Failed to initialize FastEmbed, checking for OpenAI fallback", error=str(e))
                self.fastembed_model = None
        
        # 2. Initialize OpenAI client (Fallback or alternative)
        if settings.openai_api_key:
            try:
                from openai import AsyncOpenAI
                self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
                logger.info("OpenAI client initialized as fallback")
            except Exception as e:
                logger.warning("Failed to initialize OpenAI client", error=str(e))
                self.openai_client = None
                
        if not self.fastembed_model and not self.openai_client:
            logger.error("No embedding providers available! FastEmbed failed and OpenAI key not found.")

    def _truncate_text(self, text: str, max_words: int = 512) -> str:
        words = text.split()
        if len(words) > max_words:
            return " ".join(words[:max_words])
        return text

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate dense embeddings for a list of texts.
        Checks cache first, then utilizes FastEmbed (or OpenAI fallback).
        """
        if not texts:
            return []

        try:
            embeddings_map: Dict[str, List[float]] = {}
            texts_to_embed: List[str] = []
            
            # Step 1: Check embedding cache
            for text in texts:
                cached = await self.cache_service.get_embedding_cache(text)
                if cached:
                    embeddings_map[text] = cached
                else:
                    texts_to_embed.append(text)

            logger.info("Embedding cache lookup", hits=len(embeddings_map), misses=len(texts_to_embed))

            # Step 2: Generate missing embeddings
            if texts_to_embed:
                if self.fastembed_model:
                    # Run FastEmbed ONNX inference in thread pool to not block asyncio
                    safe_texts = [self._truncate_text(t, 512) for t in texts_to_embed]
                    
                    def run_fastembed():
                        # FastEmbed returns a generator of numpy ndarrays
                        generator = self.fastembed_model.embed(safe_texts)
                        return [vec.tolist() for vec in generator]
                    
                    new_embeddings = await asyncio.to_thread(run_fastembed)
                    
                    for orig_text, emb in zip(texts_to_embed, new_embeddings):
                        embeddings_map[orig_text] = emb
                        await self.cache_service.set_embedding_cache(orig_text, emb)
                        
                    logger.info("Generated embeddings locally via FastEmbed", count=len(texts_to_embed))

                elif self.openai_client:
                    logger.info("Generating embeddings via OpenAI fallback", count=len(texts_to_embed))
                    response = await self.openai_client.embeddings.create(
                        model=settings.openai_embedding_model,
                        input=texts_to_embed
                    )
                    new_embeddings = [item.embedding for item in response.data]
                    for orig_text, emb in zip(texts_to_embed, new_embeddings):
                        embeddings_map[orig_text] = emb
                        await self.cache_service.set_embedding_cache(orig_text, emb)
                else:
                    raise ValueError("No embedding engine available (FastEmbed and OpenAI unavailable).")

            # Return in the original input order
            return [embeddings_map[t] for t in texts]

        except Exception as e:
            logger.error("Failed to generate embeddings", error=str(e), exc_info=True)
            raise

    async def store_embeddings(self, chunks: List[EmbeddingChunk], collection_name: str = None) -> bool:
        """Store chunk embeddings in Qdrant."""
        return await self.qdrant_service.store_embeddings(chunks, collection_name)