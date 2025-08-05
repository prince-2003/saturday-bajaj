import asyncio
from typing import List, Dict, Any, Optional, Tuple
import structlog
import numpy as np
from openai import AsyncOpenAI

from app.core.config import settings
from app.models.schemas import EmbeddingChunk
from app.services.qdrant_service import QdrantService

logger = structlog.get_logger(__name__)

class EmbeddingService:
    """Manages embeddings operations using OpenAI."""
    
    def __init__(self):
        self.openai_client = None
        self.qdrant_service = QdrantService()
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
        """Generate embeddings for a list of texts."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        try:
            logger.info("Generating embeddings", text_count=len(texts))
            
            response = await self.openai_client.embeddings.create(
                model=settings.openai_embedding_model,
                input=texts
            )
            
            embeddings = [embedding.embedding for embedding in response.data]
            logger.info("Generated embeddings successfully", embedding_count=len(embeddings))
            
            return embeddings
        
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