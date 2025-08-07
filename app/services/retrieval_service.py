import asyncio
import time
import gc
from typing import List, Dict, Any, Optional, Tuple
import structlog
import hashlib
import psutil

from app.services.embedding_service import EmbeddingService
from app.services.optimized_llm_service import OptimizedLLMService
from app.services.cache_service import IntelligentCacheService
from app.services.database_service import DatabaseService
from app.services.qdrant_service import QdrantService
from app.models.schemas import QueryRequest, QueryResponse, DocumentMetadata, EmbeddingChunk

logger = structlog.get_logger(__name__)

class RetrievalService:
    """Memory-optimized retrieval service with streaming processing."""
    
    def __init__(self, 
                 document_processor = None,
                 embedding_service: EmbeddingService = None,
                 llm_service: OptimizedLLMService = None,
                 cache_service: IntelligentCacheService = None,
                 database_service: DatabaseService = None):
        
        self.embedding_service = embedding_service or EmbeddingService()
        self.cache_service = cache_service or IntelligentCacheService()
        self.database_service = database_service or DatabaseService()
        self.llm_service = llm_service or OptimizedLLMService(self.cache_service)
        
        # Use injected document processor or create new one
        if document_processor:
            self.document_processor = document_processor
        else:
            from app.services.document_processor import DocumentProcessor
            self.document_processor = DocumentProcessor()
        
        # Memory management settings
        self.max_memory_threshold = 0.85  # 85% of available memory
        self.chunk_processing_batch_size = 20  # Process chunks in batches
        self.embedding_batch_size = 10  # Embed in smaller batches

    def _check_memory_and_cleanup(self) -> bool:
        """Check memory usage and perform cleanup if needed."""
        try:
            memory_percent = psutil.virtual_memory().percent / 100
            if memory_percent > self.max_memory_threshold:
                logger.warning("High memory usage, performing cleanup", 
                             memory_percent=f"{memory_percent*100:.1f}%")
                gc.collect()
                return False
            return True
        except Exception:
            return True

    async def process_query_streaming(self, request: QueryRequest) -> QueryResponse:
        """Process query with streaming document processing to minimize memory usage."""
        start_time = time.time()
        document_url = str(request.documents)
        document_id = hashlib.md5(document_url.encode()).hexdigest()[:12]

        try:
            logger.info("Starting memory-optimized query processing", 
                        document_url=document_url,
                        document_id=document_id,
                        question_count=len(request.questions))

            # Step 1: Check if document already exists in Qdrant
            is_processed = await self.embedding_service.qdrant_service.document_exists(document_id=document_id)
            
            if not is_processed:
                logger.info("Document not in Qdrant, starting streaming processing", document_id=document_id)
                
                # Process document with streaming to avoid memory buildup
                metadata, chunk_generator = await self.document_processor.process_document_streaming(document_url)
                
                # Process and store chunks in batches to control memory usage
                total_chunks_processed = 0
                for chunk_batch in chunk_generator:
                    await self._process_and_store_chunk_batch(chunk_batch, document_id)
                    total_chunks_processed += len(chunk_batch)
                    
                    # Memory check after each batch
                    if not self._check_memory_and_cleanup():
                        await asyncio.sleep(0.2)  # Brief pause for memory recovery
                
                metadata.total_chunks = total_chunks_processed
                logger.info("Streaming document processing completed", 
                          document_id=document_id, 
                          total_chunks=total_chunks_processed)
            else:
                logger.info("Document found in Qdrant, skipping processing", document_id=document_id)
                metadata = DocumentMetadata(
                    document_id=document_id, 
                    document_type="pdf", 
                    total_pages=None, 
                    total_chunks=None, 
                    processing_time=0
                )

            # Step 2: Batch process all questions efficiently 
            all_questions = list(request.questions)
            logger.info("Generating embeddings for all questions in batch", count=len(all_questions))
            
            # Generate question embeddings in batch (more efficient than individual calls)
            question_embeddings = await self.embedding_service.generate_embeddings(all_questions)
            embedding_map = {question: emb for question, emb in zip(all_questions, question_embeddings)}

            # Step 3: Process questions concurrently with memory management
            semaphore = asyncio.Semaphore(5)  # Limit concurrent questions
            
            async def process_single_question_optimized(i, question, doc_id):
                async with semaphore:
                    # Check memory before processing each question
                    if not self._check_memory_and_cleanup():
                        await asyncio.sleep(0.1)
                    
                    logger.info("Processing question", index=i+1, question=question[:100])

                    # Use pre-generated embedding
                    query_embedding = embedding_map[question]
                    
                    # Search for similar content
                    context_chunks = await self.embedding_service.qdrant_service.search_similar(
                        query_embedding=query_embedding, 
                        top_k=7, 
                        document_id=doc_id
                    )
                    
                    # Get answer from LLM
                    answer_result = await self.llm_service.answer_question_fast(question, context_chunks, doc_id)
                    
                    return {
                        "answer": answer_result["answer"],
                        "metadata": {
                            "question_index": i,
                            "confidence": answer_result["confidence"],
                            "sources": answer_result["sources"],
                            "reasoning": answer_result["reasoning"],
                            "token_usage": answer_result.get("token_usage", 0)
                        }
                    }

            # Process all questions concurrently but with memory management
            tasks = [process_single_question_optimized(i, question, document_id) 
                    for i, question in enumerate(request.questions)]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results and exceptions
            all_answers = []
            all_metadata = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error("Question processing failed", 
                               question_index=i, 
                               error=str(result))
                    all_answers.append(f"Error processing question: {str(result)}")
                    all_metadata.append({
                        "question_index": i,
                        "confidence": 0.0,
                        "sources": [],
                        "reasoning": "Processing error",
                        "token_usage": 0
                    })
                else:
                    all_answers.append(result["answer"])
                    all_metadata.append(result["metadata"])

            # Calculate final metrics
            processing_time = time.time() - start_time
            total_tokens = sum(meta.get("token_usage", 0) for meta in all_metadata)
            
            logger.info("Memory-optimized query processing completed", 
                        processing_time=processing_time,
                        total_tokens=total_tokens,
                        memory_usage=f"{psutil.virtual_memory().percent:.1f}%")
            
            return QueryResponse(answers=all_answers)
            
        except Exception as e:
            logger.error("Memory-optimized query processing failed", error=str(e), exc_info=True)
            raise ValueError(f"Query processing failed: {str(e)}")

    async def _process_and_store_chunk_batch(self, chunk_batch: List[EmbeddingChunk], document_id: str):
        """Process and store a batch of chunks with memory optimization."""
        try:
            if not chunk_batch:
                return

            logger.info("Processing chunk batch", batch_size=len(chunk_batch), document_id=document_id)
            
            # Generate embeddings for batch
            texts = [chunk.text for chunk in chunk_batch]
            
            # Process embeddings in smaller sub-batches if needed
            if len(texts) > self.embedding_batch_size:
                embeddings = []
                for i in range(0, len(texts), self.embedding_batch_size):
                    sub_batch = texts[i:i + self.embedding_batch_size]
                    sub_embeddings = await self.embedding_service.generate_embeddings(sub_batch)
                    embeddings.extend(sub_embeddings)
                    
                    # Memory check between sub-batches
                    if not self._check_memory_and_cleanup():
                        await asyncio.sleep(0.1)
            else:
                embeddings = await self.embedding_service.generate_embeddings(texts)
            
            # Attach embeddings to chunks
            for chunk, embedding in zip(chunk_batch, embeddings):
                chunk.embedding = embedding
            
            # Store in Qdrant
            success = await self.embedding_service.qdrant_service.store_embeddings(chunk_batch)
            
            if success:
                logger.info("Chunk batch stored successfully", batch_size=len(chunk_batch))
            else:
                logger.error("Failed to store chunk batch", batch_size=len(chunk_batch))
                
        except Exception as e:
            logger.error("Chunk batch processing failed", error=str(e))
            raise

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Main query processing method - routes to streaming version."""
        return await self.process_query_streaming(request)

    async def health_check(self) -> Dict[str, str]:
        """Health check with memory information."""
        health_status = {}
        
        try:
            # Memory information
            memory_info = psutil.virtual_memory()
            health_status["memory_usage"] = f"{memory_info.percent:.1f}%"
            health_status["memory_available"] = f"{memory_info.available // (1024**3):.1f}GB"
            
            # Check services
            if self.embedding_service.openai_client:
                health_status["embedding_service"] = "healthy"
            else:
                health_status["embedding_service"] = "unhealthy"
            
            qdrant_health = await self.embedding_service.qdrant_service.health_check()
            health_status["qdrant"] = qdrant_health
            
            database_health = await self.database_service.health_check()
            health_status["database"] = database_health
            
            llm_health = await self.llm_service.health_check()
            health_status.update(llm_health)
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            health_status["error"] = str(e)
        
        return health_status

    async def clear_all_caches(self):
        """Clear all caches and perform memory cleanup."""
        try:
            if hasattr(self.cache_service, 'clear_cache'):
                await self.cache_service.clear_cache()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Caches cleared and memory cleaned up")
        except Exception as e:
            logger.error("Failed to clear caches", error=str(e))
            raise

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        try:
            memory_info = psutil.virtual_memory()
            return {
                "total_memory_gb": memory_info.total // (1024**3),
                "used_memory_gb": memory_info.used // (1024**3),
                "available_memory_gb": memory_info.available // (1024**3),
                "memory_percent": memory_info.percent,
                "threshold_percent": self.max_memory_threshold * 100
            }
        except Exception as e:
            logger.error("Failed to get memory stats", error=str(e))
            return {"error": str(e)}