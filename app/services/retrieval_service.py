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
            from app.services.document_processor import OptimizedDocumentProcessor
            self.document_processor = OptimizedDocumentProcessor()
        
        # 🚀 PRODUCTION OPTIMIZED: 800MB memory available - USE FULL 800MB!
        self.max_memory_threshold = 1.00  # 100% of 800MB = 800MB fully utilized
        self.chunk_processing_batch_size = 10000  # MASSIVE: 10,000 chunks per batch (doubled from 5000)
        self.embedding_batch_size = 250  # MAXIMUM: 250 embeddings per API call (increased from 150)
        self.max_concurrent_questions = 50  # MAXIMUM: 50 concurrent questions (doubled from 25)
        self.memory_check_frequency = 50  # Check even less frequently for speed (was 20)
        
        # Ultra-fast caching
        self._collection_exists_cache = None
        
        logger.info("PRODUCTION mode initialized - MAXIMUM 800MB USAGE", 
                   memory_threshold=f"{self.max_memory_threshold*100:.0f}%",
                   memory_available="800MB - FULL UTILIZATION",
                   chunk_batch_size=self.chunk_processing_batch_size,
                   embedding_batch_size=self.embedding_batch_size,
                   concurrent_questions=self.max_concurrent_questions)

    def _check_memory_and_cleanup(self) -> bool:
        """Ultra-fast memory check with aggressive cleanup."""
        try:
            memory_percent = psutil.virtual_memory().percent / 100
            if memory_percent > self.max_memory_threshold:
                logger.warning("High memory usage, performing ultra-fast cleanup", 
                             memory_percent=f"{memory_percent*100:.1f}%")
                # ✅ Aggressive cleanup for speed
                gc.collect()
                gc.collect()  # Double collection for better cleanup
                import ctypes
                ctypes.CDLL('msvcrt').malloc_trim(0)  # Force memory release on Windows
                return False
            return True
        except Exception:
            return True

    async def process_query_ultra_fast(self, request: QueryRequest) -> QueryResponse:
        """
        ULTRA-FAST processing specifically designed to prevent client timeouts.
        
        Optimizations:
        - Skip all unnecessary checks and delays
        - Maximum parallel processing
        - Minimal logging
        - Fast-fail on errors
        """
        document_url = str(request.documents)
        document_id = hashlib.md5(document_url.encode()).hexdigest()[:12]

        try:
            # ⚡ STEP 1: Lightning-fast document check
            collection_exists = await asyncio.wait_for(
                self.embedding_service.qdrant_service.collection_exists(),
                timeout=10.0
            )
            if not collection_exists:
                await asyncio.wait_for(
                    self.embedding_service.qdrant_service.create_collection(),
                    timeout=30.0
                )

            is_processed = await asyncio.wait_for(
                self.embedding_service.qdrant_service.document_exists(document_id=document_id),
                timeout=10.0
            )
            
            if not is_processed:
                # ⚡ STEP 2: ULTRA-FAST document processing with timeout
                document_task = asyncio.wait_for(
                    self.document_processor.process_document_streaming(document_url),
                    timeout=180.0  # 3 minutes max for document processing
                )
                metadata, chunk_generator = await document_task
                
                # 🚀 NEW STAGED PROCESSING: Process everything in memory first
                store_task = asyncio.wait_for(
                    self._process_and_store_all_chunks_staged(chunk_generator, document_id),
                    timeout=300.0  # 5 minutes for entire staged processing
                )
                total_stored = await store_task
                logger.info(f"Staged processing completed: {total_stored} chunks processed")

            # ⚡ STEP 3: MAXIMUM SPEED question processing with timeout
            # Generate ALL embeddings at once
            embedding_task = asyncio.wait_for(
                self.embedding_service.generate_embeddings(list(request.questions)),
                timeout=60.0  # 1 minute for all embeddings
            )
            question_embeddings = await embedding_task
            
            # Execute ALL searches in parallel with timeout
            search_tasks = []
            for i, (question, embedding) in enumerate(zip(request.questions, question_embeddings)):
                search_task = asyncio.wait_for(
                    self.embedding_service.qdrant_service.search_similar(
                        query_embedding=embedding, 
                        top_k=7, 
                        document_id=document_id
                    ),
                    timeout=10.0  # 10 seconds per search
                )
                search_tasks.append((i, question, search_task))
            
            # Get all search results
            search_results = await asyncio.gather(*[task for _, _, task in search_tasks], return_exceptions=True)
            
            # Execute ALL LLM calls in parallel with timeout
            questions_data = []
            for (i, question, _), search_result in zip(search_tasks, search_results):
                if isinstance(search_result, Exception):
                    search_result = []
                questions_data.append({
                    "question": question,
                    "context_chunks": search_result,
                    "index": i
                })
            
            # Process all questions simultaneously with timeout
            llm_task = asyncio.wait_for(
                self.llm_service.answer_multiple_questions_batch(questions_data, document_id),
                timeout=120.0  # 2 minutes for all LLM calls
            )
            results = await llm_task
            
            # Extract answers
            all_answers = [result["answer"] for result in results]
            
            return QueryResponse(answers=all_answers)
            
        except Exception as e:
            logger.error("Ultra-fast processing failed", error=str(e))
            # Fast-fail with minimal error response
            error_answers = [f"Processing error: {str(e)}" for _ in request.questions]
            return QueryResponse(answers=error_answers)

    async def process_query_streaming(self, request: QueryRequest) -> QueryResponse:
        """Ultra-fast query processing under 60 seconds."""
        start_time = time.time()
        document_url = str(request.documents)
        document_id = hashlib.md5(document_url.encode()).hexdigest()[:12]

        try:
            logger.info("ULTRA-FAST processing started", 
                        target_time="<60s",
                        document_url=document_url,
                        document_id=document_id,
                        question_count=len(request.questions))

            # ✅ Step 0: Fast collection check (cached)
            if self._collection_exists_cache is None:
                self._collection_exists_cache = await self.embedding_service.qdrant_service.collection_exists()
                if not self._collection_exists_cache:
                    await self.embedding_service.qdrant_service.create_collection()
                    self._collection_exists_cache = True

            # ✅ Step 1: Fast document existence check
            is_processed = await self.embedding_service.qdrant_service.document_exists(document_id=document_id)
            
            if is_processed:
                logger.info("Document found in Qdrant, ultra-fast path", document_id=document_id)
                metadata = DocumentMetadata(
                    document_id=document_id, 
                    document_type="pdf",
                    total_pages=None, 
                    total_chunks=None, 
                    processing_time=0
                )
            else:
                logger.info("Document processing with ULTRA-FAST mode", document_id=document_id)
                
                # ⚡ Use ultra-fast document processing
                metadata, chunk_generator = await self.document_processor.process_document_streaming(document_url)
                
                # Process with minimal memory checks for speed
                total_chunks_processed = 0
                batch_count = 0
                
                async for chunk_batch in chunk_generator:
                    await self._process_and_store_chunk_batch_ultra_fast(chunk_batch, document_id)
                    total_chunks_processed += len(chunk_batch)
                    batch_count += 1
                    
                    # ✅ Minimal memory checking for speed - every 2 batches only
                    if batch_count % 2 == 0:
                        if not self._check_memory_and_cleanup():
                            await asyncio.sleep(0.1)  # Shorter pause
                
                metadata.total_chunks = total_chunks_processed
                logger.info("Ultra-fast document processing completed", 
                          document_id=document_id, 
                          total_chunks=total_chunks_processed,
                          batches_processed=batch_count)

            # ✅ Step 2: ULTRA-PARALLEL question processing - ALL AT ONCE
            all_questions = list(request.questions)
            logger.info("ULTRA-PARALLEL question processing", count=len(all_questions))
            
            # ⚡ BATCH 1: Generate ALL question embeddings in one go
            question_embeddings = await self.embedding_service.generate_embeddings(all_questions)
            embedding_map = {question: emb for question, emb in zip(all_questions, question_embeddings)}

            # ⚡ BATCH 2: Get ALL search results in parallel (no semaphore limit!)
            search_tasks = []
            for question in all_questions:
                query_embedding = embedding_map[question]
                search_task = self.embedding_service.qdrant_service.search_similar(
                    query_embedding=query_embedding, 
                    top_k=7, 
                    document_id=document_id
                )
                search_tasks.append(search_task)
            
            logger.info("Executing parallel searches", count=len(search_tasks))
            all_search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # ⚡ BATCH 3: Process ALL questions using batch method
            questions_data = []
            for i, (question, search_result) in enumerate(zip(all_questions, all_search_results)):
                if isinstance(search_result, Exception):
                    logger.error("Search failed for question", question_index=i, error=str(search_result))
                    search_result = []  # Use empty context
                
                questions_data.append({
                    "question": question,
                    "context_chunks": search_result,
                    "index": i
                })
            
            logger.info("Using batch LLM processing for maximum speed", count=len(questions_data))
            results = await self.llm_service.answer_multiple_questions_batch(questions_data, document_id)
            
            # Handle results - extract answers and metadata
            all_answers = []
            all_metadata = []
            
            for result in results:
                all_answers.append(result["answer"])
                all_metadata.append({
                    "question_index": result.get("question_index", 0),
                    "confidence": result.get("confidence", 0.0),
                    "sources": result.get("sources", []),
                    "reasoning": result.get("reasoning", ""),
                    "token_usage": result.get("token_usage", 0)
                })

            # Calculate final metrics
            processing_time = time.time() - start_time
            total_tokens = sum(meta.get("token_usage", 0) for meta in all_metadata)
            
            logger.info("ULTRA-FAST processing completed", 
                        time_taken=f"{processing_time:.1f}s",
                        target_met=processing_time < 60.0,
                        total_tokens=total_tokens,
                        memory_usage=f"{psutil.virtual_memory().percent:.1f}%",
                        total_chunks=total_chunks_processed if not is_processed else "cached")
            
            return QueryResponse(answers=all_answers)
            
        except Exception as e:
            logger.error("Ultra-fast processing failed", error=str(e), exc_info=True)
            raise ValueError(f"Ultra-fast query processing failed: {str(e)}")

    async def _process_and_store_all_chunks_staged(self, chunk_generator, document_id: str):
        """
        STAGED PROCESSING: Process all chunks in memory first, then embeddings, then storage.
        Uses available memory (800MB) more efficiently for maximum speed.
        """
        logger.info("Starting STAGED processing - all chunks in memory first")
        start_memory_mb = psutil.virtual_memory().used // (1024 * 1024)
        logger.info(f"Starting memory usage: {start_memory_mb}MB")
        
        # 🚀 STAGE 1: Collect ALL chunks in memory first
        all_chunks = []
        chunk_count = 0
        
        logger.info("STAGE 1: Processing all chunks into memory...")
        async for chunk_batch in chunk_generator:
            all_chunks.extend(chunk_batch)
            chunk_count += len(chunk_batch)
            
            # Log progress every 1000 chunks
            if chunk_count % 1000 == 0:
                memory_mb = psutil.virtual_memory().used // (1024 * 1024)
                logger.info(f"Processed {chunk_count} chunks, memory: {memory_mb}MB")
        
        total_chunks = len(all_chunks)
        stage1_memory_mb = psutil.virtual_memory().used // (1024 * 1024)
        logger.info(f"STAGE 1 COMPLETE: {total_chunks} chunks in memory, memory: {stage1_memory_mb}MB")
        
        # 🚀 STAGE 2: Generate ALL embeddings at once
        logger.info("STAGE 2: Generating ALL embeddings...")
        texts = [chunk.text for chunk in all_chunks]
        
        # Process in large batches but keep everything in memory
        all_embeddings = await self.embedding_service.generate_embeddings(texts)
        
        # Validate embeddings were generated
        valid_embeddings = [e for e in all_embeddings if e is not None and len(e) > 0]
        logger.info(f"Embedding validation: {len(valid_embeddings)}/{len(all_embeddings)} valid embeddings")
        
        # Attach embeddings to chunks
        for chunk, embedding in zip(all_chunks, all_embeddings):
            chunk.embedding = embedding
            
        stage2_memory_mb = psutil.virtual_memory().used // (1024 * 1024)
        logger.info(f"STAGE 2 COMPLETE: {len(all_embeddings)} embeddings generated, memory: {stage2_memory_mb}MB")
        
        # Clear text array to free memory
        del texts, all_embeddings
        
        # 🚀 STAGE 3: Store ALL chunks in vector DB with concurrent batches
        logger.info("STAGE 3: Storing all chunks in vector DB with concurrent processing...")
        
        # Store in mega-batches for speed with concurrent processing
        storage_batch_size = 1000  # Store 1000 chunks at once
        stored_count = 0
        
        # Create concurrent storage tasks
        storage_tasks = []
        for i in range(0, total_chunks, storage_batch_size):
            batch = all_chunks[i:i + storage_batch_size]
            task = self.embedding_service.qdrant_service.store_embeddings(batch)
            storage_tasks.append((i, len(batch), task))
        
        # Execute all storage tasks concurrently
        logger.info(f"Executing {len(storage_tasks)} concurrent storage batches...")
        results = await asyncio.gather(*[task for _, _, task in storage_tasks], return_exceptions=True)
        
        # Count successful storage
        for (batch_idx, batch_size, _), result in zip(storage_tasks, results):
            if isinstance(result, Exception):
                logger.error(f"Storage batch {batch_idx//storage_batch_size + 1} failed: {str(result)}")
            elif result:
                stored_count += batch_size
                logger.info(f"Storage batch {batch_idx//storage_batch_size + 1} succeeded: {batch_size} chunks")
            else:
                # Investigate why storage returned False
                batch = all_chunks[batch_idx:batch_idx + storage_batch_size]
                chunks_with_embeddings = sum(1 for chunk in batch if chunk.embedding is not None and len(chunk.embedding) > 0)
                logger.error(f"Storage batch {batch_idx//storage_batch_size + 1} returned False: {chunks_with_embeddings}/{len(batch)} chunks had valid embeddings")
        
        stage3_memory_mb = psutil.virtual_memory().used // (1024 * 1024)
        logger.info(f"STAGE 3 COMPLETE: {stored_count}/{total_chunks} chunks stored, memory: {stage3_memory_mb}MB")
        
        # 🚀 STAGE 4: Clear all memory
        logger.info("STAGE 4: Clearing memory...")
        del all_chunks
        import gc
        gc.collect()
        
        final_memory_mb = psutil.virtual_memory().used // (1024 * 1024)
        logger.info(f"STAGED PROCESSING COMPLETE: Memory cleared, now using {final_memory_mb}MB")
        
        return stored_count

    async def _process_and_store_chunk_batch_ultra_fast(self, chunk_batch: List[EmbeddingChunk], document_id: str):
        """Ultra-fast chunk processing with mega-batches."""
        if not chunk_batch:
            return

        logger.info("Ultra-fast chunk processing", batch_size=len(chunk_batch))
        
        # ⚡ MEGA-BATCH: Process up to 150 chunks per API call (safe for token limits)
        mega_batch_size = self.embedding_batch_size  # 150 chunks per API call
        texts = [chunk.text for chunk in chunk_batch]
        
        # Use asyncio.gather for concurrent API calls
        if len(texts) > mega_batch_size:
            embedding_tasks = []
            for i in range(0, len(texts), mega_batch_size):
                sub_batch = texts[i:i + mega_batch_size]
                task = self.embedding_service.generate_embeddings(sub_batch)
                embedding_tasks.append(task)
            
            # ⚡ Concurrent API calls instead of sequential
            logger.info(f"Making {len(embedding_tasks)} concurrent API calls")
            embedding_results = await asyncio.gather(*embedding_tasks)
            embeddings = []
            for result in embedding_results:
                embeddings.extend(result)
        else:
            embeddings = await self.embedding_service.generate_embeddings(texts)
        
        # Attach embeddings
        for chunk, embedding in zip(chunk_batch, embeddings):
            chunk.embedding = embedding
        
        # ⚡ Bulk storage
        success = await self.embedding_service.qdrant_service.store_embeddings(chunk_batch)
        
        if success:
            logger.info("Ultra-fast batch completed", 
                       batch_size=len(chunk_batch),
                       memory_usage=f"{psutil.virtual_memory().percent:.1f}%")
        else:
            logger.error("Failed to store ultra-fast batch", batch_size=len(chunk_batch))

    async def _process_and_store_chunk_batch(self, chunk_batch: List[EmbeddingChunk], document_id: str):
        """Process and store a batch of chunks with optimized performance."""
        try:
            if not chunk_batch:
                return

            logger.info("Processing chunk batch", batch_size=len(chunk_batch), document_id=document_id)
            
            # Generate embeddings for batch with larger sub-batches for speed
            texts = [chunk.text for chunk in chunk_batch]
            
            # Use larger embedding batches for fewer API calls
            if len(texts) > self.embedding_batch_size:
                embeddings = []
                api_calls = 0
                for i in range(0, len(texts), self.embedding_batch_size):
                    sub_batch = texts[i:i + self.embedding_batch_size]
                    sub_embeddings = await self.embedding_service.generate_embeddings(sub_batch)
                    embeddings.extend(sub_embeddings)
                    api_calls += 1
                    
                    # Quick memory check - less frequent for speed
                    if api_calls % 3 == 0 and not self._check_memory_and_cleanup():
                        await asyncio.sleep(0.05)  # Shorter pause for speed
                
                logger.info(f"Generated embeddings with {api_calls} API calls", 
                           chunks=len(texts), batch_size=self.embedding_batch_size)
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
            
            # Immediate cleanup for speed
            del texts, embeddings
            gc.collect()
                
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

    async def _optimize_for_speed(self):
        """Optimize system components for maximum speed."""
        try:
            # Pre-warm connections
            if hasattr(self.embedding_service, 'openai_client'):
                logger.info("Pre-warming OpenAI connections for speed")
                # Pre-generate a small embedding to warm up the connection
                await self.embedding_service.generate_embeddings(["warmup"])
            
            # Pre-warm Qdrant connection
            if hasattr(self.embedding_service, 'qdrant_service'):
                logger.info("Pre-warming Qdrant connections for speed")
                await self.embedding_service.qdrant_service.health_check()
            
            # Pre-warm LLM connections
            if hasattr(self.llm_service, 'anthropic_client'):
                logger.info("Pre-warming Claude connections for speed")
                await self.llm_service.health_check()
                
            logger.info("System optimized for maximum speed")
            
        except Exception as e:
            logger.warning("Speed optimization partially failed", error=str(e))
    
    async def _fast_document_check(self, document_id: str) -> bool:
        """Ultra-fast document existence check with caching."""
        try:
            # Check memory cache first (fastest)
            if hasattr(self.cache_service, 'memory_cache'):
                cached_result = self.cache_service.memory_cache.get(f"doc_exists_{document_id}")
                if cached_result is not None:
                    logger.info("Fast cache hit for document existence", document_id=document_id)
                    return cached_result
            
            # Fall back to Qdrant check
            exists = await self.embedding_service.qdrant_service.document_exists(document_id=document_id)
            
            # Cache the result for next time
            if hasattr(self.cache_service, 'memory_cache'):
                self.cache_service.memory_cache[f"doc_exists_{document_id}"] = exists
            
            return exists
            
        except Exception as e:
            logger.error("Fast document check failed", error=str(e))
            return False