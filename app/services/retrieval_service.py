import asyncio
import time
from typing import List, Dict, Any, Optional
import structlog
import psutil

from app.services.document_processor import DocumentProcessor, OptimizedDocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.optimized_llm_service import OptimizedLLMService
from app.services.cache_service import IntelligentCacheService
from app.services.database_service import DatabaseService
from app.services.qdrant_service import QdrantService
from app.services.rerank_service import RerankService
from app.models.schemas import QueryRequest, QueryResponse, DocumentMetadata, EmbeddingChunk

logger = structlog.get_logger(__name__)


class RetrievalService:
    """
    High-Performance Production Retrieval Service with Two-Stage Retrieval
    and Semaphore-Bounded LLM Generation.
    
    Pipelines:
    1. Single-pass PyMuPDF in-memory parsing with deterministic byte SHA-256 ID.
    2. Stage 1: Vector search for Top-25 candidates from Qdrant.
    3. Stage 2: Cross-encoder FlashRank reranking down to Top-4 high-precision chunks.
    4. Strict 2,000-token context budgeting and semaphore-bounded generation.
    """

    def __init__(
        self,
        document_processor: Optional[OptimizedDocumentProcessor] = None,
        embedding_service: Optional[EmbeddingService] = None,
        llm_service: Optional[OptimizedLLMService] = None,
        cache_service: Optional[IntelligentCacheService] = None,
        database_service: Optional[DatabaseService] = None,
        rerank_service: Optional[RerankService] = None,
    ):
        self.document_processor = document_processor or OptimizedDocumentProcessor()
        self.embedding_service = embedding_service or EmbeddingService()
        self.cache_service = cache_service or IntelligentCacheService()
        self.database_service = database_service or DatabaseService()
        self.llm_service = llm_service or OptimizedLLMService(self.cache_service)
        self.rerank_service = rerank_service or RerankService()
        
        self.candidate_top_k = 25
        self.rerank_top_k = 4
        self.ingest_batch_size = 100

        logger.info("RetrievalService initialized with Two-Stage Reranking", 
                    candidate_top_k=self.candidate_top_k, 
                    rerank_top_k=self.rerank_top_k)

    async def _fast_document_check(self, document_id: str) -> bool:
        """Check if document is already ingested (memory cache -> Qdrant)."""
        try:
            if hasattr(self.cache_service, "memory_cache"):
                cached = self.cache_service.memory_cache.get(f"doc_exists_{document_id}")
                if cached is not None:
                    return cached

            exists = await self.embedding_service.qdrant_service.document_exists(document_id=document_id)
            if hasattr(self.cache_service, "memory_cache"):
                self.cache_service.memory_cache[f"doc_exists_{document_id}"] = exists
            return exists
        except Exception as e:
            logger.warning("Document existence check failed", error=str(e), document_id=document_id)
            return False

    async def _ingest_document(self, document_url: str, content: bytes, document_id: str):
        """Extract text, chunk, embed, and store in Qdrant."""
        start_time = time.time()
        logger.info("Starting document ingestion", document_id=document_id, url=document_url[:80])

        # Parse document into text
        stripped_content = content.lstrip()
        if stripped_content.startswith(b"%PDF"):
            text, pages = self.document_processor.extract_text_from_pdf(content)
        elif stripped_content.startswith(b"PK"):
            text, pages = self.document_processor.extract_text_from_docx(content)
        elif stripped_content.startswith(b"From:") or stripped_content.startswith(b"Received:"):
            text, pages = self.document_processor.extract_text_from_email(content)
        else:
            text = content.decode("utf-8", errors="ignore")
            pages = None

        chunks = self.document_processor.create_chunks(text, document_id)
        if not chunks:
            logger.warning("No chunks created from document", document_id=document_id)
            return

        # Embed and store in batches
        total_chunks = len(chunks)
        stored_chunks = 0
        for i in range(0, total_chunks, self.ingest_batch_size):
            chunk_batch = chunks[i : i + self.ingest_batch_size]
            texts = [c.text for c in chunk_batch]

            try:
                dense_embs, sparse_embs = await self.embedding_service.generate_hybrid_embeddings(texts)
                for chunk, d_emb, (s_idx, s_val) in zip(chunk_batch, dense_embs, sparse_embs):
                    chunk.embedding = d_emb
                    chunk.sparse_indices = s_idx
                    chunk.sparse_values = s_val

                success = await self.embedding_service.qdrant_service.store_embeddings(chunk_batch)
                if success:
                    stored_chunks += len(chunk_batch)
            except Exception as e:
                logger.error("Failed to store embedding batch", batch_idx=i, error=str(e))

        if hasattr(self.cache_service, "memory_cache"):
            self.cache_service.memory_cache[f"doc_exists_{document_id}"] = True

        logger.info("Document ingestion completed", 
                    document_id=document_id, 
                    total_chunks=total_chunks, 
                    stored=stored_chunks,
                    time_seconds=round(time.time() - start_time, 2))

    async def process_query_ultra_fast(self, request: QueryRequest) -> QueryResponse:
        """
        Main query processing pipeline:
        1. Fast download & SHA-256 byte hashing.
        2. Document ingestion if new.
        3. Multi-layer cache check for questions.
        4. Parallel Stage 1 Qdrant hybrid retrieval (dense + sparse BM25 via RRF, top 25).
        5. Stage 2 FlashRank cross-encoder reranking (top 4).
        6. Semaphore-bounded LLM generation with 2,000 token context budget.
        """
        start_time = time.time()
        if isinstance(request.documents, list):
            document_url = str(request.documents[0]).strip() if request.documents else ""
        else:
            document_url = str(request.documents).strip()
        all_questions = [q.strip() for q in request.questions if q and q.strip()]

        try:
            # Step 1: Download & Deterministic Identity
            content = await self.document_processor.download_document(document_url)
            document_id = self.document_processor.generate_document_id(content)

            # Step 2: Ingestion Check
            await self.embedding_service.qdrant_service.create_collection()
            is_ingested = await self._fast_document_check(document_id)
            if not is_ingested:
                await self._ingest_document(document_url, content, document_id)

            # Step 3: Check QA Cache for exact matches
            answers_map: Dict[int, str] = {}
            questions_to_retrieve: List[tuple[int, str]] = []

            for idx, question in enumerate(all_questions):
                cached = await self.cache_service.get_qa_cache(question, document_id)
                if cached and "answer" in cached:
                    answers_map[idx] = cached["answer"]
                else:
                    questions_to_retrieve.append((idx, question))

            # Step 4: Retrieval and Generation for uncached questions
            if questions_to_retrieve:
                q_indices = [idx for idx, _ in questions_to_retrieve]
                q_texts = [q for _, q in questions_to_retrieve]

                # Parallel question hybrid embeddings (dense + sparse BM25)
                q_dense_embs, q_sparse_embs = await self.embedding_service.generate_hybrid_embeddings(q_texts)

                # Parallel Stage 1 Hybrid Retrieval: Fetch top 25 candidates per question via RRF
                search_tasks = [
                    self.embedding_service.qdrant_service.search_hybrid(
                        dense_embedding=d_emb,
                        sparse_indices=s_emb[0],
                        sparse_values=s_emb[1],
                        top_k=self.candidate_top_k,
                        document_id=document_id
                    )
                    for d_emb, s_emb in zip(q_dense_embs, q_sparse_embs)
                ]
                candidates_results = await asyncio.gather(*search_tasks, return_exceptions=True)

                # Stage 2: Cross-Encoder Reranking down to top 4 high-signal passages
                questions_data = []
                for (idx, question), candidates in zip(questions_to_retrieve, candidates_results):
                    if isinstance(candidates, Exception) or not candidates:
                        logger.warning("Retrieval returned empty or failed for query", query=question[:40])
                        top_passages = []
                    else:
                        top_passages = self.rerank_service.rerank(
                            query=question,
                            chunks=candidates,
                            top_k=self.rerank_top_k
                        )

                    questions_data.append({
                        "question": question,
                        "context_chunks": top_passages,
                        "index": idx
                    })

                # Bounded parallel synthesis via semaphore in llm_service
                llm_results = await self.llm_service.answer_multiple_questions_batch(
                    questions_data, 
                    document_id
                )

                for res in llm_results:
                    q_idx = res.get("question_index", 0)
                    answers_map[q_idx] = res.get("answer", "")

            final_answers = [
                answers_map.get(i, "Unable to find sufficient context in document.") 
                for i in range(len(all_questions))
            ]

            logger.info("Pipeline processing completed", 
                        questions_count=len(all_questions), 
                        cached_hits=len(all_questions) - len(questions_to_retrieve),
                        total_time=round(time.time() - start_time, 2))

            return QueryResponse(answers=final_answers)

        except Exception as e:
            logger.error("Query processing failed", error=str(e), exc_info=True)
            error_answers = [f"Processing error: {str(e)}" for _ in all_questions]
            return QueryResponse(answers=error_answers)

    # Compatibility Aliases for endpoints
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Route to ultra-fast query processing."""
        return await self.process_query_ultra_fast(request)

    async def process_query_streaming(self, request: QueryRequest) -> QueryResponse:
        """Route streaming requests to ultra-fast query processing."""
        return await self.process_query_ultra_fast(request)

    async def batch_process_optimized(self, questions: List[str], document_url: str) -> List[Dict[str, Any]]:
        """Batch process questions and return structured dict."""
        req = QueryRequest(documents=document_url, questions=questions)
        response = await self.process_query_ultra_fast(req)
        return [
            {"question": q, "answer": ans, "status": "completed"}
            for q, ans in zip(questions, response.answers)
        ]

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Return system performance, memory, and cache telemetry."""
        mem = psutil.virtual_memory()
        return {
            "memory": {
                "total_mb": round(mem.total / (1024 * 1024), 1),
                "used_mb": round(mem.used / (1024 * 1024), 1),
                "available_mb": round(mem.available / (1024 * 1024), 1),
                "percent": mem.percent
            },
            "reranker": {
                "model": self.rerank_service.model_name,
                "active": self.rerank_service.ranker is not None
            },
            "llm": {
                "request_count": self.llm_service.request_count,
                "cache_hits": self.llm_service.cache_hits
            }
        }

    async def clear_all_caches(self):
        """Clear memory and remote Redis caches."""
        if hasattr(self.cache_service, "clear_cache"):
            await self.cache_service.clear_cache()
        if hasattr(self.cache_service, "memory_cache"):
            self.cache_service.memory_cache.clear()
        logger.info("All caches successfully cleared")

    async def warm_up_system(self, documents: List[str]):
        """Warm up system by pre-indexing documents."""
        for doc_url in documents:
            try:
                content = await self.document_processor.download_document(doc_url)
                doc_id = self.document_processor.generate_document_id(content)
                exists = await self._fast_document_check(doc_id)
                if not exists:
                    await self._ingest_document(doc_url, content, doc_id)
            except Exception as e:
                logger.warning("Failed to warm up document", url=doc_url, error=str(e))

    async def health_check(self) -> Dict[str, str]:
        """Component health check for dependent services."""
        health: Dict[str, str] = {}
        try:
            mem = psutil.virtual_memory()
            health["memory_usage"] = f"{mem.percent:.1f}%"
            health["qdrant"] = await self.embedding_service.qdrant_service.health_check()
            health["database"] = await self.database_service.health_check()
            health["reranker"] = "healthy" if self.rerank_service.ranker else "fallback_mode"
            health["status"] = "healthy"
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        return health