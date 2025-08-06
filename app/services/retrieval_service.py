import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
import structlog

from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.optimized_llm_service import OptimizedLLMService
from app.services.cache_service import IntelligentCacheService
from app.services.database_service import DatabaseService
from app.services.qdrant_service import QdrantService
from app.models.schemas import QueryRequest, QueryResponse, DocumentMetadata
import hashlib 


logger = structlog.get_logger(__name__)

class RetrievalService:
    """Main service orchestrating document processing, retrieval, and question answering."""
    
    def __init__(self, 
                 document_processor: DocumentProcessor = None,
                 embedding_service: EmbeddingService = None,
                 llm_service: OptimizedLLMService = None,
                 cache_service: IntelligentCacheService = None,
                 database_service: DatabaseService = None):
        """
        Initialize RetrievalService with injected dependencies.
        
        Args:
            document_processor: Document processing service
            embedding_service: Embedding and vector search service
            llm_service: LLM service for question answering
            cache_service: Caching service
            database_service: Database service
        """
        # Use dependency injection if services are provided, otherwise create new instances
        # This allows backward compatibility while supporting centralized initialization
        self.document_processor = document_processor or DocumentProcessor()
        self.embedding_service = embedding_service or EmbeddingService()
        self.cache_service = cache_service or IntelligentCacheService()
        self.database_service = database_service or DatabaseService()
        self.llm_service = llm_service or OptimizedLLMService(self.cache_service)
        self.document_chunks_cache = {}  # In-memory cache for document chunks

    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a complete query request with document and questions, with intelligent caching and batching.
        """
        start_time = time.time()
        document_url = str(request.documents)
        document_id = hashlib.md5(document_url.encode()).hexdigest()[:12]

        try:
            logger.info("Starting query processing", 
                        document_url=document_url,
                        document_id=document_id,
                        question_count=len(request.questions))

            # Step 1: Check if the document has already been processed and stored in Qdrant.
            is_processed = await self.embedding_service.qdrant_service.document_exists(document_id=document_id)
            
            if not is_processed:
                logger.info("CACHE MISS: Document not found in Qdrant. Processing for the first time.", document_id=document_id)
                metadata, chunks = await self.document_processor.process_document(document_url)
                await self.embedding_service.store_embeddings(chunks)
                logger.info("New document processed and stored in Qdrant.", document_id=document_id)
            else:
                logger.info("CACHE HIT: Document found in Qdrant. Skipping processing and embedding.", document_id=document_id)
                metadata = DocumentMetadata(document_id=document_id, document_type="pdf", total_pages=None, total_chunks=None, processing_time=0)

            # --- PERFORMANCE OPTIMIZATION: BATCH EMBEDDING GENERATION ---

            # Step 2: Collect all questions and generate embeddings in a single batch.
            all_questions = [q for q in request.questions]
            logger.info("Generating embeddings for all questions in one batch", count=len(all_questions))
            
            # This makes ONE efficient API call to OpenAI instead of N calls.
            question_embeddings = await self.embedding_service.generate_embeddings(all_questions)

            # Create a lookup map for easy access to each question's embedding.
            embedding_map = {question: emb for question, emb in zip(all_questions, question_embeddings)}

            # --- END OF OPTIMIZATION ---

            # Step 3: Process each question concurrently using the pre-fetched embeddings.
            async def process_single_question_optimized(i, question, doc_id):
                logger.info("Processing question", index=i+1, question=question[:100])

                # Get the pre-generated embedding from our map. No new API call here.
                query_embedding = embedding_map[question]
                
                # Call the Qdrant service directly for the vector search. This is fast.
                context_chunks = await self.embedding_service.qdrant_service.search_similar(
                    query_embedding=query_embedding, 
                    top_k=3, 
                    document_id=doc_id
                )
                
                # The LLM call remains the same.
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

            tasks = [process_single_question_optimized(i, question, document_id) for i, question in enumerate(request.questions)]
            results = await asyncio.gather(*tasks)
            all_answers = [r["answer"] for r in results]
            all_metadata = [r["metadata"] for r in results]

            # The rest of your metrics and response formatting logic remains the same
            processing_time = time.time() - start_time
            total_tokens = sum(meta.get("token_usage", 0) for meta in all_metadata)
            
            response_metadata = {
                "processing_time": round(processing_time, 2),
                "total_tokens": total_tokens,
                "document_metadata": metadata.model_dump(),
                "question_metadata": all_metadata,
                "avg_confidence": round(sum(meta["confidence"] for meta in all_metadata) / len(all_metadata), 3) if all_metadata else 0
            }
            
            logger.info("Query processing completed", 
                        processing_time=processing_time,
                        total_tokens=total_tokens)
            
            return QueryResponse(answers=all_answers)
            
        except Exception as e:
            logger.error("Query processing failed", error=str(e), exc_info=True)
            raise ValueError(f"Query processing failed: {str(e)}")    
    async def _retrieve_context(self, question: str, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant context for a question."""
        try:
            logger.info("Context retrieval requested", question=question[:100], document_id=document_id)
            
            if not document_id or document_id not in self.document_chunks_cache:
                logger.warning("Document chunks not found in cache", document_id=document_id, cached_docs=list(self.document_chunks_cache.keys()))
                return []
            
            chunks = self.document_chunks_cache[document_id]
            logger.info("Retrieved chunks from cache", document_id=document_id, chunk_count=len(chunks))
            
            # Enhanced text-based relevance scoring for insurance documents
            scored_chunks = []
            question_lower = question.lower()
            question_words = set(question_lower.split())
            
            # Add insurance-specific keywords mapping
            insurance_keywords = {
                'grace period': ['grace', 'period', 'premium', 'payment', 'due'],
                'waiting period': ['waiting', 'period', 'coverage', 'days', 'months'], 
                'pre-existing': ['pre-existing', 'ped', 'diseases', 'condition'],
                'maternity': ['maternity', 'pregnancy', 'childbirth', 'delivery'],
                'cataract': ['cataract', 'surgery', 'eye', 'treatment'],
                'organ donor': ['organ', 'donor', 'transplant', 'medical'],
                'claim discount': ['claim', 'discount', 'ncd', 'bonus'],
                'health check': ['health', 'check', 'preventive', 'screening'],
                'hospital': ['hospital', 'definition', 'facility', 'treatment'],
                'ayush': ['ayush', 'alternative', 'treatment', 'therapy'],
                'room rent': ['room', 'rent', 'icu', 'charges', 'limits']
            }
            
            for chunk in chunks:
                chunk_text_lower = chunk.text.lower()
                chunk_words = set(chunk_text_lower.split())
                
                # Basic word overlap score
                overlap = len(question_words.intersection(chunk_words))
                
                # Enhanced scoring for insurance-specific terms
                bonus_score = 0
                for key_phrase, related_words in insurance_keywords.items():
                    if key_phrase in question_lower:
                        for word in related_words:
                            if word in chunk_text_lower:
                                bonus_score += 2
                
                total_score = overlap + bonus_score
                
                # Also check for partial phrase matches
                question_bigrams = [question_lower[i:i+10] for i in range(len(question_lower)-9)]
                for bigram in question_bigrams:
                    if len(bigram) > 5 and bigram in chunk_text_lower:
                        total_score += 3
                
                if total_score > 0:
                    scored_chunks.append({
                        "text": chunk.text,
                        "chunk_id": chunk.chunk_id,
                        "page_number": chunk.page_number,
                        "section": chunk.section,
                        "score": total_score,
                        "chunk_index": chunk.chunk_index
                    })
                    logger.info("Found matching chunk", chunk_id=chunk.chunk_id, score=total_score, chunk_preview=chunk.text[:100])
            
            # Sort by relevance score and return top 5 chunks
            scored_chunks.sort(key=lambda x: x["score"], reverse=True)
            relevant_chunks = scored_chunks[:5]
            
            # If no relevant chunks found by overlap, return first few chunks
            if not relevant_chunks:
                logger.info("No keyword overlap found, returning first 3 chunks")
                relevant_chunks = []
                for i, chunk in enumerate(chunks[:3]):
                    relevant_chunks.append({
                        "text": chunk.text,
                        "chunk_id": chunk.chunk_id,
                        "page_number": chunk.page_number,
                        "section": chunk.section,
                        "score": 0.5,  # Default score
                        "chunk_index": chunk.chunk_index
                    })
            
            logger.info("Context retrieved", chunks_returned=len(relevant_chunks))
            return relevant_chunks
            
        except Exception as e:
            logger.error("Context retrieval failed", error=str(e))
            return []
    
    async def health_check(self) -> Dict[str, str]:
        """Check health of all services."""
        health_status = {}
        
        # Check document processor
        try:
            # Simple test - this will succeed if dependencies are available
            processor = DocumentProcessor()
            health_status["document_processor"] = "healthy"
        except Exception:
            health_status["document_processor"] = "unhealthy"
        
        # Check embedding service
        try:
            if self.embedding_service.openai_client:
                health_status["embedding_service"] = "healthy"
            else:
                health_status["embedding_service"] = "unhealthy"
        except Exception:
            health_status["embedding_service"] = "unhealthy"
        
        # Check Qdrant service
        try:
            qdrant_health = await self.embedding_service.qdrant_service.health_check()
            health_status["qdrant"] = qdrant_health
        except Exception:
            health_status["qdrant"] = "unhealthy"
        
        # Check database service
        try:
            database_health = await self.database_service.health_check()
            health_status["database"] = database_health
        except Exception:
            health_status["database"] = "unhealthy"
        
        # Check LLM service
        try:
            llm_health = await self.llm_service.health_check()
            health_status.update(llm_health)
        except Exception:
            health_status["anthropic"] = "unhealthy"
            health_status["openai"] = "unhealthy"
        
        return health_status
    
    async def clear_all_caches(self):
        """Clear all service caches."""
        try:
            # Clear cache service
            if hasattr(self.cache_service, 'clear_all'):
                await self.cache_service.clear_all()
            
            logger.info("All caches cleared successfully")
        except Exception as e:
            logger.error("Failed to clear caches", error=str(e))
            raise
    
    async def warm_up_system(self, documents: List[str]):
        """Warm up system by pre-processing common documents."""
        try:
            logger.info("Starting system warm-up", document_count=len(documents))
            
            for doc_url in documents:
                try:
                    # Pre-process document to warm up embeddings
                    metadata, chunks = await self.document_processor.process_document(doc_url)
                    await self.embedding_service.store_embeddings(chunks)
                    logger.info("Document warmed up", document=doc_url)
                except Exception as e:
                    logger.warning("Failed to warm up document", document=doc_url, error=str(e))
            
            logger.info("System warm-up completed successfully")
        except Exception as e:
            logger.error("System warm-up failed", error=str(e))
            raise