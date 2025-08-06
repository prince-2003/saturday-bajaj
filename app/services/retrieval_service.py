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

logger = structlog.get_logger(__name__)

class RetrievalService:
    """Main service orchestrating document processing, retrieval, and question answering."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.cache_service = IntelligentCacheService()
        self.database_service = DatabaseService()
        self.llm_service = OptimizedLLMService(self.cache_service)
        self.document_chunks_cache = {}  # In-memory cache for document chunks
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a complete query request with document and questions."""
        start_time = time.time()
        
        try:
            logger.info("Starting query processing", 
                       document_url=str(request.documents),
                       question_count=len(request.questions))
            
            # Step 1: Process document
            self.embedding_service.qdrant_service.reset_qdrant()
            metadata, chunks = await self.document_processor.process_document(str(request.documents))
            # Step 2: Store chunks in cache for retrieval
            self.document_chunks_cache[metadata.document_id] = chunks
            logger.info("Document chunks cached", document_id=metadata.document_id, chunk_count=len(chunks))
            
            # Step 3: Store embeddings (for future use)
            await self.embedding_service.store_embeddings(chunks)
            
            # Step 4: Process each question concurrently
            async def process_single_question(i, question, metadata):
                logger.info("Processing question", index=i+1, question=question[:100])
                context_chunks = await self.embedding_service.search_similar(query=question, top_k=3, document_id=metadata.document_id)
                answer_result = await self.llm_service.answer_question_fast(question, context_chunks, metadata.document_id)
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

            tasks = [process_single_question(i, question, metadata) for i, question in enumerate(request.questions)]
            results = await asyncio.gather(*tasks)
            all_answers = [r["answer"] for r in results]
            all_metadata = [r["metadata"] for r in results]

            # Calculate processing metrics
            processing_time = time.time() - start_time
            total_tokens = sum(meta.get("token_usage", 0) for meta in all_metadata)
            
            response_metadata = {
                "processing_time": round(processing_time, 2),
                "total_tokens": total_tokens,
                "document_metadata": metadata.dict(),
                "question_metadata": all_metadata,
                "avg_confidence": round(sum(meta["confidence"] for meta in all_metadata) / len(all_metadata), 3)
            }
            
            logger.info("Query processing completed", 
                       processing_time=processing_time,
                       total_tokens=total_tokens,
                       avg_confidence=response_metadata["avg_confidence"])
            
            return QueryResponse(
                answers=all_answers
                # metadata=response_metadata  # Commented out to remove from API response
            )
            
        except Exception as e:
            logger.error("Query processing failed", error=str(e))
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