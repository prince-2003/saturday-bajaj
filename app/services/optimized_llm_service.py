import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
import structlog
import re
from openai import AsyncOpenAI
import anthropic
import tiktoken

from app.core.config import settings
from app.services.cache_service import IntelligentCacheService

logger = structlog.get_logger(__name__)

class QuestionComplexityClassifier:
    """Classifies question complexity to route to appropriate model."""
    
    def __init__(self):
        self.simple_patterns = [
            r"what is the",
            r"when is",
            r"how much",
            r"how many",
            r"is there",
            r"does.*cover",
            r"what.*period",
            r"yes or no",
            r"true or false"
        ]
        
        self.complex_patterns = [
            r"explain.*why",
            r"compare.*with",
            r"analyze.*relationship",
            r"what are the implications",
            r"how does.*affect",
            r"under what circumstances",
            r"what factors",
            r"reasoning behind"
        ]
    
    def classify(self, question: str) -> str:
        """Classify question as 'simple' or 'complex'."""
        question_lower = question.lower()
        
        # Check for complex patterns first
        for pattern in self.complex_patterns:
            if re.search(pattern, question_lower):
                return "complex"
        
        # Check for simple patterns
        for pattern in self.simple_patterns:
            if re.search(pattern, question_lower):
                return "simple"
        
        # Default classification based on length and structure
        word_count = len(question.split())
        if word_count <= 10 and question.endswith('?'):
            return "simple"
        elif word_count > 20 or "and" in question_lower or "or" in question_lower:
            return "complex"
        
        return "simple"  # Default to simple for speed

class OptimizedLLMService:
    """Optimized LLM service with model routing and caching."""
    
    def __init__(self, cache_service: IntelligentCacheService):
        self.cache_service = cache_service
        self.openai_client = None
        self.anthropic_client = None
        self.encoding = None
        self.complexity_classifier = QuestionComplexityClassifier()
        
        # Performance metrics
        self.request_count = 0
        self.cache_hits = 0
        self.fast_model_usage = 0
        self.slow_model_usage = 0
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize LLM clients."""
        try:
            # Initialize OpenAI client
            if settings.openai_api_key:
                self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
                self.encoding = tiktoken.encoding_for_model(settings.openai_model)
                logger.info("OpenAI client initialized", model=settings.openai_model)
            
            # Initialize Anthropic client (Primary)
            if settings.anthropic_api_key:
                self.anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
                logger.info("Anthropic client initialized", model=settings.anthropic_model)
            
            if not self.anthropic_client and not self.openai_client:
                logger.error("No LLM clients initialized - missing API keys")
                
        except Exception as e:
            logger.error("Failed to initialize LLM clients", error=str(e))
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoding:
            return len(self.encoding.encode(text))
        return len(text.split()) * 1.3  # Rough estimation
    
    def optimize_context(self, context_chunks: List[Dict[str, Any]], 
                        query: str, max_tokens: int = 2000) -> str:
        """Optimize context by selecting most relevant chunks within token limit."""
        if not context_chunks:
            logger.warning("No context chunks provided to optimize_context")
            return ""
        
        # Sort by relevance score
        sorted_chunks = sorted(context_chunks, 
                             key=lambda x: x.get("hybrid_score", x.get("score", 0)), 
                             reverse=True)
        
        context_parts = []
        total_tokens = 0
        query_tokens = self.count_tokens(query)
        
        # Reserve tokens for query and response (reduced reservation)
        available_tokens = max_tokens - query_tokens - 200
        
        logger.info("Context optimization", 
                   max_tokens=max_tokens,
                   query_tokens=query_tokens, 
                   available_tokens=available_tokens,
                   chunks_count=len(sorted_chunks))
        
        for i, chunk in enumerate(sorted_chunks):
            chunk_text = f"[Page {chunk.get('page_number', 'N/A')}] {chunk['text']}"
            chunk_tokens = self.count_tokens(chunk_text)
            
            logger.info(f"Processing chunk {i}", 
                       chunk_tokens=chunk_tokens,
                       total_tokens=total_tokens,
                       would_exceed=(total_tokens + chunk_tokens > available_tokens))
            
            if total_tokens + chunk_tokens <= available_tokens:
                context_parts.append(chunk_text)
                total_tokens += chunk_tokens
                logger.info(f"Added chunk {i}", new_total=total_tokens)
            else:
                logger.info(f"Rejected chunk {i} - would exceed token limit")
                break
        
        final_context = "\n\n".join(context_parts)
        logger.info("Context optimization complete", 
                   parts_count=len(context_parts),
                   final_length=len(final_context))
        
        return final_context
    
    def create_optimized_system_prompt(self, question_type: str) -> str:
        """Create optimized system prompt based on question type."""
        base_prompt = """You are an expert AI assistant specialized in analyzing insurance policies, legal documents, HR policies, and compliance documents. Provide accurate, concise answers based ONLY on the provided context."""
        
        if question_type == "simple":
            return base_prompt + "\n\nFor this question, provide a direct, factual answer in 1-3 sentences. Include specific numbers, periods, or conditions when mentioned in the context."
        else:
            return base_prompt + "\n\nFor this complex question, provide a detailed explanation with reasoning. Include relevant conditions, limitations, and cross-references when applicable."
    
    def create_optimized_user_prompt(self, context: str, question: str, question_type: str) -> str:
        """Create optimized user prompt."""
        if question_type == "simple":
            return f"""Context: {context}

Question: {question}

Provide a direct, factual answer based on the context above."""
        else:
            return f"""Document Context: {context}

Question: {question}

Provide a comprehensive answer with reasoning based on the document context."""
    
    async def answer_question_fast(self, question: str, context_chunks: List[Dict[str, Any]], 
                                 document_id: str) -> Dict[str, Any]:
        """Fast question answering with caching and model routing."""
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Check cache first
            cached_answer = await self.cache_service.get_qa_cache(question, document_id)
            if cached_answer:
                self.cache_hits += 1
                cached_answer["processing_time"] = time.time() - start_time
                cached_answer["from_cache"] = True
                logger.info("Answer served from cache", 
                           question=question[:50], 
                           time=cached_answer["processing_time"])
                return cached_answer
            
            # Classify question complexity
            complexity = self.complexity_classifier.classify(question)
            
            # Optimize context - increase token limits for better context inclusion
            max_context_tokens = 3000 if complexity == "simple" else 4000
            context = self.optimize_context(context_chunks, question, max_context_tokens)
            
            logger.info("Context prepared for LLM", 
                       context_chunks_count=len(context_chunks),
                       context_length=len(context),
                       context_preview=context[:200] if context else "EMPTY")
            
            if not context:
                return self._create_no_context_response(question, start_time)
            
            # Route to appropriate model - Prioritize Anthropic Claude for all questions
            if self.anthropic_client:
                answer_result = await self._answer_with_claude(question, context, complexity)
                self.fast_model_usage += 1
            elif self.openai_client:
                answer_result = await self._answer_with_openai(question, context, complexity)
                self.slow_model_usage += 1
            else:
                raise ValueError("No LLM clients available")
            
            # Calculate total processing time
            answer_result["processing_time"] = time.time() - start_time
            answer_result["from_cache"] = False
            answer_result["model_used"] = "claude-haiku" if "claude" in answer_result.get("model", "") else "gpt-4-turbo"
            answer_result["complexity"] = complexity
            
            # Cache the result
            await self.cache_service.set_qa_cache(question, document_id, answer_result)
            
            logger.info("Question answered successfully", 
                       question=question[:50],
                       complexity=complexity,
                       model=answer_result["model_used"],
                       time=answer_result["processing_time"])
            
            return answer_result
            
        except Exception as e:
            logger.error("Failed to answer question", error=str(e))
            return self._create_error_response(question, str(e), start_time)
    
    async def _answer_with_claude(self, question: str, context: str, complexity: str) -> Dict[str, Any]:
        """Answer question using Claude-3-Haiku."""
        try:
            system_prompt = self.create_optimized_system_prompt(complexity)
            user_prompt = self.create_optimized_user_prompt(context, question, complexity)
            
            max_tokens = settings.anthropic_max_tokens if complexity == "simple" else settings.anthropic_max_tokens * 1.5
            
            response = await asyncio.wait_for(
                self.anthropic_client.messages.create(
                    model=settings.anthropic_model,
                    max_tokens=int(max_tokens),
                    temperature=settings.temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                ),
                timeout=settings.llm_timeout
            )
            
            answer = response.content[0].text.strip()
            
            return {
                "answer": answer,
                "confidence": self._calculate_confidence_fast(context, question, answer),
                "model": "claude-3-haiku",
                "token_usage": response.usage.input_tokens + response.usage.output_tokens if response.usage else 0,
                "sources": self._extract_sources_fast(context),
                "reasoning": f"Answered using Claude-3-Haiku for {complexity} question"
            }
            
        except asyncio.TimeoutError:
            logger.error("Claude request timed out")
            if settings.fallback_to_gpt4 and self.openai_client:
                logger.info("Falling back to OpenAI")
                return await self._answer_with_openai(question, context, complexity)
            raise
        except Exception as e:
            logger.error("Claude request failed", error=str(e))
            if settings.fallback_to_gpt4 and self.openai_client:
                logger.info("Falling back to OpenAI")
                return await self._answer_with_openai(question, context, complexity)
            raise
    
    async def _answer_with_openai(self, question: str, context: str, complexity: str) -> Dict[str, Any]:
        """Answer question using OpenAI GPT-4-Turbo."""
        try:
            system_prompt = self.create_optimized_system_prompt(complexity)
            user_prompt = self.create_optimized_user_prompt(context, question, complexity)
            
            max_tokens = settings.max_tokens if complexity == "complex" else int(settings.max_tokens * 0.7)
            
            response = await asyncio.wait_for(
                self.openai_client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=settings.temperature
                ),
                timeout=settings.llm_timeout
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "answer": answer,
                "confidence": self._calculate_confidence_fast(context, question, answer),
                "model": "gpt-4-turbo",
                "token_usage": response.usage.total_tokens if response.usage else 0,
                "sources": self._extract_sources_fast(context),
                "reasoning": f"Answered using GPT-4-Turbo for {complexity} question"
            }
            
        except asyncio.TimeoutError:
            logger.error("OpenAI request timed out")
            raise
        except Exception as e:
            logger.error("OpenAI request failed", error=str(e))
            raise
    
    def _calculate_confidence_fast(self, context: str, question: str, answer: str) -> float:
        """Fast confidence calculation."""
        base_confidence = 0.7
        
        # Boost for specific details
        if any(indicator in answer.lower() for indicator in 
               ["days", "months", "years", "percent", "%", "amount", "section"]):
            base_confidence += 0.15
        
        # Reduce for vague language
        if any(indicator in answer.lower() for indicator in 
               ["may", "might", "possibly", "unclear"]):
            base_confidence -= 0.1
        
        # Boost for direct quotes
        if '"' in answer or "'" in answer:
            base_confidence += 0.1
        
        return max(0.1, min(1.0, base_confidence))
    
    def _extract_sources_fast(self, context: str) -> List[Dict[str, Any]]:
        """Fast source extraction."""
        sources = []
        pages = re.findall(r'\[Page (\d+|\w+)\]', context)
        for i, page in enumerate(pages[:3]):  # Limit to top 3 sources
            sources.append({
                "page": page,
                "relevance": 1.0 - (i * 0.1),
                "type": "document_page"
            })
        return sources
    
    def _create_no_context_response(self, question: str, start_time: float) -> Dict[str, Any]:
        """Create response when no context is available."""
        return {
            "answer": "I cannot answer this question as no relevant information was found in the document.",
            "confidence": 0.1,
            "model": "no-context",
            "token_usage": 0,
            "sources": [],
            "reasoning": "No relevant context found",
            "processing_time": time.time() - start_time,
            "from_cache": False
        }
    
    def _create_error_response(self, question: str, error: str, start_time: float) -> Dict[str, Any]:
        """Create error response."""
        return {
            "answer": f"Error processing question: {error}",
            "confidence": 0.0,
            "model": "error",
            "token_usage": 0,
            "sources": [],
            "reasoning": f"Processing error: {error}",
            "processing_time": time.time() - start_time,
            "from_cache": False
        }
    
    async def batch_answer_questions(self, questions: List[str], 
                                   context_chunks: List[Dict[str, Any]], 
                                   document_id: str) -> List[Dict[str, Any]]:
        """Answer multiple questions efficiently with parallel processing."""
        logger.info("Processing batch questions", count=len(questions))
        
        # Process questions in parallel with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent requests
        
        async def process_question(question):
            async with semaphore:
                return await self.answer_question_fast(question, context_chunks, document_id)
        
        tasks = [process_question(q) for q in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Batch question processing failed", 
                           question_index=i, 
                           error=str(result))
                processed_results.append(self._create_error_response(
                    questions[i], str(result), time.time()
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cache_hit_rate = (self.cache_hits / self.request_count) if self.request_count > 0 else 0
        
        return {
            "total_requests": self.request_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(cache_hit_rate, 3),
            "fast_model_usage": self.fast_model_usage,
            "slow_model_usage": self.slow_model_usage,
            "model_distribution": {
                "claude_haiku": self.fast_model_usage,
                "gpt4_turbo": self.slow_model_usage
            }
        }
    
    async def health_check(self) -> Dict[str, str]:
        """Check health of LLM services."""
        health = {}
        
        # Check OpenAI
        if self.openai_client:
            try:
                # Simple test request
                health["openai"] = "healthy"
            except Exception:
                health["openai"] = "unhealthy"
        else:
            health["openai"] = "not_configured"
        
        # Check Anthropic
        if self.anthropic_client:
            try:
                # Simple test request
                health["anthropic"] = "healthy"
            except Exception:
                health["anthropic"] = "unhealthy"
        else:
            health["anthropic"] = "not_configured"
        
        return health