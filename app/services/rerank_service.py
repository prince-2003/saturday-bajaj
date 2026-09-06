"""
Rerank Service using FlashRank for low-latency CPU-based Cross-Encoder scoring.
"""
import os
import tempfile
from typing import List, Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)

class RerankService:
    """
    Ultra-fast, zero-PyTorch CPU Cross-Encoder using FlashRank.
    Provides second-stage precision ranking on retrieved candidate chunks.
    """
    
    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2", cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir or os.getenv("FLASHRANK_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "flashrank_models")
        self.ranker = None
        self._initialize_ranker()

    def _initialize_ranker(self):
        """Safely initialize FlashRank ranker with fallback on import/model failure."""
        try:
            from flashrank import Ranker
            os.makedirs(self.cache_dir, exist_ok=True)
            self.ranker = Ranker(model_name=self.model_name, cache_dir=self.cache_dir)
            logger.info("FlashRank Ranker initialized successfully", 
                        model=self.model_name, 
                        cache_dir=self.cache_dir)
        except ImportError:
            logger.warning("FlashRank not installed. RerankService will fallback to initial vector scores.")
            self.ranker = None
        except Exception as e:
            logger.warning("Failed to initialize FlashRank Ranker, falling back to vector scores", error=str(e))
            self.ranker = None

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Rerank retrieved chunks for a specific query.
        
        Args:
            query: The user question or search query
            chunks: List of candidate chunk dicts containing 'text' and other metadata
            top_k: Number of highest-ranked chunks to return
            
        Returns:
            List of top_k chunk dicts with 'rerank_score' attached
        """
        if not chunks:
            return []
            
        if len(chunks) <= top_k and self.ranker is None:
            return chunks

        # Fallback if FlashRank is unavailable
        if self.ranker is None:
            sorted_chunks = sorted(
                chunks, 
                key=lambda x: x.get("hybrid_score", x.get("score", 0.0)), 
                reverse=True
            )
            return sorted_chunks[:top_k]

        try:
            from flashrank import RerankRequest
            
            # FlashRank expects passages as list of dicts with 'id', 'text', and optional 'meta'
            passages = [
                {
                    "id": idx,
                    "text": chunk.get("text", ""),
                    "meta": chunk
                }
                for idx, chunk in enumerate(chunks)
                if chunk.get("text")
            ]

            if not passages:
                return []

            request = RerankRequest(query=query, passages=passages)
            scored_results = self.ranker.rerank(request)

            output = []
            for item in scored_results[:top_k]:
                meta = dict(item["meta"])
                meta["rerank_score"] = float(item["score"])
                output.append(meta)

            logger.info("Reranking completed", 
                        input_candidates=len(chunks), 
                        top_k=len(output),
                        top_score=output[0]["rerank_score"] if output else 0.0)
            return output

        except Exception as e:
            logger.error("Reranking failed with error, using fallback score sorting", error=str(e))
            sorted_chunks = sorted(
                chunks, 
                key=lambda x: x.get("hybrid_score", x.get("score", 0.0)), 
                reverse=True
            )
            return sorted_chunks[:top_k]
