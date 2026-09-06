import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    api_title: str = "LLM-Powered Intelligent Query-Retrieval System"
    api_version: str = "2.0.0"
    debug: bool = False
    port: int = int(os.getenv("PORT", 8000))  # Port detection from environment
    
    # Authentication
    api_key: str = "f187d1bc4df8a6a7e6cba86fc31bdedfcce699eac885b85570bb61c6d6e8c7f2"
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo"
    openai_embedding_model: str = "text-embedding-3-large"
    max_tokens: int = 4000  # Increased for testing - was 1500
    temperature: float = 0.1
    
    # Anthropic Configuration (Primary LLM)
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-haiku-20240307"
    anthropic_max_tokens: int = 3000  # Increased for testing - was 1000
    
    # Google Gemini Configuration (New Primary LLM)
    google_api_key: Optional[str] = None
    gemini_model: str = "gemini-3.6-flash"
    gemini_max_tokens: int = 8192
    
    # Timeout Configuration
    client_timeout: int = 300  # 5 minutes max for client requests
    llm_timeout: int = 30      # 30 seconds per LLM call
    embedding_timeout: int = 60 # 1 minute for embedding generation
    document_processing_timeout: int = 180  # 3 minutes for document processing
    
    # Embedding Configuration
    use_local_embeddings: bool = True
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384  # BAAI/bge-small-en-v1.5 dimension (was 3072 for OpenAI large)
    
    # Qdrant Configuration (Primary Vector DB)
    qdrant_url: Optional[str] = None
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "documents"
    qdrant_api_key: Optional[str] = None
    
    # Redis Configuration (Upstash)
    redis_url: str = "redis://localhost:6379"
    upstash_redis_rest_url: Optional[str] = None
    upstash_redis_rest_token: Optional[str] = None
    redis_password: Optional[str] = None
    redis_db: int = 0
    redis_cluster_mode: bool = False
    
    # Cache Configuration
    cache_ttl: int = 3600  # 1 hour
    document_cache_ttl: int = 7200  # 2 hours
    embedding_cache_ttl: int = 86400  # 24 hours
    qa_cache_ttl: int = 1800  # 30 minutes
    semantic_cache_threshold: float = 0.95  # Similarity threshold
    
    # Database Configuration
    database_url: Optional[str] = None
    
    # Document Processing (Optimized)
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    chunk_size: int = 1500  # Larger chunks for better context
    chunk_overlap: int = 300
    max_chunks_per_document: int = 200
    max_paragraph_chunk_size: int = 1000  # Maximum size for paragraph chunks
    min_chunk_size: int = 100  # Minimum size for chunks
    
    # Performance Settings (Optimized)
    max_concurrent_requests: int = 50
    request_timeout: int = 180  # Reduced timeout
    llm_timeout: int = 30  # LLM-specific timeout
    embedding_timeout: int = 15
    vector_search_timeout: int = 5
    
    # Model Routing
    simple_question_threshold: float = 0.3  # Complexity threshold
    use_fast_model_first: bool = True
    fallback_to_gpt4: bool = True
    
    # Rate Limiting Settings
    openai_requests_per_minute: int = 3000
    anthropic_requests_per_minute: int = 1000
    base_retry_delay: float = 1.0
    max_retries: int = 3
    
    # Memory Management
    max_memory_cache_size: int = 1000  # Number of items
    hot_documents_limit: int = 10
    faiss_index_size: int = 10000
    
    # Performance Targets
    target_response_time: int = 15  # seconds
    cache_hit_rate_target: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()