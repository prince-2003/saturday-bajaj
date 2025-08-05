"""
Pydantic schemas for request/response models.
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict


# Configuration to resolve Pydantic v2 namespace warnings
class BaseModelConfig(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
        populate_by_name=True,  # Allow both field names and aliases
        extra='allow'  # Allow extra fields that might be present
    )


class EmbeddingChunk(BaseModelConfig):
    """Model for document chunks with embeddings."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="Text content of the chunk")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    document_id: Optional[str] = Field(default=None, description="ID of the source document")
    chunk_index: int = Field(default=0, description="Index of the chunk within the document")
    start_position: Optional[int] = Field(default=None, description="Start position in the original document")
    end_position: Optional[int] = Field(default=None, description="End position in the original document")
    
    # Add a property for backward compatibility with 'id' field
    @property
    def id(self) -> str:
        return self.chunk_id


class QueryRequest(BaseModelConfig):
    """Request model for document query processing."""
    documents: Union[str, List[str]] = Field(
        ..., 
        description="Document URL(s) or file path(s) to process"
    )
    questions: List[str] = Field(
        ..., 
        min_items=1, 
        description="List of questions to answer based on the documents"
    )
    use_cache: bool = Field(
        default=True, 
        description="Whether to use cached results if available"
    )
    model_preference: Optional[str] = Field(
        default=None, 
        description="Preferred model: 'claude', 'gpt4', or 'auto'"
    )


class DocumentMetadata(BaseModelConfig):
    """Metadata about processed documents."""
    document_id: str = Field(..., description="Document identifier")
    filename: Optional[str] = Field(default=None, description="Original filename")
    size_bytes: Optional[int] = None
    pages: Optional[int] = None
    processing_time: Optional[float] = None
    chunks_created: Optional[int] = None


class AnswerMetadata(BaseModelConfig):
    """Metadata for individual answers."""
    question: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    model_used: str
    processing_time: float
    sources: List[str] = Field(default_factory=list)
    tokens_used: Optional[int] = None


class PerformanceMetrics(BaseModelConfig):
    """Performance metrics for the query processing."""
    total_processing_time: float
    document_processing_time: float
    embedding_time: float
    retrieval_time: float
    llm_time: float
    questions_processed: int
    avg_time_per_question: float
    cache_hit_rate: Optional[float] = None
    target_time_met: bool
    optimization_level: str = "standard"


class ModelUsage(BaseModelConfig):
    """Model usage statistics."""
    claude_haiku: int = 0
    gpt4_turbo: int = 0
    total_tokens: int = 0
    cost_estimate: Optional[float] = None


class QueryResponse(BaseModelConfig):
    """Response model for document query processing."""
    answers: List[str] = Field(
        ..., 
        description="List of answers corresponding to the input questions"
    )
    # metadata: Dict[str, Any] = Field(
    #     default_factory=dict,
    #     description="Additional metadata about the processing"
    # )
    # answer_metadata: Optional[List[AnswerMetadata]] = Field(
    #     default=None,
    #     description="Detailed metadata for each answer"
    # )
    # document_metadata: Optional[List[DocumentMetadata]] = Field(
    #     default=None,
    #     description="Metadata about processed documents"
    # )
    # performance_metrics: Optional[PerformanceMetrics] = Field(
    #     default=None,
    #     description="Performance metrics for the query"
    # )
    # model_usage: Optional[ModelUsage] = Field(
    #     default=None,
    #     description="Model usage statistics"
    # )
    # processing_id: Optional[str] = Field(
    #     default=None,
    #     description="Unique identifier for this processing request"
    # )


class ErrorResponse(BaseModelConfig):
    """Error response model."""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    request_id: Optional[str] = Field(default=None, description="Request identifier")


class HealthResponse(BaseModelConfig):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Response timestamp")
    version: str = Field(..., description="API version")
    services: Dict[str, str] = Field(default_factory=dict, description="Status of dependent services")
    uptime: Optional[float] = Field(default=None, description="Service uptime in seconds")


class PerformanceResponse(BaseModelConfig):
    """Performance metrics response model."""
    current_metrics: Dict[str, Any] = Field(default_factory=dict)
    historical_metrics: Optional[Dict[str, Any]] = Field(default=None)
    system_info: Optional[Dict[str, Any]] = Field(default=None)


class BatchQueryRequest(BaseModelConfig):
    """Request model for batch query processing."""
    requests: List[QueryRequest] = Field(
        ..., 
        min_items=1, 
        max_items=10,
        description="List of query requests to process in batch"
    )
    parallel_processing: bool = Field(
        default=True,
        description="Whether to process requests in parallel"
    )


class BatchQueryResponse(BaseModelConfig):
    """Response model for batch query processing."""
    responses: List[QueryResponse] = Field(
        ...,
        description="List of responses corresponding to the batch requests"
    )
    batch_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the batch processing"
    )


class WarmupRequest(BaseModelConfig):
    """Request model for system warmup."""
    documents: List[str] = Field(
        ...,
        description="List of document URLs to preload"
    )
    preload_embeddings: bool = Field(
        default=True,
        description="Whether to precompute embeddings"
    )


class AnalysisRequest(BaseModelConfig):
    """Request model for document analysis."""
    documents: Union[str, List[str]] = Field(
        ...,
        description="Document URL(s) to analyze"
    )
    analysis_type: str = Field(
        default="structure",
        description="Type of analysis: 'structure', 'content', 'metadata'"
    )


class AnalysisResponse(BaseModelConfig):
    """Response model for document analysis."""
    document_analysis: Dict[str, Any] = Field(
        ...,
        description="Analysis results for each document"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of the analysis"
    )


# Legacy compatibility aliases
QueryRequestModel = QueryRequest
QueryResponseModel = QueryResponse
ErrorResponseModel = ErrorResponse
