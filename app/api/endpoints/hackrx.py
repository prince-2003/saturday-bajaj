from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, AsyncGenerator
import structlog
import time
import json
import asyncio

from app.core.security import verify_api_key
from app.models.schemas import QueryRequest, QueryResponse, ErrorResponse
from app.core.container import get_retrieval_service

logger = structlog.get_logger(__name__)
router = APIRouter()

# No need to initialize service here - it's managed by the container

@router.post("/hackrx/run", 
             response_model=QueryResponse,
             responses={
                 400: {"model": ErrorResponse},
                 401: {"model": ErrorResponse},
                 500: {"model": ErrorResponse}
             })
async def run_hackrx_query(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
) -> QueryResponse:
    """
    Main endpoint for processing documents and answering questions.
    
    This endpoint:
    1. Downloads and processes the document from the provided URL
    2. Creates embeddings and stores them in the vector database
    3. Answers each question using semantic search and LLM reasoning
    4. Returns structured answers with metadata
    
    Args:
        request: QueryRequest containing document URL and questions
        api_key: API key for authentication (automatically verified)
    
    Returns:
        QueryResponse with answers and processing metadata
    
    Raises:
        HTTPException: For various error conditions (400, 401, 500)
    """
    start_time = time.time()
    
    try:
        logger.info("Received hackrx query request", 
                   document_url=str(request.documents),
                   question_count=len(request.questions))

        # Fast validation
        if not request.questions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one question must be provided"
            )

        if len(request.questions) > 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 50 questions allowed per request"
            )

        # Process with ultra-fast mode - no timeout concerns
        retrieval_service = get_retrieval_service()
        response = await retrieval_service.process_query_ultra_fast(request)
        
        processing_time = time.time() - start_time
        logger.info("Request processed successfully", 
                   processing_time=processing_time,
                   answers_count=len(response.answers))
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except ValueError as e:
        logger.error("Validation error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
        
    except Exception as e:
        logger.error("Unexpected error processing request", 
                    error=str(e),
                    error_type=type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred while processing the request"
        )


@router.post("/hackrx/run-stream")
async def run_hackrx_query_stream(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Streaming endpoint that sends progress updates to prevent client timeouts.
    
    Returns Server-Sent Events (SSE) with processing status and final results.
    """
    async def generate_stream() -> AsyncGenerator[str, None]:
        start_time = time.time()
        
        try:
            logger.info("Starting streaming processing", 
                       document_url=str(request.documents),
                       question_count=len(request.questions))

            # Send initial status
            yield f"data: {json.dumps({'status': 'started', 'message': 'Processing document...', 'progress': 0})}\n\n"
            
            # Fast validation
            if not request.questions:
                yield f"data: {json.dumps({'status': 'error', 'error': 'At least one question must be provided'})}\n\n"
                return

            if len(request.questions) > 50:
                yield f"data: {json.dumps({'status': 'error', 'error': 'Maximum 50 questions allowed per request'})}\n\n"
                return

            # Keep client alive with progress updates
            keepalive_active = True
            
            async def send_keepalive():
                progress = 10
                while keepalive_active and progress < 90:
                    await asyncio.sleep(5)  # Send update every 5 seconds
                    if keepalive_active:
                        progress += 10
                        yield f"data: {json.dumps({'status': 'processing', 'message': f'Processing... {progress}%', 'progress': progress})}\n\n"
            
            # Start keepalive task
            keepalive_gen = send_keepalive()
            
            try:
                # Process the request
                retrieval_service = get_retrieval_service()
                response = await retrieval_service.process_query_ultra_fast(request)
                
                # Cancel keepalive
                keepalive_active = False
                
                processing_time = time.time() - start_time
                
                # Send final result
                result = {
                    'status': 'completed',
                    'processing_time': processing_time,
                    'answers': response.answers,
                    'message': f'Completed in {processing_time:.1f} seconds'
                }
                
                yield f"data: {json.dumps(result)}\n\n"
                logger.info("Streaming request completed", processing_time=processing_time)
                
            except Exception as e:
                keepalive_active = False
                yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
                logger.error("Streaming processing failed", error=str(e))
                
        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
            logger.error("Stream generation failed", error=str(e))
    
    return StreamingResponse(generate_stream(), media_type="text/plain")


@router.post("/hackrx/run-legacy", 
             response_model=QueryResponse,
             responses={
                 400: {"model": ErrorResponse},
                 401: {"model": ErrorResponse},
                 500: {"model": ErrorResponse}
             })
async def run_hackrx_query_legacy(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
) -> QueryResponse:
    """
    Legacy endpoint for backward compatibility (may timeout on large documents).
    """
    start_time = time.time()
    
    try:
        logger.info("Legacy endpoint - received hackrx query request", 
                   document_url=str(request.documents),
                   question_count=len(request.questions))
        
        # Validate request
        if not request.questions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one question must be provided"
            )
        
        if len(request.questions) > 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 50 questions allowed per request"
            )
        
        # Process the query
        retrieval_service = get_retrieval_service()
        response = await retrieval_service.process_query(request)
        
        processing_time = time.time() - start_time
        logger.info("Legacy request processed successfully", 
                   processing_time=processing_time,
                   answers_count=len(response.answers))
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except ValueError as e:
        logger.error("Validation error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
        
    except Exception as e:
        logger.error("Unexpected error processing request", 
                    error=str(e),
                    error_type=type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred while processing the request"
        )

@router.post("/hackrx/analyze")
async def analyze_document(
    document_url: str,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Analyze document structure and provide insights.
    
    Args:
        document_url: URL of the document to analyze
        api_key: API key for authentication
    
    Returns:
        Document analysis including structure, clause types, and sections
    """
    try:
        logger.info("Document analysis requested", document_url=document_url)
        
        # For now, use the original method or implement in optimized service
        analysis = {"message": "Document analysis endpoint - implement if needed"}
        
        logger.info("Document analysis completed", 
                   document_id=analysis.get("document_id"))
        
        return analysis
        
    except Exception as e:
        logger.error("Document analysis failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document analysis failed: {str(e)}"
        )

@router.post("/hackrx/batch")
async def batch_process_questions(
    document_url: str,
    questions: List[str],
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Process multiple questions for a single document efficiently.
    
    Args:
        document_url: URL of the document to process
        questions: List of questions to answer
        api_key: API key for authentication
    
    Returns:
        Batch processing results with answers and metadata
    """
    try:
        logger.info("Batch processing requested", 
                   document_url=document_url,
                   question_count=len(questions))
        
        if len(questions) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 100 questions allowed for batch processing"
            )
        
        retrieval_service = get_retrieval_service()
        results = await retrieval_service.batch_process_optimized(questions, document_url)
        
        return {
            "document_url": document_url,
            "question_count": len(questions),
            "results": results,
            "success": True
        }
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error("Batch processing failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )

@router.get("/hackrx/performance")
async def get_performance_stats(api_key: str = Depends(verify_api_key)) -> Dict[str, Any]:
    """
    Get comprehensive performance statistics.
    
    Args:
        api_key: API key for authentication
    
    Returns:
        Detailed performance metrics for all system components
    """
    try:
        retrieval_service = get_retrieval_service()
        stats = retrieval_service.get_comprehensive_stats()
        return {
            "status": "success",
            "timestamp": time.time(),
            "performance_data": stats
        }
        
    except Exception as e:
        logger.error("Failed to get performance stats", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance stats: {str(e)}"
        )

@router.post("/hackrx/clear-cache")
async def clear_system_cache(api_key: str = Depends(verify_api_key)) -> Dict[str, str]:
    """
    Clear all system caches for fresh start.
    
    Args:
        api_key: API key for authentication
    
    Returns:
        Cache clearing status
    """
    try:
        retrieval_service = get_retrieval_service()
        await retrieval_service.clear_all_caches()
        return {
            "status": "success",
            "message": "All caches cleared successfully"
        }
        
    except Exception as e:
        logger.error("Failed to clear caches", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear caches: {str(e)}"
        )

@router.post("/hackrx/warm-up")
async def warm_up_system(
    documents: List[str],
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Warm up the system with commonly accessed documents.
    
    Args:
        documents: List of document URLs to pre-process
        api_key: API key for authentication
    
    Returns:
        Warm-up status and results
    """
    try:
        logger.info("System warm-up requested", documents=len(documents))
        
        retrieval_service = get_retrieval_service()
        await retrieval_service.warm_up_system(documents)
        
        return {
            "status": "success",
            "message": f"System warmed up with {len(documents)} documents",
            "documents_processed": len(documents)
        }
        
    except Exception as e:
        logger.error("System warm-up failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"System warm-up failed: {str(e)}"
        )