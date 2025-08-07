import asyncio
from typing import List, Dict, Any, Optional
import structlog
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
from sqlalchemy import text # Add this import

from app.core.config import settings

logger = structlog.get_logger(__name__)

Base = declarative_base()

class Document(Base):
    """Database model for documents."""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(String, unique=True, nullable=False)
    filename = Column(String, nullable=False)
    content_type = Column(String)
    file_size = Column(Integer)
    page_count = Column(Integer)
    doc_metadata = Column(JSON)
    processed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

class DocumentChunk(Base):
    """Database model for document chunks."""
    __tablename__ = "document_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(String, unique=True, nullable=False)
    document_id = Column(String, nullable=False)
    text = Column(Text, nullable=False)
    chunk_index = Column(Integer)
    page_number = Column(Integer)
    section = Column(String)
    doc_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class Query(Base):
    """Database model for queries."""
    __tablename__ = "queries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_id = Column(String, unique=True, nullable=False)
    document_id = Column(String)
    question = Column(Text, nullable=False)
    answer = Column(Text)
    confidence = Column(Float)
    processing_time = Column(Float)
    token_usage = Column(Integer)
    doc_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseService:
    """Service for managing PostgreSQL database operations."""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and create tables."""
        try:
            if settings.database_url:
                self.engine = create_engine(settings.database_url)
                self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
                
                # Create tables
                Base.metadata.create_all(bind=self.engine)
                
                logger.info("Database initialized successfully")
            else:
                logger.warning("Database URL not provided")
                
        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
    
    def get_db(self):
        """Get database session."""
        if not self.SessionLocal:
            raise ValueError("Database not initialized")
        
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    async def store_document(self, document_id: str, filename: str, 
                           content_type: str = None, file_size: int = None,
                           page_count: int = None, metadata: dict = None) -> bool:
        """Store document metadata."""
        if not self.SessionLocal:
            logger.warning("Database not initialized, skipping document storage")
            return False
        
        try:
            db = self.SessionLocal()
            
            # Check if document already exists
            existing = db.query(Document).filter(Document.document_id == document_id).first()
            if existing:
                logger.info("Document already exists in database", document_id=document_id)
                db.close()
                return True
            
            # Create new document record
            document = Document(
                document_id=document_id,
                filename=filename,
                content_type=content_type,
                file_size=file_size,
                page_count=page_count,
                metadata=metadata or {}
            )
            
            db.add(document)
            db.commit()
            db.close()
            
            logger.info("Document stored in database", document_id=document_id)
            return True
            
        except Exception as e:
            logger.error("Failed to store document", error=str(e), document_id=document_id)
            return False
    
    async def store_document_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Store document chunks."""
        if not self.SessionLocal:
            logger.warning("Database not initialized, skipping chunk storage")
            return False
        
        try:
            db = self.SessionLocal()
            
            chunk_records = []
            for chunk_data in chunks:
                chunk = DocumentChunk(
                    chunk_id=chunk_data.get("chunk_id"),
                    document_id=chunk_data.get("document_id"),
                    text=chunk_data.get("text", ""),
                    chunk_index=chunk_data.get("chunk_index"),
                    page_number=chunk_data.get("page_number"),
                    section=chunk_data.get("section"),
                    metadata=chunk_data.get("metadata", {})
                )
                chunk_records.append(chunk)
            
            db.add_all(chunk_records)
            db.commit()
            db.close()
            
            logger.info("Document chunks stored in database", chunk_count=len(chunks))
            return True
            
        except Exception as e:
            logger.error("Failed to store document chunks", error=str(e))
            return False
    
    async def store_query(self, query_id: str, document_id: str, question: str,
                         answer: str = None, confidence: float = None,
                         processing_time: float = None, token_usage: int = None,
                         metadata: dict = None) -> bool:
        """Store query and answer data."""
        if not self.SessionLocal:
            logger.warning("Database not initialized, skipping query storage")
            return False
        
        try:
            db = self.SessionLocal()
            
            query_record = Query(
                query_id=query_id,
                document_id=document_id,
                question=question,
                answer=answer,
                confidence=confidence,
                processing_time=processing_time,
                token_usage=token_usage,
                metadata=metadata or {}
            )
            
            db.add(query_record)
            db.commit()
            db.close()
            
            logger.info("Query stored in database", query_id=query_id)
            return True
            
        except Exception as e:
            logger.error("Failed to store query", error=str(e), query_id=query_id)
            return False
    
    async def get_document_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent document processing history."""
        if not self.SessionLocal:
            return []
        
        try:
            db = self.SessionLocal()
            
            documents = db.query(Document).order_by(Document.created_at.desc()).limit(limit).all()
            
            result = []
            for doc in documents:
                result.append({
                    "document_id": doc.document_id,
                    "filename": doc.filename,
                    "content_type": doc.content_type,
                    "file_size": doc.file_size,
                    "page_count": doc.page_count,
                    "processed_at": doc.processed_at.isoformat() if doc.processed_at else None,
                    "created_at": doc.created_at.isoformat() if doc.created_at else None,
                    "metadata": doc.doc_metadata
                })
            
            db.close()
            return result
            
        except Exception as e:
            logger.error("Failed to get document history", error=str(e))
            return []
    
    async def get_query_history(self, document_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent query history."""
        if not self.SessionLocal:
            return []
        
        try:
            db = self.SessionLocal()
            
            query = db.query(Query)
            if document_id:
                query = query.filter(Query.document_id == document_id)
            
            queries = query.order_by(Query.created_at.desc()).limit(limit).all()
            
            result = []
            for q in queries:
                result.append({
                    "query_id": q.query_id,
                    "document_id": q.document_id,
                    "question": q.question,
                    "answer": q.answer,
                    "confidence": q.confidence,
                    "processing_time": q.processing_time,
                    "token_usage": q.token_usage,
                    "created_at": q.created_at.isoformat() if q.created_at else None,
                    "metadata": q.doc_metadata
                })
            
            db.close()
            return result
            
        except Exception as e:
            logger.error("Failed to get query history", error=str(e))
            return []
    
    # In database_service.py

    async def health_check(self) -> str:
        """Check database service health."""
        if not self.engine:
            return "unhealthy"

        try:
            # Test database connection
            with self.engine.connect() as connection:
                # Wrap the raw SQL string in the text() function
                connection.execute(text("SELECT 1")) # THE FIX IS HERE
            return "healthy"
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return "unhealthy"