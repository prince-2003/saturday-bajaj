import asyncio
import aiofiles
import tempfile
import os
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import structlog
import httpx
from io import BytesIO

import PyPDF2
import pdfplumber
from docx import Document
import email
from email.mime.text import MIMEText

from app.core.config import settings
from app.models.schemas import DocumentMetadata, EmbeddingChunk

logger = structlog.get_logger(__name__)

class DocumentProcessor:
    """Handles document ingestion and processing for various file formats."""
    
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        
    async def download_document(self, url: str) -> bytes:
        """Download document from URL."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.content
        except Exception as e:
            logger.error("Failed to download document", url=url, error=str(e))
            raise ValueError(f"Failed to download document: {str(e)}")
    
    async def process_document(self, url: str) -> tuple[DocumentMetadata, List[EmbeddingChunk]]:
        """Process document from URL and return metadata and chunks."""
        logger.info("Processing document", url=url)
        
        # Download document
        document_content = await self.download_document(url)
        
        # Determine file type from URL
        parsed_url = urlparse(url)
        file_extension = os.path.splitext(parsed_url.path)[1].lower()
        
        if not file_extension:
            # Try to determine from content type or content
            if document_content.startswith(b'%PDF'):
                file_extension = '.pdf'
            elif document_content.startswith(b'PK'):
                file_extension = '.docx'
            else:
                file_extension = '.txt'
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file.write(document_content)
            temp_file_path = temp_file.name
        
        try:
            # Process based on file type
            if file_extension == '.pdf':
                text_content, total_pages = await self._process_pdf(temp_file_path)
            elif file_extension == '.docx':
                text_content, total_pages = await self._process_docx(temp_file_path)
            elif file_extension in ['.eml', '.msg']:
                text_content, total_pages = await self._process_email(temp_file_path)
            else:
                # Fallback to text processing
                async with aiofiles.open(temp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text_content = await f.read()
                total_pages = None
            
            # Create chunks
            chunks = self._create_chunks(text_content, url)
            
            # Create metadata
            document_id = self._generate_document_id(url)
            metadata = DocumentMetadata(
                document_id=document_id,
                document_type=file_extension[1:],
                total_pages=total_pages,
                total_chunks=len(chunks),
                processing_time=0.0  # Will be updated by caller
            )
            
            logger.info("Document processed successfully", 
                       document_id=document_id, 
                       chunks=len(chunks), 
                       pages=total_pages)
            
            return metadata, chunks
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    
    async def _process_pdf(self, file_path: str) -> tuple[str, int]:
        """Process PDF file and extract text."""
        text_content = ""
        total_pages = 0
        
        try:
            # Use pdfplumber for better text extraction
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        except Exception as e:
            logger.warning("pdfplumber failed, trying PyPDF2", error=str(e))
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    total_pages = len(pdf_reader.pages)
                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            except Exception as e2:
                logger.error("Both PDF processors failed", error=str(e2))
                raise ValueError(f"Failed to process PDF: {str(e2)}")
        
        return text_content.strip(), total_pages
    
    async def _process_docx(self, file_path: str) -> tuple[str, Optional[int]]:
        """Process DOCX file and extract text."""
        try:
            doc = Document(file_path)
            text_content = ""
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content += paragraph.text + "\n"
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join([cell.text.strip() for cell in row.cells])
                    if row_text.strip():
                        text_content += row_text + "\n"
            
            return text_content.strip(), None
            
        except Exception as e:
            logger.error("Failed to process DOCX", error=str(e))
            raise ValueError(f"Failed to process DOCX: {str(e)}")
    
    async def _process_email(self, file_path: str) -> tuple[str, Optional[int]]:
        """Process email file and extract text."""
        try:
            with open(file_path, 'rb') as f:
                msg = email.message_from_bytes(f.read())
            
            text_content = ""
            
            # Extract headers
            text_content += f"From: {msg.get('From', 'Unknown')}\n"
            text_content += f"To: {msg.get('To', 'Unknown')}\n"
            text_content += f"Subject: {msg.get('Subject', 'No Subject')}\n"
            text_content += f"Date: {msg.get('Date', 'Unknown')}\n\n"
            
            # Extract body
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        text_content += part.get_payload(decode=True).decode('utf-8', errors='ignore')
            else:
                text_content += msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            
            return text_content.strip(), None
            
        except Exception as e:
            logger.error("Failed to process email", error=str(e))
            raise ValueError(f"Failed to process email: {str(e)}")
    
    def _create_chunks(self, text: str, document_url: str) -> List[EmbeddingChunk]:
        """Create chunks from text content."""
        chunks = []
        document_id = self._generate_document_id(document_url)
        
        # Simple chunking strategy
        words = text.split()
        chunk_index = 0
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if chunk_text.strip():
                chunk = EmbeddingChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    document_id=document_id,
                    text=chunk_text.strip(),
                    chunk_index=chunk_index,
                    page_number=self._extract_page_number(chunk_text),
                    section=self._extract_section(chunk_text),
                    clause_type=self._classify_clause_type(chunk_text)
                )
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def _generate_document_id(self, url: str) -> str:
        """Generate a unique document ID from URL."""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def _extract_page_number(self, text: str) -> Optional[int]:
        """Extract page number from chunk text."""
        import re
        page_match = re.search(r'--- Page (\d+) ---', text)
        if page_match:
            return int(page_match.group(1))
        return None
    
    def _extract_section(self, text: str) -> Optional[str]:
        """Extract section information from chunk text."""
        # Simple heuristic to identify sections
        lines = text.split('\n')
        for line in lines[:3]:  # Check first few lines
            if any(keyword in line.lower() for keyword in ['section', 'article', 'clause', 'chapter']):
                return line.strip()
        return None
    
    def _classify_clause_type(self, text: str) -> Optional[str]:
        """Classify the type of clause based on content."""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ['cover', 'benefit', 'include']):
            return 'coverage_clause'
        elif any(keyword in text_lower for keyword in ['exclude', 'not cover', 'limitation']):
            return 'exclusion_clause'
        elif any(keyword in text_lower for keyword in ['condition', 'require', 'must', 'shall']):
            return 'condition_clause'
        elif any(keyword in text_lower for keyword in ['premium', 'payment', 'cost', 'fee']):
            return 'payment_clause'
        elif any(keyword in text_lower for keyword in ['waiting period', 'grace period', 'time']):
            return 'time_clause'
        else:
            return 'general_clause'