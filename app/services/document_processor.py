import asyncio
import aiofiles
import tempfile
import os
import gc
import psutil
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from urllib.parse import urlparse
import structlog
import httpx
from io import BytesIO
import re
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import PyPDF2
import pdfplumber
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except ImportError:
    pdfminer_extract_text = None

from docx import Document
import email
from email.mime.text import MIMEText

from app.core.config import settings
from app.models.schemas import DocumentMetadata, EmbeddingChunk

logger = structlog.get_logger(__name__)

class OptimizedDocumentProcessor:
    """
    🚀 PRODUCTION-OPTIMIZED Document Processor with Memory Management
    
    Features:
    - Advanced text compression and chunking
    - Smart paragraph-based processing
    - Aggressive memory cleanup
    - Multi-threaded PDF processing with timeouts
    - RAM usage monitoring and cleanup
    """

    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        
        # 🚀 PRODUCTION OPTIMIZED: 1200MB memory available - USE FULL 1200MB!
        self.max_memory_usage = 1.00  # 100% of 1200MB = 1200MB fully utilized
        self.batch_size = 1000  # MASSIVE: 1000 pages at once (increased from 800)
        self.embedding_batch_size = 300  # MAXIMUM: 300 embeddings per API call (increased from 250)
        self.max_workers = min(40, cpu_count() * 5)  # Maximum workers for full utilization
        
        # Cost-optimized chunking configuration
        self.use_semantic_chunking = True
        self.max_paragraph_chunk_size = getattr(settings, 'max_paragraph_chunk_size', self.chunk_size * 1.25)
        self.min_chunk_size = getattr(settings, 'min_chunk_size', 100)
        self.preserve_section_boundaries = True
        
        # Cost control settings
        self.max_tokens_per_chunk = 800
        self.enable_chunk_compression = True
        
        logger.info("PRODUCTION DocumentProcessor initialized - MAXIMUM 1200MB USAGE", 
                   batch_size=self.batch_size,
                   memory_limit=f"{self.max_memory_usage*100:.0f}%",
                   memory_available="1200MB - FULL UTILIZATION",
                   max_workers=self.max_workers,
                   embedding_batch_size=self.embedding_batch_size)

    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within limits with cleanup."""
        try:
            memory_percent = psutil.virtual_memory().percent / 100
            if memory_percent > self.max_memory_usage:
                logger.warning("Memory limit exceeded, forcing cleanup", 
                              current=f"{memory_percent*100:.1f}%",
                              limit=f"{self.max_memory_usage*100:.1f}%")
                # Force aggressive cleanup
                gc.collect()
                return False
            return True
        except Exception as e:
            logger.warning("Memory check failed", error=str(e))
            return True  # Continue processing if memory check fails

    def _force_memory_cleanup(self, context: str = ""):
        """Force aggressive memory cleanup."""
        try:
            before = psutil.virtual_memory().percent
            gc.collect()  # Force garbage collection
            after = psutil.virtual_memory().percent
            logger.info(f"Memory cleanup completed {context}", 
                       before=f"{before:.1f}%",
                       after=f"{after:.1f}%",
                       freed=f"{before-after:.1f}%")
        except Exception as e:
            logger.warning("Memory cleanup failed", error=str(e))

    async def download_document(self, url: str) -> bytes:
        """Download document from URL with memory monitoring."""
        try:
            logger.info("Starting document download", url=url[:100])
            # 🚀 PRODUCTION: Increase timeout for larger files (800MB memory = bigger files)
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                content = response.content
                logger.info("Document downloaded successfully", 
                           size_mb=round(len(content) / 1024 / 1024, 1))
                return content
                
        except Exception as e:
            logger.error("Failed to download document", url=url, error=str(e))
            raise ValueError(f"Failed to download document: {str(e)}")

    async def process_document_streaming(self, url: str) -> tuple[DocumentMetadata, AsyncGenerator[List[EmbeddingChunk], None]]:
        """
        🚀 STAGED PROCESSING: Process document with aggressive memory management
        
        Stage 1: Download and process all chunks in memory
        Stage 2: Yield chunks in batches with cleanup between batches
        """
        logger.info("Starting streaming document processing", url=url)
        
        # Download document
        document_content = await self.download_document(url)
        document_id = self._generate_document_id(url)
        
        # Determine file type
        parsed_url = urlparse(url)
        file_extension = os.path.splitext(parsed_url.path)[1].lower()
        
        if not file_extension:
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
            logger.info("Document streaming setup complete", document_id=document_id)
            
            # STAGE 1: Process document into chunks with memory monitoring
            if file_extension == '.pdf':
                text_content, total_pages = await self._process_pdf_optimized(temp_file_path)
            elif file_extension == '.docx':
                text_content, total_pages = await self._process_docx(temp_file_path)
            elif file_extension in ['.eml', '.msg']:
                text_content, total_pages = await self._process_email(temp_file_path)
            else:
                async with aiofiles.open(temp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text_content = await f.read()
                total_pages = None

            # Create all chunks in memory first
            all_chunks = self._create_chunks(text_content, url)
            
            # Clean up text content immediately
            del text_content
            self._force_memory_cleanup("after text processing")
            
            # Create metadata
            metadata = DocumentMetadata(
                document_id=document_id,
                document_type=file_extension[1:],
                total_pages=total_pages,
                total_chunks=len(all_chunks),
                processing_time=0.0
            )
            
            logger.info("STAGED processing complete", 
                       total_chunks=len(all_chunks),
                       pages=total_pages)

            # STAGE 2: Yield chunks in batches - MAXIMUM performance with full 1200MB
            async def chunk_generator():
                batch_size = 150  # INCREASED: Process 150 chunks at a time (was 100)
                for i in range(0, len(all_chunks), batch_size):
                    # Create a COPY of the batch to prevent reference issues
                    batch = [chunk for chunk in all_chunks[i:i + batch_size]]
                    yield batch
                    
                    # Minimal memory cleanup - only when absolutely necessary
                    if i % (batch_size * 15) == 0:  # Every 15 batches (2250 chunks)
                        memory_percent = psutil.virtual_memory().percent
                        if memory_percent > 99:  # Only cleanup at 99% usage
                            logger.info(f"Memory cleanup at batch {i//batch_size + 1}, memory: {memory_percent}%")
                            gc.collect()  # Light garbage collection only

            return metadata, chunk_generator()
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
                logger.info("Cleaned up temporary PDF file", file_path=temp_file_path)
            except Exception as e:
                logger.warning("Failed to clean up temp file", error=str(e))

    async def _process_pdf_optimized(self, file_path: str) -> tuple[str, int]:
        """
        🚀 ULTRA-OPTIMIZED PDF processing with memory management and multi-threading
        """
        batch_results = []  # Initialize at function scope
        try:
            total_pages = self._get_pdf_page_count_safe(file_path)
            logger.info("ULTRA-FAST PDF processing started", 
                       total_pages=total_pages,
                       file_size_mb=round(os.path.getsize(file_path) / 1024 / 1024, 1))
            
            # 🚀 PRODUCTION: Process ALL pages in one massive batch
            batch_results = self._extract_batch_ultra_fast(file_path, 0, total_pages, self.max_workers)
            
            # Build text with memory monitoring - NEVER DELETE batch_results until we're done
            logger.info("Building final text content", pages_processed=len(batch_results))
            text_content = ""
            successful_pages = 0
            
            for page_num, text in batch_results:
                if text.strip():
                    text_content += f"\n--- Page {page_num + 1} ---\n{text}\n"
                    successful_pages += 1
                
                # Memory check every 100 pages but don't break - we need all data
                if page_num % 100 == 0:
                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent > 95:  # Only stop if extremely critical
                        logger.error(f"Critical memory usage {memory_percent}%, stopping text building")
                        break
            
            # Calculate success rate
            success_rate = successful_pages / total_pages * 100 if total_pages > 0 else 0
            
            logger.info("ULTRA-FAST PDF completed", 
                       success_rate=f"{success_rate:.1f}%",
                       total_pages=total_pages,
                       successful_pages=successful_pages,
                       text_length=len(text_content))
            
            # Only clean up AFTER we've built the text content
            del batch_results
            self._force_memory_cleanup("after PDF processing")
            
            # If we got very little text, try the fallback
            if len(text_content.strip()) < 1000 and total_pages > 10:
                logger.warning("PDF extraction yielded very little text, trying fallback")
                return await self._process_pdf_simple(file_path)
            
            return text_content.strip(), total_pages
            
        except Exception as e:
            logger.error("ULTRA-FAST PDF processing failed", error=str(e))
            # Clean up batch_results if it exists
            if batch_results:
                del batch_results
            # Fallback to simple processing
            try:
                logger.info("Trying fallback PDF processing...")
                return await self._process_pdf_simple(file_path)
            except Exception as fallback_error:
                logger.error("All PDF processing methods failed", fallback_error=str(fallback_error))
                return "", 0

    def _extract_batch_ultra_fast(self, file_path: str, start_page: int, end_page: int, max_workers: int) -> List[Tuple[int, str]]:
        """ULTRA-FAST batch extraction with memory monitoring."""
        results = []
        batch_size = end_page - start_page
        
        # 🚀 PRODUCTION: Very generous timeout for maximum success rate
        # 3.0 seconds per page minimum, 45 minutes maximum for complex PDFs
        base_timeout_per_page = 3.0  # TRIPLED from 1.0 to 3.0 seconds per page
        min_timeout = 600  # 10 minutes minimum (increased from 180s)
        max_timeout = 2700  # 45 minutes maximum (increased from 1200s)
        
        timeout = min(max(batch_size * base_timeout_per_page, min_timeout), max_timeout)
        
        logger.info(f"Processing batch with {max_workers} workers, timeout: {timeout}s")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all page extraction tasks
            future_to_page = {}
            for page_num in range(start_page, end_page):
                future = executor.submit(self._extract_single_page_fast, file_path, page_num)
                future_to_page[future] = page_num
            
            # Collect results with more generous per-page timeout
            completed_futures = []
            try:
                for future in as_completed(future_to_page, timeout=timeout):
                    try:
                        page_num = future_to_page[future]
                        # Give each page up to 5 seconds to complete (increased from 2.0)
                        text = future.result(timeout=5.0)  
                        results.append((page_num, text))
                        completed_futures.append(future)
                        
                        # Show progress every 100 pages
                        if len(completed_futures) % 100 == 0:
                            success_rate = len([r for r in results if r[1].strip()]) / len(results) * 100
                            logger.info(f"Progress: {len(completed_futures)}/{len(future_to_page)} pages, "
                                      f"success rate: {success_rate:.1f}%")
                        
                        # Only stop for critical memory issues (99% usage)
                        if len(completed_futures) % 50 == 0:
                            memory_percent = psutil.virtual_memory().percent
                            if memory_percent > 99:  # Only stop if extremely critical
                                logger.error(f"Critical memory {memory_percent}%, stopping batch processing")
                                break
                                
                    except Exception as e:
                        page_num = future_to_page[future]
                        logger.debug(f"Page {page_num} extraction failed", error=str(e))
                        results.append((page_num, ""))
                        completed_futures.append(future)  # Still count as completed
                        
            except Exception as timeout_error:
                logger.info(f"Batch processing completed: {len(completed_futures)}/{len(future_to_page)} pages processed")
                # Don't log as error - partial processing is still valuable
                
                # Add empty results for unprocessed pages to maintain page order
                processed_pages = {future_to_page[f] for f in completed_futures}
                for page_num in range(start_page, end_page):
                    if page_num not in processed_pages:
                        results.append((page_num, ""))
        
        # Sort results by page number
        results.sort(key=lambda x: x[0])
        
        # Memory cleanup
        self._force_memory_cleanup("after batch processing")
        
        return results

    def _extract_single_page_fast(self, file_path: str, page_num: int) -> str:
        """Extract single page using the best available method."""
        extraction_methods = []
        
        # Prioritize pdfplumber for quality
        if pdfplumber:
            extraction_methods.append(('pdfplumber', self._extract_with_pdfplumber))
        
        # Add other methods
        if fitz:
            extraction_methods.append(('pymupdf', self._extract_with_pymupdf))
        if pdfminer_extract_text:
            extraction_methods.append(('pdfminer', self._extract_with_pdfminer))
        if PyPDF2:
            extraction_methods.append(('pypdf2', self._extract_with_pypdf2))
        
        # Try each method
        for method_name, extract_func in extraction_methods:
            try:
                return extract_func(file_path, page_num)
            except Exception:
                continue
        
        return ""  # All methods failed

    def _extract_with_pdfplumber(self, file_path: str, page_num: int) -> str:
        """Extract text using pdfplumber."""
        with pdfplumber.open(file_path) as pdf:
            if page_num < len(pdf.pages):
                page = pdf.pages[page_num]
                return page.extract_text() or ""
        return ""

    def _extract_with_pymupdf(self, file_path: str, page_num: int) -> str:
        """Extract text using PyMuPDF."""
        doc = fitz.open(file_path)
        try:
            if page_num < doc.page_count:
                page = doc[page_num]
                return page.get_text()
        finally:
            doc.close()
        return ""

    def _extract_with_pdfminer(self, file_path: str, page_num: int) -> str:
        """Extract text using pdfminer."""
        # pdfminer doesn't support single page extraction easily
        return ""

    def _extract_with_pypdf2(self, file_path: str, page_num: int) -> str:
        """Extract text using PyPDF2."""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            if page_num < len(pdf_reader.pages):
                page = pdf_reader.pages[page_num]
                return page.extract_text() or ""
        return ""

    def _get_pdf_page_count_safe(self, file_path: str) -> int:
        """Get PDF page count safely."""
        try:
            if pdfplumber:
                with pdfplumber.open(file_path) as pdf:
                    return len(pdf.pages)
        except:
            pass
            
        try:
            if fitz:
                doc = fitz.open(file_path)
                count = doc.page_count
                doc.close()
                return count
        except:
            pass
            
        try:
            if PyPDF2:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    return len(pdf_reader.pages)
        except:
            pass
        
        return 0

    async def _process_pdf_simple(self, file_path: str) -> tuple[str, int]:
        """Simple PDF processing fallback."""
        text_content = ""
        total_pages = 0
        
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        except Exception as e:
            logger.warning("Simple PDF processing failed", error=str(e))
            
        return text_content.strip(), total_pages

    async def _process_docx(self, file_path: str) -> tuple[str, Optional[int]]:
        """Process DOCX file and extract text with memory monitoring."""
        try:
            doc = Document(file_path)
            text_content = ""
            
            # Process paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content += paragraph.text + "\n"
            
            # Process tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join([cell.text.strip() for cell in row.cells])
                    if row_text.strip():
                        text_content += row_text + "\n"
            
            # Memory cleanup
            del doc
            self._force_memory_cleanup("after DOCX processing")
            
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
            text_content += f"From: {msg.get('From', 'Unknown')}\n"
            text_content += f"To: {msg.get('To', 'Unknown')}\n"
            text_content += f"Subject: {msg.get('Subject', 'No Subject')}\n"
            text_content += f"Date: {msg.get('Date', 'Unknown')}\n\n"
            
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        text_content += part.get_payload(decode=True).decode('utf-8', errors='ignore')
            else:
                text_content += msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            
            # Memory cleanup
            del msg
            self._force_memory_cleanup("after email processing")
            
            return text_content.strip(), None
            
        except Exception as e:
            logger.error("Failed to process email", error=str(e))
            raise ValueError(f"Failed to process email: {str(e)}")

    def _create_chunks_generator(self, text: str, document_url: str, batch_size: int = 150):
        """Create chunks in batches to enable streaming processing and immediate memory cleanup."""
        document_id = self._generate_document_id(document_url)
        chunk_index = 0
        
        paragraphs = self._split_into_paragraphs(text)
        current_batch = []
        
        current_chunk = ""
        current_chunk_paragraphs = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Compress paragraph if enabled
            if self.enable_chunk_compression:
                paragraph = self._compress_text(paragraph)
            
            para_token_count = len(paragraph.split()) * 0.75
            current_token_count = len(current_chunk.split()) * 0.75
            
            # Check if we need to create a new chunk
            if current_token_count > 0 and (
                current_token_count + para_token_count > self.max_tokens_per_chunk or
                len(current_chunk.split()) + len(paragraph.split()) > self.max_paragraph_chunk_size
            ):
                if current_chunk.strip() and len(current_chunk.split()) >= self.min_chunk_size:
                    chunk = self._create_chunk(
                        current_chunk.strip(), 
                        document_id, 
                        chunk_index, 
                        document_url
                    )
                    current_batch.append(chunk)
                    chunk_index += 1
                    
                    # Yield batch when it reaches batch_size
                    if len(current_batch) >= batch_size:
                        yield current_batch
                        current_batch = []
                
                # Handle overlap
                if self.chunk_overlap > 0 and current_chunk_paragraphs:
                    overlap_context = self._get_overlap_context(current_chunk_paragraphs[-1])
                    current_chunk = overlap_context + "\n\n" + paragraph if overlap_context else paragraph
                else:
                    current_chunk = paragraph
                current_chunk_paragraphs = [paragraph]
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_chunk_paragraphs.append(paragraph)
        
        # Add the last chunk
        if current_chunk.strip() and len(current_chunk.split()) >= self.min_chunk_size:
            chunk = self._create_chunk(
                current_chunk.strip(), 
                document_id, 
                chunk_index, 
                document_url
            )
            current_batch.append(chunk)
        
        # Yield final batch if not empty
        if current_batch:
            yield current_batch

    def _create_chunks(self, text: str, document_url: str) -> List[EmbeddingChunk]:
        """Legacy method - collects all chunks for backward compatibility."""
        all_chunks = []
        for batch in self._create_chunks_generator(text, document_url, batch_size=1000):
            all_chunks.extend(batch)
        
        # Statistics
        if all_chunks:
            total_tokens = sum(len(chunk.text.split()) * 0.75 for chunk in all_chunks)
            avg_tokens_per_chunk = total_tokens / len(all_chunks)
            # Skip expensive max calculation for performance
            max_tokens = 0
        else:
            total_tokens = avg_tokens_per_chunk = max_tokens = 0
        
        logger.info("Created cost-optimized chunks", 
                    total_chunks=len(all_chunks),
                    estimated_total_tokens=int(total_tokens),
                    avg_tokens_per_chunk=int(avg_tokens_per_chunk),
                    max_tokens_per_chunk=int(max_tokens))
        
        return all_chunks
        """Create chunks from text content using cost-optimized paragraph-based strategy."""
        chunks = []
        document_id = self._generate_document_id(document_url)
        chunk_index = 0
        
        paragraphs = self._split_into_paragraphs(text)
        
        current_chunk = ""
        current_chunk_paragraphs = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Compress paragraph if enabled
            if self.enable_chunk_compression:
                paragraph = self._compress_text(paragraph)
            
            para_token_count = len(paragraph.split()) * 0.75
            current_token_count = len(current_chunk.split()) * 0.75
            
            # Check if we need to create a new chunk
            if current_token_count > 0 and (
                current_token_count + para_token_count > self.max_tokens_per_chunk or
                len(current_chunk.split()) + len(paragraph.split()) > self.max_paragraph_chunk_size
            ):
                if current_chunk.strip() and len(current_chunk.split()) >= self.min_chunk_size:
                    chunk = self._create_chunk(
                        current_chunk.strip(), 
                        document_id, 
                        chunk_index, 
                        document_url
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Handle overlap
                if self.chunk_overlap > 0 and current_chunk_paragraphs:
                    overlap_context = self._get_overlap_context(current_chunk_paragraphs[-1])
                    current_chunk = overlap_context + "\n\n" + paragraph if overlap_context else paragraph
                else:
                    current_chunk = paragraph
                current_chunk_paragraphs = [paragraph]
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_chunk_paragraphs.append(paragraph)
            
            # MEMORY MONITORING: Check but don't stop - preserve data integrity
            if len(chunks) % 50 == 0:
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 95:
                    logger.error(f"Critical memory {memory_percent}% during chunking - continuing to preserve data", 
                                chunks_created=len(chunks))
                elif memory_percent > 90:
                    logger.warning(f"High memory {memory_percent}% during chunking", 
                                  chunks_created=len(chunks))
                # Force cleanup every 200 chunks to prevent memory issues
                if len(chunks) % 200 == 0:
                    self._force_memory_cleanup(f"after {len(chunks)} chunks")
        
        # Add the last chunk
        if current_chunk.strip() and len(current_chunk.split()) >= self.min_chunk_size:
            chunk = self._create_chunk(
                current_chunk.strip(), 
                document_id, 
                chunk_index, 
                document_url
            )
            chunks.append(chunk)
        
        # Memory cleanup - more aggressive
        del paragraphs
        if 'current_chunk_paragraphs' in locals():
            del current_chunk_paragraphs
        if 'current_chunk' in locals():
            del current_chunk
        self._force_memory_cleanup("after chunking")
        
        # Statistics (skip max calculation if no chunks to save memory)
        if chunks:
            total_tokens = sum(len(chunk.text.split()) * 0.75 for chunk in chunks)
            avg_tokens_per_chunk = total_tokens / len(chunks)
            # Skip expensive max calculation for performance
            max_tokens = 0
        else:
            total_tokens = avg_tokens_per_chunk = max_tokens = 0
        
        logger.info("Created cost-optimized chunks", 
                    total_chunks=len(chunks),
                    estimated_total_tokens=int(total_tokens),
                    avg_tokens_per_chunk=int(avg_tokens_per_chunk),
                    max_tokens_per_chunk=int(max_tokens))
        
        return chunks

    def _compress_text(self, text: str) -> str:
        """
        Compresses text by replacing long, redundant phrases with shorter equivalents.
        """
        replacement_map = {
            # General legal and formal phrases
            r'\bwhich shall be the basis of this contract and is deemed to be incorporated herein\b': '[part of contract]',
            r'\bfollowing the advice of a duly qualified professional\b': 'on professional advice',
            r'\bshall not be liable to make any payment in respect of any expenses incurred in connection with or in respect of\b': 'excludes payment for',
            r'\bsubject to the terms, conditions, and limitations contained herein\b': 'subject to terms and conditions',
            r'\bsudden, unforeseen and involuntary event caused by external, visible and violent means\b': '[definition of Accident]',
            r'\bunder the supervision of a registered and qualified professional\b': 'under professional supervision',
            r'\baccessible to the authorized representative\b': 'accessible to representative',
            
            # Specific recurring clauses
            r'\bIn the event of \w+, the person/representative shall notify\b': 'Must notify for events',
            r'\b(for|under) any of the following circumstances\b': 'if:',
            r'\bin the event of misrepresentation, mis description or non-disclosure of any material fact\b': 'for any non-disclosure',

            # Remove repeated header/footer info
            r'Page \d+ of \d+': '',
            r'\s+': ' '  # Final cleanup of excessive whitespace
        }
        # Apply all replacements
        for pattern, replacement in replacement_map.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
        # Final cleanup of excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _get_overlap_context(self, paragraph: str) -> str:
        """Get smart overlap context - last sentence instead of full paragraph."""
        sentences = paragraph.split('.')
        if len(sentences) > 1:
            return sentences[-2].strip() + '.' if sentences[-2].strip() else ''
        return paragraph[:100] + '...' if len(paragraph) > 100 else paragraph

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into meaningful paragraphs with enhanced logic."""
        paragraphs = text.split('\n\n')
        refined_paragraphs = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(para.split()) > self.chunk_size * 2:
                sentences = self._split_into_sentences(para)
                current_para = ""
                
                for sentence in sentences:
                    if not current_para:
                        current_para = sentence
                    elif len((current_para + " " + sentence).split()) <= self.chunk_size * 2:
                        current_para += " " + sentence
                    else:
                        refined_paragraphs.append(current_para)
                        current_para = sentence
                
                if current_para:
                    refined_paragraphs.append(current_para)
            else:
                refined_paragraphs.append(para)
        
        return refined_paragraphs
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved logic."""
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n+\s*(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence.split()) >= 3:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _create_chunk(self, text: str, document_id: str, chunk_index: int, document_url: str) -> EmbeddingChunk:
        """Create a single embedding chunk with metadata."""
        return EmbeddingChunk(
            chunk_id=f"{document_id}_chunk_{chunk_index}",
            document_id=document_id,
            text=text,
            chunk_index=chunk_index,
            metadata={
                "page_number": self._extract_page_number(text),
                "section": self._extract_section(text),

                "document_url": document_url
            }
        )
    
    def _generate_document_id(self, url: str) -> str:
        """Generate a unique document ID from URL."""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def _extract_page_number(self, text: str) -> Optional[int]:
        """Extract page number from chunk text."""
        page_match = re.search(r'--- Page (\d+) ---', text)
        if page_match:
            return int(page_match.group(1))
        return None
    
    def _extract_section(self, text: str) -> Optional[str]:
        """Extract section information from chunk text with enhanced logic."""
        lines = text.split('\n')
        
        for line in lines[:5]:
            line = line.strip()
            if not line:
                continue
                
            section_patterns = [
                r'^(SECTION|Section)\s+[IVX\d]+[\.\:\-\s]*(.+)',
                r'^(ARTICLE|Article)\s+[IVX\d]+[\.\:\-\s]*(.+)',
                r'^(CLAUSE|Clause)\s+[IVX\d]+[\.\:\-\s]*(.+)',
                r'^(CHAPTER|Chapter)\s+[IVX\d]+[\.\:\-\s]*(.+)',
                r'^([IVX\d]+[\.\)]\s*.{5,50})',
                r'^([A-Z][A-Z\s]{10,50}):',
            ]
            
            for pattern in section_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(0).strip()
        
        document_sections = [
            'introduction', 'overview', 'definitions', 'terms', 'conditions',
            'procedures', 'requirements', 'limitations', 'restrictions',
            'responsibilities', 'obligations', 'rights', 'benefits'
        ]
        
        text_lower = text.lower()
        for section in document_sections:
            if section in text_lower:
                for line in lines[:3]:
                    if section in line.lower():
                        return line.strip()
                        
        return None
    
# Create aliases for compatibility
DocumentProcessor = OptimizedDocumentProcessor
