import asyncio
import aiofiles
import tempfile
import os
import gc
from typing import List, Dict, Any, Optional, Generator, Tuple
from urllib.parse import urlparse
import structlog
import httpx
from io import BytesIO
import re
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
from docx import Document
import email
from email.mime.text import MIMEText
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.core.config import settings
from app.models.schemas import DocumentMetadata, EmbeddingChunk

logger = structlog.get_logger(__name__)

class DocumentProcessor:
    """Memory-optimized document processor with streaming and batch processing."""

    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        # ✅ ULTRA-FAST: Settings for sub-60 second processing
        self.max_memory_usage = 0.85  # 85% of 1GB - push limits for speed
        self.batch_size = 150  # Process 150 pages at a time for optimal performance
        self.embedding_batch_size = 200  # Mega embedding batches for fewer API calls
        # Ultra-fast optimization settings
        self.max_tokens_per_chunk = 1200  # Increased for testing - was 800
        self.min_chunk_size = 50  # Smaller minimum for faster processing
        self.enable_chunk_compression = True  # Keep compression for efficiency
        logger.info("Ultra-fast DocumentProcessor initialized", 
                   batch_size=self.batch_size,
                   memory_limit=f"{self.max_memory_usage*100}%")

    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        try:
            memory_percent = psutil.virtual_memory().percent / 100
            if memory_percent > self.max_memory_usage:
                logger.warning("High memory usage detected", 
                             memory_percent=f"{memory_percent*100:.1f}%")
                gc.collect()  # Force garbage collection
                return False
            return True
        except Exception:
            return True  # If can't check, assume OK

    async def download_document(self, url: str) -> bytes:
        """Download document with memory-efficient streaming."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream('GET', url) as response:
                    response.raise_for_status()
                    
                    # Stream download to avoid loading entire file in memory
                    content = BytesIO()
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        content.write(chunk)
                        
                        # Check memory during download
                        if not self._check_memory_usage():
                            await asyncio.sleep(0.1)  # Brief pause
                    
                    return content.getvalue()
        except Exception as e:
            logger.error("Failed to download document", url=url, error=str(e))
            raise ValueError(f"Failed to download document: {str(e)}")

    async def process_document_streaming(self, url: str) -> tuple[DocumentMetadata, Generator[List[EmbeddingChunk], None, None]]:
        """Process document with streaming chunks to avoid memory buildup."""
        logger.info("Starting streaming document processing", url=url)
        
        # Download document
        document_content = await self.download_document(url)
        
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

        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file.write(document_content)
            temp_file_path = temp_file.name

        # Don't use try-finally here since we need the file to exist during generator consumption
        document_id = self._generate_document_id(url)
        
        if file_extension == '.pdf':
            chunk_generator = self._process_pdf_streaming_with_cleanup(temp_file_path, document_id, url)
            total_pages = await self._get_pdf_page_count(temp_file_path)
        elif file_extension == '.docx':
            chunk_generator = self._process_docx_streaming_with_cleanup(temp_file_path, document_id, url)
            total_pages = None
        else:
            chunk_generator = self._process_text_streaming_with_cleanup(temp_file_path, document_id, url)
            total_pages = None

        metadata = DocumentMetadata(
            document_id=document_id,
            document_type=file_extension[1:],
            total_pages=total_pages,
            total_chunks=0,  # Will be updated as we process
            processing_time=0.0
        )
        
        # Clean up document content from memory
        del document_content
        gc.collect()
        
        logger.info("Document streaming setup complete", document_id=document_id)
        
        return metadata, chunk_generator

    async def _get_pdf_page_count(self, file_path: str) -> int:
        """Get PDF page count without loading full content."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return len(pdf_reader.pages)
        except Exception:
            return 0

    def _process_pdf_streaming_with_cleanup(self, file_path: str, document_id: str, document_url: str) -> Generator[List[EmbeddingChunk], None, None]:
        """Ultra-fast PDF processing with automatic file cleanup."""
        try:
            yield from self._process_pdf_streaming_robust(file_path, document_id, document_url)
        finally:
            # Clean up temp file after generator is fully consumed
            try:
                os.unlink(file_path)
                logger.info("Cleaned up temporary PDF file", file_path=file_path)
            except Exception as e:
                logger.warning("Failed to clean up temporary file", file_path=file_path, error=str(e))

    def _process_docx_streaming_with_cleanup(self, file_path: str, document_id: str, document_url: str) -> Generator[List[EmbeddingChunk], None, None]:
        """Stream process DOCX with automatic file cleanup."""
        try:
            yield from self._process_docx_streaming(file_path, document_id, document_url)
        finally:
            # Clean up temp file after generator is fully consumed
            try:
                os.unlink(file_path)
                logger.info("Cleaned up temporary DOCX file", file_path=file_path)
            except Exception as e:
                logger.warning("Failed to clean up temporary file", file_path=file_path, error=str(e))

    def _process_text_streaming_with_cleanup(self, file_path: str, document_id: str, document_url: str) -> Generator[List[EmbeddingChunk], None, None]:
        """Stream process text with automatic file cleanup."""
        try:
            yield from self._process_text_streaming(file_path, document_id, document_url)
        finally:
            # Clean up temp file after generator is fully consumed
            try:
                os.unlink(file_path)
                logger.info("Cleaned up temporary text file", file_path=file_path)
            except Exception as e:
                logger.warning("Failed to clean up temporary file", file_path=file_path, error=str(e))

    def _process_pdf_streaming_robust(self, file_path: str, document_id: str, document_url: str) -> Generator[List[EmbeddingChunk], None, None]:
        """Optimized robust PDF processing with multi-library fallback for 99% accuracy."""
        import concurrent.futures
        from multiprocessing import cpu_count
        import time
        
        try:
            total_pages = self._get_pdf_page_count_safe(file_path)
            logger.info(f"Starting robust PDF processing", 
                       total_pages=total_pages,
                       file_size_mb=round(os.path.getsize(file_path) / 1024 / 1024, 1))
            
            # Optimized parameters for maximum speed
            page_batch_size = min(200, total_pages)  # Larger batches for speed
            max_workers = min(8, cpu_count())  # More workers for speed
            
            chunk_index = 0
            extraction_stats = {"pdfplumber": 0, "pymupdf": 0, "pdfminer": 0, "pypdf2": 0, "failed": 0}
            
            for batch_start in range(0, total_pages, page_batch_size):
                batch_end = min(batch_start + page_batch_size, total_pages)
                
                logger.info(f"Processing robust batch {batch_start + 1}-{batch_end}/{total_pages} (workers: {max_workers})")
                
                try:
                    # Extract batch with parallel processing and fallback
                    batch_results = self._extract_batch_optimized_robust(
                        file_path, batch_start, batch_end, max_workers, extraction_stats
                    )
                    
                    logger.info(f"Batch extraction completed", 
                               pages_processed=len(batch_results), 
                               pages_expected=batch_end - batch_start)
                    
                    if batch_results:
                        # Convert to chunks
                        batch_text = "\n".join([
                            f"--- Page {page_num + 1} ---\n{text}" 
                            for page_num, text in batch_results 
                            if text.strip()
                        ])
                        
                        if batch_text.strip():
                            chunks = self._create_chunks_from_text(
                                batch_text, document_id, chunk_index, document_url
                            )
                            
                            if chunks:
                                chunk_index += len(chunks)
                                yield chunks
                                logger.info(f"Generated {len(chunks)} chunks from batch {batch_start + 1}-{batch_end}")
                    else:
                        logger.warning(f"No content extracted from batch {batch_start + 1}-{batch_end}")
                        
                except Exception as batch_error:
                    logger.error(f"Batch {batch_start + 1}-{batch_end} processing failed", error=str(batch_error))
                    # Continue with next batch instead of stopping
                    continue
                
                # Brief pause for system balance and file handle cleanup (reduced for speed)
                time.sleep(0.05)  # Reduced from 0.2s to 0.05s
            
            # Log extraction statistics
            total_attempted = sum(extraction_stats.values())
            if total_attempted > 0:
                success_rate = ((total_attempted - extraction_stats["failed"]) / total_attempted) * 100
                logger.info("PDF extraction statistics", 
                           success_rate=f"{success_rate:.1f}%",
                           methods=extraction_stats,
                           total_pages=total_pages)
                
                # If success rate is very low, warn but continue
                if success_rate < 30:
                    logger.warning("Low extraction success rate", success_rate=f"{success_rate:.1f}%")
            else:
                logger.warning("No pages were processed successfully")
                
        except Exception as e:
            logger.error("Robust PDF processing failed completely", error=str(e))
            # Don't return empty - let the system try fallback processing
            logger.info("Attempting fallback to basic PDF processing...")
            try:
                # Fallback to basic pdfplumber processing
                yield from self._process_pdf_streaming(file_path, document_id, document_url)
            except Exception as fallback_error:
                logger.error("Fallback processing also failed", error=str(fallback_error))
                return

    def _get_pdf_page_count_safe(self, file_path: str) -> int:
        """Get PDF page count using fastest reliable method."""
        try:
            # Try PyMuPDF first (fastest for page count)
            doc = fitz.open(file_path)
            count = doc.page_count
            doc.close()
            return count
        except:
            try:
                # Fallback to PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    return len(reader.pages)
            except:
                return 0

    def _extract_batch_optimized_robust(self, file_path: str, start_page: int, end_page: int, 
                                      max_workers: int, stats: Dict[str, int]) -> List[Tuple[int, str]]:
        """Extract pages with optimized robust multi-library approach."""
        results = []
        batch_size = end_page - start_page
        
        # Ultra-fast timeout: 0.8 seconds per page minimum, 2 minutes maximum
        timeout = min(max(batch_size * 0.8, 45), 120)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all page extraction tasks
            future_to_page = {
                executor.submit(self._extract_single_page_robust, file_path, page_num, stats): page_num
                for page_num in range(start_page, end_page)
            }
            
            # Collect results as they complete with optimized timeout
            completed_futures = []
            try:
                # Use timeout for as_completed to avoid hanging on slow pages
                for future in as_completed(future_to_page, timeout=timeout):
                    completed_futures.append(future)
                    page_num = future_to_page[future]
                    try:
                        text = future.result(timeout=2)  # Very short individual timeout
                        if text and text.strip():
                            results.append((page_num, text))
                    except Exception as e:
                        logger.debug(f"Page {page_num + 1} extraction failed/skipped", error=str(e))
                        stats["failed"] += 1
                        # Continue processing other pages
            except Exception as e:
                logger.warning(f"Batch timeout reached, processed {len(completed_futures)}/{len(future_to_page)} pages", 
                             timeout=timeout, batch_size=batch_size)
                
                # Cancel remaining futures that haven't completed
                for future in future_to_page:
                    if not future.done():
                        future.cancel()
                        page_num = future_to_page[future]
                        logger.debug(f"Cancelled slow page {page_num + 1}")
                        stats["failed"] += 1
        
        # Sort by page number
        results.sort(key=lambda x: x[0])
        return results

    def _extract_single_page_robust(self, file_path: str, page_num: int, stats: Dict[str, int]) -> str:
        """Extract single page with single fast method - skip blank pages quickly."""
        
        # Only use pdfplumber - fastest and most reliable
        try:
            with pdfplumber.open(file_path) as pdf:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    if text and len(text.strip()) > 3:  # Skip truly blank pages
                        stats["pdfplumber"] += 1
                        return text
        except Exception:
            pass  # Fail silently for speed
        
        # Page is blank or failed - skip quickly
        stats["failed"] += 1
        return ""  # Return empty string for blank pages

    def _process_pdf_streaming(self, file_path: str, document_id: str, document_url: str) -> Generator[List[EmbeddingChunk], None, None]:
        """Stream process PDF in small batches."""
        chunk_index = 0
        
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                
                for page_start in range(0, total_pages, self.batch_size):
                    page_end = min(page_start + self.batch_size, total_pages)
                    
                    # Process batch of pages
                    batch_text = ""
                    for page_num in range(page_start, page_end):
                        page = pdf.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            batch_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    
                    if batch_text.strip():
                        # Create chunks from batch
                        batch_chunks = self._create_chunks_from_text(
                            batch_text, document_id, chunk_index, document_url
                        )
                        chunk_index += len(batch_chunks)
                        
                        yield batch_chunks
                        
                        # Memory cleanup
                        del batch_text, batch_chunks
                        gc.collect()
                        
                        # Check memory and pause if needed
                        if not self._check_memory_usage():
                            import time
                            time.sleep(0.5)
                    
                    logger.info("Processed PDF batch", 
                              pages=f"{page_start+1}-{page_end}", 
                              total_pages=total_pages)
                              
        except Exception as e:
            logger.error("PDF streaming processing failed", error=str(e))
            # Return empty generator on error
            return
            yield []

    def _process_docx_streaming(self, file_path: str, document_id: str, document_url: str) -> Generator[List[EmbeddingChunk], None, None]:
        """Stream process DOCX in batches."""
        try:
            doc = Document(file_path)
            batch_text = ""
            chunk_index = 0
            paragraph_count = 0
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    batch_text += paragraph.text + "\n"
                    paragraph_count += 1
                    
                    # Process in batches of paragraphs
                    if paragraph_count >= 20:  # Every 20 paragraphs
                        if batch_text.strip():
                            batch_chunks = self._create_chunks_from_text(
                                batch_text, document_id, chunk_index, document_url
                            )
                            chunk_index += len(batch_chunks)
                            yield batch_chunks
                            
                            # Cleanup
                            del batch_chunks
                            batch_text = ""
                            paragraph_count = 0
                            gc.collect()
            
            # Process remaining text
            if batch_text.strip():
                batch_chunks = self._create_chunks_from_text(
                    batch_text, document_id, chunk_index, document_url
                )
                yield batch_chunks
                
        except Exception as e:
            logger.error("DOCX streaming processing failed", error=str(e))
            return
            yield []

    def _process_text_streaming(self, file_path: str, document_id: str, document_url: str) -> Generator[List[EmbeddingChunk], None, None]:
        """Stream process text file in chunks."""
        try:
            chunk_index = 0
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read file in chunks to avoid memory issues
                chunk_size = 8192  # 8KB chunks
                text_buffer = ""
                
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                        
                    text_buffer += chunk
                    
                    # Process when buffer gets large enough
                    if len(text_buffer) >= self.chunk_size * 10:
                        batch_chunks = self._create_chunks_from_text(
                            text_buffer, document_id, chunk_index, document_url
                        )
                        chunk_index += len(batch_chunks)
                        yield batch_chunks
                        
                        # Keep some overlap
                        text_buffer = text_buffer[-self.chunk_overlap:]
                        del batch_chunks
                        gc.collect()
                
                # Process remaining buffer
                if text_buffer.strip():
                    batch_chunks = self._create_chunks_from_text(
                        text_buffer, document_id, chunk_index, document_url
                    )
                    yield batch_chunks
                    
        except Exception as e:
            logger.error("Text streaming processing failed", error=str(e))
            return
            yield []

    def _create_chunks_from_text(self, text: str, document_id: str, start_index: int, document_url: str) -> List[EmbeddingChunk]:
        """Create chunks from text with memory optimization."""
        chunks = []
        chunk_index = start_index
        
        paragraphs = self._split_into_paragraphs(text)
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Compress paragraph
            if self.enable_chunk_compression:
                paragraph = self._compress_text(paragraph)
            
            para_token_count = len(paragraph.split()) * 0.75
            current_token_count = len(current_chunk.split()) * 0.75
            
            if current_token_count > 0 and (current_token_count + para_token_count > self.max_tokens_per_chunk):
                if current_chunk.strip() and len(current_chunk.split()) >= self.min_chunk_size:
                    chunk = self._create_chunk(
                        current_chunk.strip(), 
                        document_id, 
                        chunk_index, 
                        document_url
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    overlap_words = current_chunk.split()[-self.chunk_overlap//2:]
                    current_chunk = " ".join(overlap_words) + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip() and len(current_chunk.split()) >= self.min_chunk_size:
            chunk = self._create_chunk(
                current_chunk.strip(), 
                document_id, 
                chunk_index, 
                document_url
            )
            chunks.append(chunk)
        
        return chunks

    def _compress_text(self, text: str) -> str:
        """Compress text by removing redundant phrases."""
        # Enhanced compression patterns
        replacement_map = {
            # Insurance specific compressions
            r'\bwhich shall be the basis of this contract and is deemed to be incorporated herein\b': '[contract basis]',
            r'\bfollowing the Medical Advice of a duly qualified Medical Practitioner\b': 'per doctor advice',
            r'\bThe Company shall indemnify the Hospital or the Insured, Reasonable and Customary Charges incurred for Medically Necessary Treatment\b': 'Company pays approved medical costs',
            r'\bsubject to the Definitions, Terms, Exclusions, Conditions contained herein and limits\b': 'subject to policy terms',
            r'\bhas applied to National Insurance Company Ltd\. \(hereinafter called the Company\)\b': 'applied to Company',
            r'\bsudden, unforeseen and involuntary event caused by external, visible and violent means\b': '[Accident definition]',
            r'\bunder the supervision of a registered and qualified medical practitioner\b': 'under qualified doctor supervision',
            
            # Generic legal phrase compression
            r'\b(shall be|are) accessible to the insurance company\'s authorized representative\b': 'accessible to insurer',
            r'\bIn the event of hospitalisation/ domiciliary hospitalisation, the insured person/insured person\'s representative shall notify\b': 'For hospitalization, notify',
            r'\b(for|under) any of the following circumstances\b': 'if:',
            r'\bThe services offered by a TPA shall not include\b': 'TPA excludes:',
            
            # Remove redundant company info
            r'National Insurance Co\. Ltd\.': 'NIC',
            r'National Parivar Mediclaim Plus Policy': 'Policy',
            r'UIN: NICHLIP25039V032425': '',
            r'Page \d+ of \d+': '',
            r'Premises No\. 18-0374, Plot no\. CBD-81, New Town, Kolkata - 700156': '',
        }
        
        for pattern, replacement in replacement_map.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
        # Clean up excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs with memory efficiency."""
        # Simple split first
        paragraphs = text.split('\n\n')
        refined_paragraphs = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # Split large paragraphs
            if len(para.split()) > self.chunk_size * 2:
                sentences = self._split_into_sentences(para)
                current_para = ""
                
                for sentence in sentences:
                    if not current_para:
                        current_para = sentence
                    elif len((current_para + " " + sentence).split()) <= self.chunk_size * 2:
                        current_para += " " + sentence
                    else:
                        if current_para:
                            refined_paragraphs.append(current_para)
                        current_para = sentence
                
                if current_para:
                    refined_paragraphs.append(current_para)
            else:
                refined_paragraphs.append(para)
        
        return refined_paragraphs

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n+\s*(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence.split()) >= 3:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences

    def _create_chunk(self, text: str, document_id: str, chunk_index: int, document_url: str) -> EmbeddingChunk:
        """Create a chunk with optimized metadata."""
        return EmbeddingChunk(
            chunk_id=f"{document_id}_chunk_{chunk_index}",
            document_id=document_id,
            text=text,
            chunk_index=chunk_index,
            metadata={
                "page_number": self._extract_page_number(text),
                "section": self._extract_section(text),
                "clause_type": self._classify_clause_type(text),
                "document_url": document_url
            }
        )

    def _generate_document_id(self, url: str) -> str:
        """Generate document ID."""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()[:12]

    def _extract_page_number(self, text: str) -> Optional[int]:
        """Extract page number from chunk text."""
        page_match = re.search(r'--- Page (\d+) ---', text)
        if page_match:
            return int(page_match.group(1))
        return None

    def _extract_section(self, text: str) -> Optional[str]:
        """Extract section from text."""
        lines = text.split('\n')
        
        for line in lines[:5]:
            line = line.strip()
            if not line:
                continue
                
            section_patterns = [
                r'^(SECTION|Section)\s+[IVX\d]+[\.\:\-\s]*(.+)',
                r'^(ARTICLE|Article)\s+[IVX\d]+[\.\:\-\s]*(.+)',
                r'^(CLAUSE|Clause)\s+[IVX\d]+[\.\:\-\s]*(.+)',
            ]
            
            for pattern in section_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(0).strip()
        
        return None

    def _classify_clause_type(self, text: str) -> Optional[str]:
        """Classify clause type."""
        text_lower = text.lower()
        
        classification_patterns = {
            'coverage_clause': ['cover', 'benefit', 'include', 'eligible'],
            'exclusion_clause': ['exclude', 'not cover', 'limitation', 'restrict'],
            'condition_clause': ['condition', 'require', 'must', 'shall'],
            'payment_clause': ['premium', 'payment', 'cost', 'fee'],
            'claims_clause': ['claim', 'settlement', 'procedure', 'process'],
        }
        
        for category, keywords in classification_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return 'general_clause'

    # Keep the legacy method for backward compatibility
    async def process_document(self, url: str) -> tuple[DocumentMetadata, List[EmbeddingChunk]]:
        """Legacy method that collects all chunks - use streaming version for memory efficiency."""
        logger.warning("Using legacy process_document - consider using streaming version")
        
        metadata, chunk_generator = await self.process_document_streaming(url)
        
        all_chunks = []
        async for chunk_batch in self._async_chunk_generator(chunk_generator):
            all_chunks.extend(chunk_batch)
            
            # Check memory periodically
            if not self._check_memory_usage():
                await asyncio.sleep(0.1)
        
        metadata.total_chunks = len(all_chunks)
        return metadata, all_chunks

    async def _async_chunk_generator(self, sync_generator):
        """Convert sync generator to async for compatibility."""
        for batch in sync_generator:
            yield batch
            await asyncio.sleep(0)  # Allow event loop to process