import asyncio
import io
import os
import re
import time
import hashlib
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple, Union
from urllib.parse import urlparse
import structlog
import httpx

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

from docx import Document
import email

from app.core.config import settings
from app.models.schemas import DocumentMetadata, EmbeddingChunk

logger = structlog.get_logger(__name__)


class OptimizedDocumentProcessor:
    """
    Production-ready Document Processor.
    
    Features:
    - Single-pass PyMuPDF in-memory stream extraction (no disk churn)
    - Deterministic SHA-256 byte-level document hashing
    - Non-destructive paragraph chunking preserving legal clauses and table layouts
    - Page-level metadata attribution for explainable retrieval
    - Multi-format support (PDF, DOCX, Email, TXT)
    """

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200):
        self.chunk_size = getattr(settings, "chunk_size", chunk_size) or 1200
        self.chunk_overlap = getattr(settings, "chunk_overlap", chunk_overlap) or 200
        self.batch_size = 150
        
        logger.info("OptimizedDocumentProcessor initialized", 
                    chunk_size=self.chunk_size, 
                    chunk_overlap=self.chunk_overlap,
                    pymupdf_available=fitz is not None)

    def generate_document_id(self, content: Union[bytes, str]) -> str:
        """
        Deterministic SHA-256 hash of document content (or string fallback).
        Ensures consistent cache hits regardless of expiring presigned URLs.
        """
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()[:16]

    # Backward compatibility alias
    def _generate_document_id(self, source: Union[bytes, str]) -> str:
        return self.generate_document_id(source)

    def resolve_cloud_url(self, url: str) -> Tuple[str, Optional[str]]:
        """
        Transforms cloud storage sharing URLs (Google Drive, Docs, Dropbox, OneDrive)
        into direct binary download endpoints.
        Returns: (resolved_url, optional_file_id)
        """
        clean_url = url.strip()

        # 1. Google Drive File URLs (/file/d/{id}/view, /open?id={id}, /uc?id={id})
        drive_file_match = re.search(r"drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:[^&]*&)*id=)([a-zA-Z0-9_-]+)", clean_url)
        if drive_file_match:
            file_id = drive_file_match.group(1)
            direct_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&authuser=0"
            return direct_url, file_id

        # 2. Google Docs
        docs_match = re.search(r"docs\.google\.com/document/d/([a-zA-Z0-9_-]+)", clean_url)
        if docs_match:
            doc_id = docs_match.group(1)
            return f"https://docs.google.com/document/d/{doc_id}/export?format=pdf", doc_id

        # 3. Google Sheets
        sheets_match = re.search(r"docs\.google\.com/spreadsheets/d/([a-zA-Z0-9_-]+)", clean_url)
        if sheets_match:
            sheet_id = sheets_match.group(1)
            return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=pdf", sheet_id

        # 4. Google Slides
        slides_match = re.search(r"docs\.google\.com/presentation/d/([a-zA-Z0-9_-]+)", clean_url)
        if slides_match:
            slide_id = slides_match.group(1)
            return f"https://docs.google.com/presentation/d/{slide_id}/export/pdf", slide_id

        # 5. Dropbox share links
        if "dropbox.com" in clean_url:
            if "dl=0" in clean_url:
                return clean_url.replace("dl=0", "dl=1"), None
            elif "dl=1" not in clean_url:
                sep = "&" if "?" in clean_url else "?"
                return f"{clean_url}{sep}dl=1", None

        return clean_url, None

    async def download_document(self, url: str) -> bytes:
        """Download document from URL with timeout and automated cloud drive translation."""
        try:
            clean_url = url.strip()

            # Handle Google Drive folder URLs: discover contained file(s)
            if "drive.google.com" in clean_url and "folders" in clean_url:
                logger.info("Detected Google Drive folder URL, scanning for files...", url=clean_url[:100])
                async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
                    folder_resp = await client.get(clean_url)
                    raw_ids = re.findall(r'ssk=\'[0-9]+:[^:]+:([a-zA-Z0-9_-]{25,50})', folder_resp.text)
                    cleaned_ids = []
                    for rid in raw_ids:
                        base_id = re.sub(r'-\d+(-\d+)*$', '', rid)
                        if len(base_id) >= 25 and base_id not in cleaned_ids:
                            cleaned_ids.append(base_id)
                    if cleaned_ids:
                        target_id = cleaned_ids[0]
                        logger.info("Found file inside Google Drive folder", file_id=target_id)
                        clean_url = f"https://drive.google.com/file/d/{target_id}/view"

            download_url, file_id = self.resolve_cloud_url(clean_url)
            logger.info("Downloading document", original_url=url[:100], download_url=download_url[:100])

            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                response = await client.get(download_url)
                response.raise_for_status()
                content = response.content

                # Check for Google Drive virus scan warning / confirm token
                if b"confirm=" in content:
                    confirm_match = re.search(r'confirm=([0-9A-Za-z_]+)', response.text)
                    if confirm_match:
                        confirm_token = confirm_match.group(1)
                        second_url = f"{download_url}&confirm={confirm_token}"
                        logger.info("Bypassing Google Drive virus scan confirmation", confirm_token=confirm_token)
                        response = await client.get(second_url)
                        response.raise_for_status()
                        content = response.content

                # If Google Drive returned an HTML preview wrapper instead of binary, retry fallback
                if file_id and (content.lstrip().startswith(b"<!DOCTYPE") or content.lstrip().startswith(b"<html")):
                    alt_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
                    logger.info("Google Drive returned HTML preview, retrying with uc export endpoint", alt_url=alt_url)
                    alt_resp = await client.get(alt_url)
                    if alt_resp.is_success and not alt_resp.content.lstrip().startswith(b"<!DOCTYPE"):
                        content = alt_resp.content

                logger.info("Document downloaded successfully", size_bytes=len(content), is_pdf=content.lstrip().startswith(b"%PDF"))
                return content
        except Exception as e:
            logger.error("Failed to download document", url=url, error=str(e))
            raise ValueError(f"Failed to download document: {str(e)}")

    def extract_text_from_pdf(self, content: bytes) -> Tuple[str, int]:
        """
        Single-pass in-memory PDF extraction using PyMuPDF (fitz) without disk churn.
        Extracts structural layout text blocks separating paragraphs with \n\n.
        Falls back to pdfplumber or PyPDF2 if PyMuPDF is not available or encounters errors.
        """
        if fitz:
            try:
                doc = fitz.open(stream=content, filetype="pdf")
                total_pages = len(doc)
                pages_text = []
                for page_idx in range(total_pages):
                    page = doc[page_idx]
                    # Block extraction: block type 0 is text (ignore images/drawings)
                    blocks = page.get_text("blocks")
                    text_blocks = [b[4].strip() for b in blocks if len(b) > 6 and b[6] == 0 and b[4].strip()]
                    if not text_blocks:
                        raw_text = page.get_text("text").strip()
                        if raw_text:
                            text_blocks = [raw_text]
                    
                    if text_blocks:
                        page_body = "\n\n".join(text_blocks)
                        pages_text.append(f"--- Page {page_idx + 1} ---\n{page_body}")
                doc.close()
                return "\n\n".join(pages_text), total_pages
            except Exception as e:
                logger.warning("PyMuPDF stream extraction failed, trying fallbacks", error=str(e))

        # Fallback 1: pdfplumber in-memory
        if pdfplumber:
            try:
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    total_pages = len(pdf.pages)
                    pages_text = []
                    for page_idx, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        if text.strip():
                            pages_text.append(f"--- Page {page_idx + 1} ---\n{text.strip()}")
                    return "\n\n".join(pages_text), total_pages
            except Exception as e:
                logger.warning("pdfplumber fallback failed", error=str(e))

        # Fallback 2: PyPDF2 in-memory
        if PyPDF2:
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(content))
                total_pages = len(reader.pages)
                pages_text = []
                for page_idx in range(total_pages):
                    text = reader.pages[page_idx].extract_text() or ""
                    if text.strip():
                        pages_text.append(f"--- Page {page_idx + 1} ---\n{text.strip()}")
                    return "\n\n".join(pages_text), total_pages
            except Exception as e:
                logger.error("PyPDF2 fallback failed", error=str(e))

        raise ValueError("All PDF extraction libraries failed or are not installed.")

    def extract_text_from_docx(self, content: bytes) -> Tuple[str, Optional[int]]:
        """Extract text and tables from DOCX in-memory."""
        doc = Document(io.BytesIO(content))
        elements = []
        for p in doc.paragraphs:
            if p.text.strip():
                elements.append(p.text.strip())
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    elements.append(row_text)
        return "\n\n".join(elements), None

    def extract_text_from_email(self, content: bytes) -> Tuple[str, Optional[int]]:
        """Extract text from Email in-memory."""
        msg = email.message_from_bytes(content)
        body = []
        for header in ["From", "To", "Subject", "Date"]:
            if msg.get(header):
                body.append(f"{header}: {msg.get(header)}")
        body.append("")
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body.append(payload.decode("utf-8", errors="ignore"))
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body.append(payload.decode("utf-8", errors="ignore"))
        return "\n\n".join(body), None

    def _split_large_segment(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Split a large paragraph/block into bounded segments on natural sentence or line boundaries."""
        if len(text) <= max_size:
            return [text]
        sub = []
        start = 0
        while start < len(text):
            end = start + max_size
            if end >= len(text):
                rem = text[start:].strip()
                if rem:
                    sub.append(rem)
                break
            # Look for newline or period boundary in the latter half of the window
            cut = text.rfind("\n", start + max_size // 2, end)
            if cut == -1:
                cut = text.rfind(". ", start + max_size // 2, end)
                if cut != -1:
                    cut += 1  # Include the period
            if cut == -1:
                cut = text.rfind(" ", start + max_size // 2, end)
            if cut == -1:
                cut = end
            segment = text[start:cut].strip()
            if segment:
                sub.append(segment)
            start = cut - overlap if cut - overlap > start else cut
        return [s for s in sub if s]

    def create_chunks(self, text: str, document_id: str) -> List[EmbeddingChunk]:
        """
        Structure-preserving paragraph and block-based chunking with bounded size.
        Ensures chunks strictly fit within embedding model token windows (e.g. 800-1200 chars)
        so no text is truncated or lost during retrieval indexing.
        """
        raw_paragraphs = text.split("\n\n")
        chunks: List[EmbeddingChunk] = []
        current_chunk_parts: List[str] = []
        current_length = 0
        current_page = 1

        def extract_page_num(s: str) -> Optional[int]:
            m = re.search(r"---\s*Page\s+(\d+)\s*---", s)
            return int(m.group(1)) if m else None

        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue

            page_match = extract_page_num(para)
            if page_match:
                current_page = page_match

            # Split large paragraphs into bite-sized segments
            segments = self._split_large_segment(para, max_size=self.chunk_size, overlap=self.chunk_overlap)
            for seg in segments:
                seg_len = len(seg)
                if current_length + seg_len > self.chunk_size and current_chunk_parts:
                    chunk_body = "\n\n".join(current_chunk_parts)
                    chunk_idx = len(chunks)
                    chunks.append(
                        EmbeddingChunk(
                            chunk_id=f"{document_id}_{chunk_idx}",
                            document_id=document_id,
                            text=chunk_body,
                            chunk_index=chunk_idx,
                            metadata={"page_number": current_page}
                        )
                    )

                    # Overlap: keep trailing segment
                    if self.chunk_overlap > 0 and len(current_chunk_parts) > 1:
                        last_part = current_chunk_parts[-1]
                        current_chunk_parts = [last_part, seg]
                        current_length = len(last_part) + seg_len
                    else:
                        current_chunk_parts = [seg]
                        current_length = seg_len
                else:
                    current_chunk_parts.append(seg)
                    current_length += seg_len

        if current_chunk_parts:
            chunk_body = "\n\n".join(current_chunk_parts)
            chunk_idx = len(chunks)
            chunks.append(
                EmbeddingChunk(
                    chunk_id=f"{document_id}_{chunk_idx}",
                    document_id=document_id,
                    text=chunk_body,
                    chunk_index=chunk_idx,
                    metadata={"page_number": current_page}
                )
            )

        logger.info("Chunking completed", document_id=document_id, total_chunks=len(chunks))
        return chunks

    # Alias for backward compatibility
    def _create_chunks(self, text: str, document_url: str) -> List[EmbeddingChunk]:
        doc_id = self.generate_document_id(document_url.encode("utf-8"))
        return self.create_chunks(text, doc_id)

    async def process_document(self, url: str) -> Tuple[DocumentMetadata, List[EmbeddingChunk]]:
        """Process document into DocumentMetadata and list of EmbeddingChunks."""
        start_time = time.time()
        content = await self.download_document(url)
        document_id = self.generate_document_id(content)

        parsed_url = urlparse(url)
        ext = os.path.splitext(parsed_url.path)[1].lower()
        stripped = content.lstrip()
        if stripped.startswith(b"%PDF"):
            ext = ".pdf"
        elif stripped.startswith(b"PK"):
            ext = ".docx"
        elif not ext:
            ext = ".txt"

        if ext == ".pdf":
            text_content, pages = self.extract_text_from_pdf(content)
        elif ext == ".docx":
            text_content, pages = self.extract_text_from_docx(content)
        elif ext in [".eml", ".msg"]:
            text_content, pages = self.extract_text_from_email(content)
        else:
            text_content = content.decode("utf-8", errors="ignore")
            pages = None

        chunks = self.create_chunks(text_content, document_id)
        metadata = DocumentMetadata(
            document_id=document_id,
            filename=os.path.basename(parsed_url.path) or "document",
            size_bytes=len(content),
            pages=pages,
            processing_time=time.time() - start_time,
            chunks_created=len(chunks)
        )
        return metadata, chunks

    async def process_document_streaming(
        self, url: str
    ) -> Tuple[DocumentMetadata, AsyncGenerator[List[EmbeddingChunk], None]]:
        """Streaming generator interface yielding batches of EmbeddingChunk."""
        metadata, all_chunks = await self.process_document(url)

        async def chunk_generator():
            batch_size = self.batch_size
            for i in range(0, len(all_chunks), batch_size):
                yield all_chunks[i:i + batch_size]

        return metadata, chunk_generator()


# Compatibility alias
DocumentProcessor = OptimizedDocumentProcessor
