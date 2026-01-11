from typing import List, Optional, Dict, Any
from fastapi import UploadFile, HTTPException, BackgroundTasks
from loguru import logger
import os
import shutil
from pathlib import Path
from datetime import datetime
import asyncio
import numpy as np

from models import (
    DocumentResponse, DocumentStatus, DocumentDB, 
    DocumentStats, row_to_document_response
)
from utils.file_processing import process_file, get_file_info
from utils.vector_store import VectorStore
from database import get_db_connection, init_database
from config import settings

def rate_limit(max_requests: int = 5, window_seconds: int = 60):
    """Rate limiting decorator for document operations"""
    def decorator(func):
        requests = []
        async def wrapper(*args, **kwargs):
            now = datetime.now()
            requests[:] = [req for req in requests if (now - req).seconds < window_seconds]
            if len(requests) >= max_requests:
                raise HTTPException(status_code=429, detail="Too many document uploads")
            requests.append(now)
            return await func(*args, **kwargs)
        return wrapper
    return decorator

class DocumentService:
    """
    Service for handling document operations.
    
    This service handles:
    - Document upload and storage
    - Text chunking and embedding generation
    - Vector store management
    - Document metadata management
    """
    
    def __init__(self):
        """Initialize document service"""
        try:
            # Initialize paths
            self.upload_dir = settings.UPLOAD_DIR
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize vector store (no parameters needed)
            self.vector_store = VectorStore()
            
            # Configuration
            self.max_file_size = 10 * 1024 * 1024  # 10MB
            self.chunk_size = settings.CHUNK_SIZE
            self.chunk_overlap = settings.CHUNK_OVERLAP
            self.supported_extensions = {'.txt', '.pdf', '.docx', '.md'}
            
            # Initialize database
            init_database()
            
            logger.info("Document service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize document service: {str(e)}")
            raise

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if not text or not text.strip():
            return []
            
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunks.append(' '.join(chunk_words))
            
            if end >= len(words):
                break
                
            # Move start position with overlap
            start = end - self.chunk_overlap
        
        return chunks

    async def _process_document_chunks(self, document_id: int):
        """Background task to process document chunks and generate embeddings"""
        try:
            with get_db_connection() as conn:
                # Set row factory for dictionary access
                conn.row_factory = lambda cursor, row: {
                    col[0]: row[idx] for idx, col in enumerate(cursor.description)
                }
                cursor = conn.cursor()

                # Get document info including file_path
                cursor.execute("""
                    SELECT filename, file_path, content, chunk_count
                    FROM documents 
                    WHERE id = ?
                """, (document_id,))
                result = cursor.fetchone()
                
                if not result:
                    raise ValueError(f"Document {document_id} not found")

                filename = result['filename']
                file_path = result['file_path']
                content = result['content']
                
                # Process based on file type
                if filename.lower().endswith('.pdf') and file_path:
                    # Use page-aware processing for PDFs
                    try:
                        from utils.file_processing import process_pdf_with_pages, split_text_with_page_info
                        pages = process_pdf_with_pages(file_path)
                        chunks_data = split_text_with_page_info(pages, self.chunk_size, self.chunk_overlap)
                        logger.info(f"Processed PDF with {len(pages)} pages into {len(chunks_data)} chunks")
                    except Exception as e:
                        logger.warning(f"Failed to process PDF with page info: {e}. Falling back to text processing.")
                        # Fallback to text-based processing
                        text_chunks = self._split_text(content)
                        chunks_data = [{'text': chunk, 'page_number': None, 'total_pages': 1, 'start_page': None, 'end_page': None} for chunk in text_chunks]
                else:
                    # Text-based processing for non-PDF files
                    text_chunks = self._split_text(content)
                    chunks_data = [{'text': chunk, 'page_number': None, 'total_pages': 1, 'start_page': None, 'end_page': None} for chunk in text_chunks]
                
                if not chunks_data:
                    raise ValueError("No chunks generated from document content")
                
                logger.info(f"Generated {len(chunks_data)} chunks for document {document_id}")
                
                # Extract text for embeddings
                chunk_texts = [chunk_data['text'] for chunk_data in chunks_data]
                
                # Generate embeddings for chunks
                from .embedding_service import embedding_service
                embeddings = await embedding_service.get_embeddings(chunk_texts)
                
                if not embeddings or len(embeddings) != len(chunk_texts):
                    raise ValueError("Failed to generate embeddings for all chunks")
                
                logger.info(f"Generated embeddings for {len(embeddings)} chunks")
                
                # Store chunks with page information in database
                for chunk_index, (chunk_data, embedding) in enumerate(zip(chunks_data, embeddings)):
                    chunk_text = chunk_data['text']
                    page_number = chunk_data.get('page_number')
                    total_pages = chunk_data.get('total_pages', 1)
                    start_page = chunk_data.get('start_page')
                    end_page = chunk_data.get('end_page')
                    
                    if embedding:
                        # Ensure it's a simple list of floats, not nested or duplicated
                        if isinstance(embedding, list):
                            embedding_array = np.array(embedding, dtype=np.float32)
                        else:
                            embedding_array = np.array(embedding).astype(np.float32)
                        
                        # DEBUG: Check dimensions before storing
                        logger.info(f"DEBUG: Storing embedding with shape: {embedding_array.shape}")
                        
                        # Store as bytes
                        embedding_bytes = embedding_array.tobytes()
                    else:
                        embedding_bytes = None

                    # Store chunk in database with page information
                    cursor.execute("""
                        INSERT INTO chunks (
                            document_id, content, chunk_index, page_number, 
                            total_pages, start_page, end_page, embedding
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        document_id, chunk_text, chunk_index, page_number, 
                        total_pages, start_page, end_page,
                        embedding_bytes
                    ))  
                    
                    logger.info(f"Stored chunk {chunk_index} for document {document_id}")
                
                # Commit ALL chunks at once after the loop
                conn.commit()
                logger.info(f"Stored {len(chunks_data)} chunks with page information for document {document_id}")

                # Update document status to completed
                cursor.execute("""
                    UPDATE documents 
                    SET status = 'completed', chunk_count = ?
                    WHERE id = ?
                """, (len(chunks_data), document_id))
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {str(e)}")
            # Update document status to failed
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE documents 
                    SET status = 'failed' 
                    WHERE id = ?
                """, (document_id,))
                conn.commit()
            raise

    @rate_limit()
    async def upload_document(self, file: UploadFile, background_tasks: BackgroundTasks) -> DocumentResponse:
        """Process and store an uploaded document"""
        log_context = {"filename": file.filename, "operation": "upload_document"}
        
        try:
            logger.info("Starting document upload", **log_context)
            
            # Input validation
            if not file.filename:
                raise HTTPException(status_code=400, detail="No filename provided")
            
            file_extension = Path(file.filename).suffix.lower()
            if file_extension not in self.supported_extensions:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type. Supported: {', '.join(self.supported_extensions)}"
                )
            
            # Read file content
            content = await file.read()
            file_size = len(content)
            
            if file_size > self.max_file_size:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File too large. Maximum size: {self.max_file_size // (1024*1024)}MB"
                )
            
            if file_size == 0:
                raise HTTPException(status_code=400, detail="Empty file")
            
            # Save file to upload directory first (needed for process_file)
            file_path = self.upload_dir / file.filename
            counter = 1
            original_stem = file_path.stem
            
            # Handle duplicate filenames
            while file_path.exists():
                file_path = self.upload_dir / f"{original_stem}_{counter}{file_extension}"
                counter += 1
            
            with open(file_path, 'wb') as f:
                f.write(content)
            
            # Process file content - FIXED
            try:
                if file_extension == '.txt':
                    text_content = content.decode('utf-8')
                elif file_extension == '.pdf':
                    # For PDF files, use the process_file function with file path
                    text_content = process_file(file_path)
                else:
                    # For other formats, try to decode as text first
                    try:
                        text_content = content.decode('utf-8')
                    except UnicodeDecodeError:
                        # If that fails, try using process_file
                        text_content = process_file(file_path)
            except Exception as e:
                # Clean up the file if processing fails
                if file_path.exists():
                    os.remove(file_path)
                logger.error(f"Failed to process file content: {str(e)}", **log_context)
                raise HTTPException(status_code=400, detail="Failed to extract text from file")
            
            if not text_content or not text_content.strip():
                # Clean up the file if no content
                if file_path.exists():
                    os.remove(file_path)
                raise HTTPException(status_code=400, detail="No text content found in file")
            
            # Save document to database
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO documents (
                        filename, file_path, file_size, content, status, upload_date, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    file.filename,
                    str(file_path),
                    file_size,
                    text_content,
                    DocumentStatus.PROCESSING.value,
                    datetime.now(),
                    None
                ))
                
                document_id = cursor.lastrowid
                conn.commit()

            # Schedule background processing
            background_tasks.add_task(self._process_document_chunks, document_id)
            
            # Return response
            return DocumentResponse(
                id=document_id,
                filename=file.filename,
                status=DocumentStatus.PROCESSING.value,
                upload_date=datetime.now(),
                file_size=file_size,
                chunk_count=0,
                metadata={"file_type": file_extension}
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Document upload failed: {str(e)}", **log_context, exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to process document upload")

    async def get_documents(self) -> List[DocumentResponse]:
        """Get all documents"""
        try:
            with get_db_connection() as conn:
                conn.row_factory = lambda cursor, row: {
                    col[0]: row[idx] for idx, col in enumerate(cursor.description)
                }
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, filename, status, upload_date, file_size, chunk_count, metadata
                    FROM documents 
                    ORDER BY upload_date DESC
                """)
                rows = cursor.fetchall()
                
                return [row_to_document_response(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to fetch documents: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch documents")

    async def get_document(self, document_id: int) -> Optional[DocumentResponse]:
        """Get a specific document by ID"""
        try:
            with get_db_connection() as conn:
                conn.row_factory = lambda cursor, row: {
                    col[0]: row[idx] for idx, col in enumerate(cursor.description)
                }
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, filename, status, upload_date, file_size, chunk_count, metadata
                    FROM documents
                    WHERE id = ?
                """, (document_id,))
                row = cursor.fetchone()
                
                if row:
                    return row_to_document_response(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch document {document_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch document")

    async def delete_document(self, document_id: int) -> bool:
        """Delete a document and its associated data"""
        try:
            with get_db_connection() as conn:
                conn.row_factory = lambda cursor, row: {
                    col[0]: row[idx] for idx, col in enumerate(cursor.description)
                }
                cursor = conn.cursor()
                
                # Get document info
                cursor.execute("""
                    SELECT file_path, chunk_count
                    FROM documents
                    WHERE id = ?
                """, (document_id,))
                doc_info = cursor.fetchone()
                
                if not doc_info:
                    return False
                
                # Delete from vector store
                try:
                    chunk_ids = [f"doc_{document_id}_{i}" for i in range(doc_info['chunk_count'])]
                    self.vector_store.delete(chunk_ids)
                except Exception as e:
                    logger.warning(f"Failed to delete from vector store: {str(e)}")
                
                # Delete chunks from database
                cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                
                # Delete document from database
                cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
                
                # Delete physical file
                try:
                    if doc_info['file_path'] and os.path.exists(doc_info['file_path']):
                        os.remove(doc_info['file_path'])
                except Exception as e:
                    logger.warning(f"Failed to delete physical file: {str(e)}")
                
                conn.commit()
                logger.info(f"Successfully deleted document {document_id}")
                return True
                    
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to delete document")

    async def get_document_stats(self) -> DocumentStats:
        """Get document processing statistics"""
        try:
            with get_db_connection() as conn:
                conn.row_factory = lambda cursor, row: {
                    col[0]: row[idx] for idx, col in enumerate(cursor.description)
                }
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_documents,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_documents,
                        SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending_documents,
                        SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as processing_documents,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_documents,
                        COALESCE(SUM(chunk_count), 0) as total_chunks,
                        COALESCE(SUM(file_size), 0) as total_storage_size
                    FROM documents
                """)
                stats = cursor.fetchone()
                
                avg_chunks = 0.0
                if stats['completed_documents'] > 0:
                    avg_chunks = stats['total_chunks'] / stats['completed_documents']
                
                return DocumentStats(
                    total_documents=stats['total_documents'],
                    completed_documents=stats['completed_documents'],
                    pending_documents=stats['pending_documents'] + stats['processing_documents'],
                    failed_documents=stats['failed_documents'],
                    total_chunks=stats['total_chunks'],
                    average_chunks_per_document=round(avg_chunks, 2),
                    total_storage_size=stats['total_storage_size']
                )
                
        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to retrieve document statistics")

# Create singleton instance
document_service = DocumentService()