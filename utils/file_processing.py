"""
File processing utilities for handling documents in the RAG system.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pdfplumber  # Changed from fitz to pdfplumber
from loguru import logger

SUPPORTED_FILE_TYPES = ['.pdf', '.txt']
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

def validate_file_type(file_path: str | Path) -> bool:
    """
    Validate if the file type is supported.
    
    Args:
        file_path: Path to the file
    
    Returns:
        bool: True if file type is supported, False otherwise
    """
    return Path(file_path).suffix.lower() in SUPPORTED_FILE_TYPES

def get_file_info(file_path: str | Path) -> Dict[str, any]:
    """
    Get file information including size, type, and creation date.
    
    Args:
        file_path: Path to the file
    
    Returns:
        Dict containing file metadata
    """
    path = Path(file_path)
    stats = path.stat()
    
    return {
        "filename": path.name,
        "file_type": path.suffix.lower(),
        "file_size": stats.st_size,
        "created_at": stats.st_ctime,
        "modified_at": stats.st_mtime
    }

def format_file_size(size_in_bytes: int) -> str:
    """
    Format file size from bytes to human readable format.
    
    Args:
        size_in_bytes: File size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.1f} TB"

def process_file(file_path: str | Path) -> str:
    """
    Process different file types and extract text content.
    
    Args:
        file_path: Path to the file
    
    Returns:
        Extracted text content
    
    Raises:
        ValueError: If file type is not supported
        FileNotFoundError: If file doesn't exist
    """
    if not validate_file_type(file_path):
        raise ValueError(f"Unsupported file type. Supported types: {SUPPORTED_FILE_TYPES}")
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if path.suffix.lower() == '.pdf':
        return _process_pdf(path)
    elif path.suffix.lower() == '.txt':
        return _process_txt(path)
    
    raise ValueError(f"No processor available for {path.suffix}")

def _process_pdf(file_path: Path) -> str:
    """Process PDF files and extract text using pdfplumber."""
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
        raise

def _process_txt(file_path: Path) -> str:
    """Process text files and extract content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"Error processing text file {file_path}: {str(e)}")
        raise

def split_text_into_chunks(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If this isn't the last chunk, try to break at a word boundary
        if end < len(text):
            # Look for the last space within the last 100 characters
            search_start = max(start + chunk_size - 100, start)
            last_space = text.rfind(' ', search_start, end)
            if last_space > start:
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move to next chunk with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks
# Add these functions to utils/file_processing.py

def process_pdf_with_pages(file_path: str | Path) -> List[Dict[str, any]]:
    """
    Process PDF files and extract text with page information preserved.
    
    Returns:
        List of dictionaries with 'page_number', 'text', and 'total_pages'
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with pdfplumber.open(path) as pdf:
            pages = []
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    pages.append({
                        'page_number': page_num,
                        'text': page_text.strip(),
                        'total_pages': total_pages
                    })
            
            return pages
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
        raise

def process_pdf_with_pages(file_path: str | Path) -> List[Dict[str, any]]:
    """
    Process PDF files and extract text with page information preserved.
    
    Returns:
        List of dictionaries with 'page_number', 'text', and 'total_pages'
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with pdfplumber.open(path) as pdf:
            pages = []
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    pages.append({
                        'page_number': page_num,
                        'text': page_text.strip(),
                        'total_pages': total_pages
                    })
            
            return pages
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
        raise

def split_text_with_page_info(pages: List[Dict[str, any]], chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Dict[str, any]]:
    """
    Split text into chunks while preserving page information.
    
    Args:
        pages: List of page dictionaries from process_pdf_with_pages
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of chunk dictionaries with page information
    """
    chunks = []
    
    for page_info in pages:
        page_text = page_info['text']
        page_number = page_info['page_number']
        total_pages = page_info['total_pages']
        
        if len(page_text) <= chunk_size:
            # Page fits in one chunk
            chunks.append({
                'text': page_text,
                'page_number': page_number,
                'total_pages': total_pages,
                'start_page': page_number,
                'end_page': page_number
            })
        else:
            # Split page into multiple chunks
            start = 0
            
            while start < len(page_text):
                end = start + chunk_size
                
                # Try to break at word boundary
                if end < len(page_text):
                    search_start = max(start + chunk_size - 100, start)
                    last_space = page_text.rfind(' ', search_start, end)
                    if last_space > start:
                        end = last_space
                
                chunk_text = page_text[start:end].strip()
                if chunk_text:
                    chunks.append({
                        'text': chunk_text,
                        'page_number': page_number,
                        'total_pages': total_pages,
                        'start_page': page_number,
                        'end_page': page_number
                    })
                
                if end >= len(page_text):
                    break
                
                # Move start position with overlap
                start = end - overlap
    
    return chunks