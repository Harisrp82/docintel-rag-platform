# utils/__init__.py
"""
Utility functions and classes for the RAG system.

This package contains helper functions for file processing,
vector operations, and other utility functions.
"""

# Package version
__version__ = "1.0.0"

# Import key functions and classes for convenient access
try:
    from .file_processing import (
        process_file,
        validate_file_type,
        get_file_info,
        format_file_size,
        intelligent_chunk_text,
        simple_chunk_text
    )
except ImportError:
    # Handle case where file_processing.py doesn't exist yet
    pass

try:
    from .vector_store import VectorStore
except ImportError:
    # Handle case where vector_store.py doesn't exist yet
    pass

# Define what gets imported with "from utils import *"
__all__ = [
    # File processing functions
    'process_file',
    'validate_file_type',
    'get_file_info',
    'format_file_size',
    'intelligent_chunk_text',
    'simple_chunk_text',
    
    # Vector store class
    'VectorStore',
]

# Package-level configuration
SUPPORTED_FILE_TYPES = ['.pdf', '.txt']
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200