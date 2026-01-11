"""
Service layer implementations for the RAG system.

This package contains the core business logic and services
for document processing, chat, and retrieval operations.
"""

from .chat_service import chat_service
from .document_service import document_service

__all__ = ['chat_service', 'document_service']

# Package version
__version__ = "1.0.0"