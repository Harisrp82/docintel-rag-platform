from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from loguru import logger

# Enums for status tracking
class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Request Models
class DocumentUpload(BaseModel):
    """Model for document upload request"""
    filename: str = Field(..., description="Name of the uploaded file")
    content: str = Field(..., description="Text content of the document")

# Add LLM settings configuration model
class LLMConfig(BaseModel):
    """Model for LLM configuration"""
    model_name: str = Field(default="gpt-4o-mini", description="LLM model name")
    max_tokens: int = Field(default=4000, description="Maximum tokens for response")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Temperature for response generation")
    system_prompt: str = Field(
        default="You are a helpful AI assistant. Answer questions based on the provided context.",
        description="System prompt for the LLM"
    )

# Add embedding configuration model
class EmbeddingConfig(BaseModel):
    """Model for embedding configuration"""
    model_name: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name"
    )
    chunk_size: int = Field(default=1000, description="Text chunk size")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")

# Add API configuration model
class APIConfig(BaseModel):
    """Model for API configuration"""
    host: str = Field(default="localhost", description="API host")
    port: int = Field(default=8000, description="API port")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "host": "localhost",
                "port": 8000
            }
        }
    )

# Add configuration validator
class AppConfig(BaseModel):
    """Main application configuration"""
    api_key: str = Field(..., description="OpenAI API key")
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    vector_db_path: str = Field(
        default="./data/chroma_db",
        description="Path to vector database"
    )
    upload_dir: str = Field(
        default="./uploads",
        description="Path to upload directory"
    )

# Add a query parameters model
class QueryParameters(BaseModel):
    """Model for query parameters"""
    max_chunks: int = Field(default=3, ge=1, le=10, description="Maximum chunks to retrieve")
    similarity_threshold: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0, 
        description="Minimum similarity score"
    )
    include_metadata: bool = Field(
        default=True,
        description="Include chunk metadata in response"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "max_chunks": 3,
                "similarity_threshold": 0.7,
                "include_metadata": True
            }
        }
    )

class ChatQuery(BaseModel):
    """Model for chat query request"""
    question: str = Field(..., min_length=1, max_length=1000, description="User's question")
    parameters: Optional[QueryParameters] = Field(
        default=None,  # ← CRITICAL FIX: Changed from default_factory=QueryParameters
        description="Query parameters"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is this document about?",
                "parameters": {
                    "max_chunks": 3,
                    "similarity_threshold": 0.7,
                    "include_metadata": True
                }
            }
        }
    )

# Response Models
class DocumentResponse(BaseModel):
    id: int
    filename: str
    status: str
    upload_date: datetime
    file_size: int = 0  # Added default
    chunk_count: int = 0  # Added default
    metadata: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "filename": "example.pdf",
                "upload_date": "2024-01-15T10:30:00",
                "file_size": 1024000,
                "chunk_count": 15,
                "status": "completed",
                "metadata": {"file_type": "pdf", "pages": 10}
            }
        }
    )

class ChunkResponse(BaseModel):
    """Model for document chunk response"""
    id: int = Field(..., description="Chunk ID")
    document_id: int = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Index of chunk within document")
    embedding_id: Optional[str] = Field(default=None, description="Vector store embedding ID")

class RetrievedChunk(BaseModel):
    """Model for retrieved document chunks with similarity scores"""
    content: str = Field(..., description="Chunk content")
    metadata: Dict[str, Any] = Field(..., description="Chunk metadata including document info")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0-1)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "This is a sample chunk of text from a document...",
                "metadata": {
                    "document_id": 1,
                    "filename": "example.pdf",
                    "chunk_index": 3
                },
                "similarity": 0.85
            }
        }
    )

class ChatResponse(BaseModel):
    """Model for chat response"""
    answer: str = Field(..., description="Generated answer to the question")
    sources: List[str] = Field(..., description="List of source document filenames")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    retrieved_chunks: Optional[List[RetrievedChunk]] = Field(
        default=None, 
        description="Details of chunks used to generate the answer"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "Based on the uploaded documents, the main topic is...",
                "sources": ["document1.pdf", "document2.txt"],
                "confidence": 0.87,
                "retrieved_chunks": [
                    {
                        "content": "Sample chunk content...",
                        "metadata": {"document_id": 1, "filename": "document1.pdf"},
                        "similarity": 0.92
                    }
                ]
            }
        }
    )

# Database Models (for internal use)
class DocumentDB(BaseModel):
    """Internal model representing a document in the database"""
    id: Optional[int] = None
    filename: str
    file_path: str
    file_size: int
    status: DocumentStatus = DocumentStatus.PENDING
    upload_date: datetime
    metadata: Optional[Dict[str, Any]] = None

class ChunkDB(BaseModel):
    """Internal model representing a document chunk in the database"""
    id: Optional[int] = None
    document_id: int
    content: str
    chunk_index: int
    embedding: Optional[bytes] = None
    metadata: Optional[Dict[str, Any]] = None

# Statistics Models
class DocumentStats(BaseModel):
    """Model for document statistics"""
    total_documents: int = Field(..., description="Total number of documents")
    completed_documents: int = Field(..., description="Number of completed documents")
    pending_documents: int = Field(..., description="Number of pending documents")
    failed_documents: int = Field(..., description="Number of failed documents")
    total_chunks: int = Field(..., description="Total number of chunks")
    average_chunks_per_document: float = Field(..., description="Average chunks per document")
    total_storage_size: int = Field(..., description="Total storage size in bytes")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_documents": 25,
                "completed_documents": 23,
                "pending_documents": 1,
                "failed_documents": 1,
                "total_chunks": 450,
                "average_chunks_per_document": 18.0,
                "total_storage_size": 15728640
            }
        }
    )

class EmbeddingStats(BaseModel):
    """Model for embedding statistics"""
    total_embeddings: int = Field(..., description="Total number of embeddings")
    cached_embeddings: int = Field(..., description="Number of cached embeddings")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    average_embedding_time: float = Field(..., description="Average embedding generation time")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_embeddings": 1500,
                "cached_embeddings": 450,
                "cache_hit_rate": 30.0,
                "average_embedding_time": 0.25
            }
        }
    )

class ChatStats(BaseModel):
    """Model for chat statistics"""
    total_queries: int = Field(..., description="Total number of chat queries")
    average_confidence: float = Field(..., description="Average confidence score")
    active_days_last_month: int = Field(..., description="Number of active days in last month")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_queries": 150,
                "average_confidence": 0.82,
                "active_days_last_month": 15,
                "cache_hit_rate": 25.5
            }
        }
    )

class RetrievalStats(BaseModel):
    """Model for retrieval statistics"""
    total_retrievals: int = Field(..., description="Total number of retrievals")
    average_chunks_retrieved: float = Field(..., description="Average chunks retrieved per query")
    average_similarity_score: float = Field(..., description="Average similarity score")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_retrievals": 500,
                "average_chunks_retrieved": 3.2,
                "average_similarity_score": 0.75
            }
        }
    )

# Error Models
class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "Document not found",
                "detail": "No document with ID 123 exists in the system",
                "timestamp": "2024-01-15T10:30:00"
            }
        }
    )

# Health Check Model
class HealthCheck(BaseModel):
    """Model for health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    version: str = Field(default="1.0.0", description="Application version")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00",
                "version": "1.0.0"
            }
        }
    )

# Utility Functions
def row_to_document_response(row: Dict[str, Any]) -> DocumentResponse:
    """Convert database row to DocumentResponse"""
    import json
    
    # FIXED: Parse metadata if it's a string
    metadata = row.get('metadata')
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Invalid metadata for document {row.get('id')}: {metadata}")
            metadata = None  # Set to None if invalid
    
    return DocumentResponse(
        id=row['id'],
        filename=row['filename'],
        status=row['status'],
        upload_date=datetime.fromisoformat(row['upload_date']),
        file_size=row.get('file_size', 0),
        chunk_count=row.get('chunk_count', 0),
        metadata=metadata
    )