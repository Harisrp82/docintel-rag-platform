from typing import List, Dict, Any, Optional
from fastapi import HTTPException
from loguru import logger
import numpy as np
from datetime import datetime

from models import (
    RetrievedChunk, 
    QueryParameters, 
    DocumentStatus
)
from database import get_db_connection
from config import settings
from .embedding_service import embedding_service

class RetrievalService:
    """
    Service for retrieving relevant document chunks.
    
    This service handles:
    - Similarity search using embeddings
    - Document chunk retrieval
    - Result ranking and filtering
    """
    
    def __init__(self):
        """Initialize retrieval service"""
        try:
            self.similarity_threshold = 0.7
            self.max_chunks = 5
            
            logger.info("Retrieval service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize retrieval service: {str(e)}")
            raise

    def _compute_similarity(
        self,
        query_embedding: List[float],
        chunk_embedding: List[float]
    ) -> float:
        """Compute cosine similarity between embeddings"""
        try:
            a = np.array(query_embedding)
            b = np.array(chunk_embedding)
            
            # Handle zero vectors
            if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
                return 0.0
                
            similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            
            # Ensure similarity is between -1 and 1
            similarity = max(-1.0, min(1.0, similarity))
            
            # Convert to 0-1 range (cosine similarity is -1 to 1)
            similarity = (similarity + 1) / 2
            
            return similarity
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {str(e)}")
            return 0.0

    async def get_relevant_chunks(
        self,
        query: str,
        params: QueryParameters
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: Search query
            params: Search parameters
            
        Returns:
            List of relevant chunks with similarity scores
        """
        try:
            logger.info(f"Retrieving chunks for query: {query[:50]}...")
            
            # Get query embedding
            query_embeddings = await embedding_service.get_embeddings([query])
            if not query_embeddings or not query_embeddings[0]:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate query embedding"
                )
            query_embedding = query_embeddings[0]
            
            logger.info(f"Generated query embedding with {len(query_embedding)} dimensions")
            
            # Get all document chunks with embeddings
            with get_db_connection() as conn:
                # Set row factory to return dictionaries - THIS IS THE KEY FIX!
                conn.row_factory = lambda cursor, row: {
                    col[0]: row[idx] for idx, col in enumerate(cursor.description)
                }
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        c.id,
                        c.content,
                        c.embedding,
                        c.chunk_index,
                        c.page_number,
                        c.total_pages,
                        c.start_page,
                        c.end_page,
                        d.id as document_id,
                        d.filename,
                        d.metadata
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE d.status = ?
                    AND c.embedding IS NOT NULL
                """, (DocumentStatus.COMPLETED.value,))
                
                chunks = cursor.fetchall()
                logger.info(f"Found {len(chunks)} chunks in database")
                
                if not chunks:
                    logger.warning("No chunks found in database")
                    return []
                
                # Compute similarities and rank chunks
                results = []
                for chunk in chunks:
                    try:
                        chunk_embedding = np.frombuffer(
                            chunk['embedding'], 
                            dtype=np.float32
                        ).tolist()
                        
                        similarity = self._compute_similarity(
                            query_embedding,
                            chunk_embedding
                        )
                        
                        logger.debug(f"Chunk {chunk['id']} similarity: {similarity:.4f}")
                        
                        # Use the more flexible threshold from params
                        if similarity >= params.similarity_threshold:
                            results.append(
                                RetrievedChunk(
                                    content=chunk['content'],
                                    metadata={
                                        "document_id": chunk['document_id'],
                                        "filename": chunk['filename'],
                                        "chunk_index": chunk['chunk_index'],
                                        "page_number": chunk['page_number'],
                                        "total_pages": chunk['total_pages'],
                                        "start_page": chunk['start_page'],
                                        "end_page": chunk['end_page']
                                    },
                                    similarity=similarity
                                )
                            )
                            #Debug: Log what we are actually retrieving 
                            logger.info(f"DEBUG RETRIEVAL: Chunk {chunk['id']} metadata: page_number={chunk['page_number']}, total_pages={chunk['total_pages']}, filename={chunk['filename']}")
                            
                    except Exception as e:
                        logger.warning(f"Failed to process chunk {chunk['id']}: {str(e)}")
                        continue
                
                logger.info(f"Found {len(results)} chunks above threshold {params.similarity_threshold}")
                
                # Sort by similarity and limit results
                results.sort(key=lambda x: x.similarity, reverse=True)
                return results[:params.max_chunks]
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve relevant chunks"
            )

    # ADDED: Convenience method that matches what you were calling
    async def retrieve_relevant_chunks(
        self,
        query: str,
        n_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[RetrievedChunk]:
        """
        Convenience method for simple chunk retrieval.
        
        Args:
            query: Search query
            n_results: Maximum number of chunks to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of relevant chunks with similarity scores
        """
        params = QueryParameters(
            max_chunks=n_results,
            similarity_threshold=similarity_threshold
        )
        return await self.get_relevant_chunks(query, params)

    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval service statistics"""
        try:
            with get_db_connection() as conn:
                # Set row factory to return dictionaries
                conn.row_factory = lambda cursor, row: {
                    col[0]: row[idx] for idx, col in enumerate(cursor.description)
                }
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_chunks,
                        COUNT(DISTINCT document_id) as total_documents,
                        AVG(LENGTH(content)) as avg_chunk_size
                    FROM chunks
                    WHERE embedding IS NOT NULL
                """)
                stats = cursor.fetchone()
                
                return {
                    "total_chunks": stats['total_chunks'],
                    "total_documents": stats['total_documents'],
                    "average_chunk_size": round(stats['avg_chunk_size'] or 0, 2)
                }
        except Exception as e:
            logger.error(f"Error getting retrieval stats: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve statistics"
            )

# Create singleton instance
retrieval_service = RetrievalService()