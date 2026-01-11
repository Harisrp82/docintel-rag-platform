from typing import List, Dict, Any, Optional
from fastapi import HTTPException
from loguru import logger
import openai
import numpy as np
from datetime import datetime
import hashlib

from database import get_db_connection
from config import settings

class EmbeddingService:
    """
    Service for generating and managing text embeddings.
    
    This service handles:
    - OpenAI embedding generation
    - Embedding caching
    - Batch processing
    - Database storage
    """
    
    def __init__(self):
        """Initialize embedding service"""
        try:
            self.model = settings.EMBEDDING_MODEL
            self.batch_size = 10
            self.client = openai.OpenAI(api_key=settings.API_KEY)
            
            logger.info(f"Embedding service initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {str(e)}")
            raise

    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text"""
        return hashlib.md5(text.encode()).hexdigest()

    async def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding if available"""
        try:
            text_hash = self._get_text_hash(text)
            
            with get_db_connection() as conn:
                # Set row factory to return dictionaries
                conn.row_factory = lambda cursor, row: {
                    col[0]: row[idx] for idx, col in enumerate(cursor.description)
                }
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT embedding 
                    FROM embeddings_cache 
                    WHERE text_hash = ?
                    AND created_at > datetime('now', '-7 days')
                """, (text_hash,))
                
                result = cursor.fetchone()
                
                if result:
                    return np.frombuffer(result['embedding'], dtype=np.float32).tolist()
                return None
                
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {str(e)}")
            return None

    async def _save_to_cache(self, text: str, embedding: List[float]) -> None:
        """Save embedding to cache"""
        try:
            text_hash = self._get_text_hash(text)
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO embeddings_cache (
                        text_hash, text, embedding, created_at
                    ) VALUES (?, ?, ?, datetime('now'))
                """, (
                    text_hash,
                    text[:1000],  # Store truncated text for reference
                    np.array(embedding, dtype=np.float32).tobytes()
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {str(e)}")

    async def get_embeddings(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """Get embeddings for multiple texts with caching."""
        try:
            # Prepare lists for texts that need embedding and their indices
            embeddings = [None] * len(texts)
            texts_to_embed = []
            indices_to_embed = []

            if use_cache:
                # Check cache for each text
                for idx, text in enumerate(texts):
                    try:
                        cached = await self.get_cached_embedding(text)
                        if cached is not None:
                            embeddings[idx] = cached
                        else:
                            texts_to_embed.append(text)
                            indices_to_embed.append(idx)
                    except Exception as e:
                        logger.warning(f"Cache check failed for text {idx}: {str(e)}")
                        texts_to_embed.append(text)
                        indices_to_embed.append(idx)
            else:
                texts_to_embed = texts
                indices_to_embed = list(range(len(texts)))

            # Get new embeddings if needed
            if texts_to_embed:
                logger.info(f"Generating embeddings for {len(texts_to_embed)} texts")
                
                # Process in batches
                for i in range(0, len(texts_to_embed), self.batch_size):
                    batch = texts_to_embed[i:i + self.batch_size]
                    batch_indices = indices_to_embed[i:i + self.batch_size]
                    
                    try:
                        # Generate embeddings using OpenAI
                        response = self.client.embeddings.create(
                            input=batch,
                            model=self.model
                        )
                        
                        batch_embeddings = [item.embedding for item in response.data]
                        
                        # Store in cache and update results
                        for text, embedding, idx in zip(batch, batch_embeddings, batch_indices):
                            if use_cache:
                                try:
                                    await self._save_to_cache(text, embedding)
                                except Exception as e:
                                    logger.warning(f"Failed to cache embedding: {str(e)}")
                            embeddings[idx] = embedding
                            
                    except Exception as e:
                        logger.error(f"Failed to generate embeddings for batch: {str(e)}")
                        # Fill with None for failed embeddings
                        for idx in batch_indices:
                            embeddings[idx] = None

            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate embeddings"
            )

    async def update_document_embeddings(self, document_id: int) -> None:
        """Update embeddings for all chunks of a document"""
        try:
            with get_db_connection() as conn:
                # Set row factory to return dictionaries
                conn.row_factory = lambda cursor, row: {
                    col[0]: row[idx] for idx, col in enumerate(cursor.description)
                }
                cursor = conn.cursor()
                
                # Get document chunks
                cursor.execute("""
                    SELECT id, content 
                    FROM chunks 
                    WHERE document_id = ?
                    ORDER BY chunk_index
                """, (document_id,))
                chunks = cursor.fetchall()
                
                if not chunks:
                    logger.warning(f"No chunks found for document {document_id}")
                    return
                
                # Get embeddings for all chunks
                texts = [chunk['content'] for chunk in chunks]
                embeddings = await self.get_embeddings(texts)
                
                # Update chunks with embeddings
                for chunk, embedding in zip(chunks, embeddings):
                    if embedding is not None:  # Only update if embedding was generated
                        cursor.execute("""
                            UPDATE chunks 
                            SET embedding = ? 
                            WHERE id = ?
                        """, (
                            np.array(embedding, dtype=np.float32).tobytes(),
                            chunk['id']
                        ))
                
                conn.commit()
                logger.info(f"Updated embeddings for document {document_id}")
                
        except Exception as e:
            logger.error(f"Failed to update document embeddings: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update embeddings for document {document_id}"
            )

    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics"""
        try:
            with get_db_connection() as conn:
                # Set row factory to return dictionaries
                conn.row_factory = lambda cursor, row: {
                    col[0]: row[idx] for idx, col in enumerate(cursor.description)
                }
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_cached,
                        COUNT(DISTINCT text_hash) as unique_embeddings,
                        AVG(LENGTH(embedding)) as avg_embedding_size
                    FROM embeddings_cache
                    WHERE created_at > datetime('now', '-7 days')
                """)
                stats = cursor.fetchone()
                
                return {
                    "total_cached": stats['total_cached'],
                    "unique_embeddings": stats['unique_embeddings'],
                    "average_embedding_size": round(stats['avg_embedding_size'] or 0, 2)
                }
        except Exception as e:
            logger.error(f"Error getting embedding stats: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve embedding statistics"
            )

# Create singleton instance
embedding_service = EmbeddingService()