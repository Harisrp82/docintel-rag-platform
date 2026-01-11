from typing import List, Dict, Any, Optional
from fastapi import HTTPException
from loguru import logger
from openai import AsyncOpenAI  # Updated import for new OpenAI API
from functools import wraps, lru_cache
from datetime import datetime
import json

from models import (
    ChatQuery, ChatResponse, RetrievedChunk, 
    QueryParameters, DocumentStatus
)
from database import get_db_connection
from config import settings
from .retrieval_service import retrieval_service

def rate_limit(max_requests: int = 10, window_seconds: int = 60):
    """Rate limiting decorator for API endpoints"""
    def decorator(func):
        requests = []
        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = datetime.now()
            requests[:] = [req for req in requests if (now - req).seconds < window_seconds]
            if len(requests) >= max_requests:
                raise HTTPException(status_code=429, detail="Too many requests")
            requests.append(now)
            return await func(*args, **kwargs)
        return wrapper
    return decorator

class ChatService:
    """
    Service for handling chat interactions and LLM responses.
    
    This service handles:
    - Chat query validation
    - Response generation with LLM
    - Chat history management
    - Response caching
    """
    
    def __init__(self):
        """Initialize chat service with LLM settings"""
        try:
            self.model_name = settings.MODEL_NAME
            self.system_prompt = settings.SYSTEM_PROMPT
            # Updated OpenAI client initialization for new API
            self.client = AsyncOpenAI(api_key=settings.API_KEY)
            self.max_chunks = 10
            self.min_question_length = 3
            
            logger.info("Chat service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize chat service: {str(e)}")
            raise

    def _validate_query(self, query: ChatQuery) -> None:
        """Validate chat query parameters"""
        if len(query.question.strip()) < self.min_question_length:
            raise HTTPException(
                status_code=400,
                detail=f"Question must be at least {self.min_question_length} characters long"
            )
        if query.parameters and query.parameters.max_chunks > self.max_chunks:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum chunks limit is {self.max_chunks}"
            )

    def _get_filename_from_chunk(self, chunk) -> str:
        """Safely extract filename from chunk metadata"""
        try:
            if hasattr(chunk, 'metadata'):
                if isinstance(chunk.metadata, str):
                    try:
                        metadata = json.loads(chunk.metadata)
                        return metadata.get("filename", "Unknown")
                    except (json.JSONDecodeError, TypeError):
                        return "Unknown"
                elif isinstance(chunk.metadata, dict):
                    return chunk.metadata.get("filename", "Unknown")
                else:
                    return "Unknown"
            else:
                return "Unknown"
        except Exception as e:
            logger.debug(f"Error in _get_filename_from_chunk: {e}")
            return "Unknown"

    def _parse_chunk_metadata(self, chunk) -> Dict[str, Any]:
        """Safely parse chunk metadata from string or dict"""
        try:
            if hasattr(chunk, 'metadata'):
                if isinstance(chunk.metadata, str):
                    try:
                        return json.loads(chunk.metadata)
                    except (json.JSONDecodeError, TypeError):
                        return {}
                elif isinstance(chunk.metadata, dict):
                    return chunk.metadata or {}
                else:
                    return {}
            return {}
        except Exception as e:
            logger.debug(f"Error in _parse_chunk_metadata: {e}")
            return {}

    def _generate_sources(self, chunks: List[RetrievedChunk]) -> List[str]:
        """Generate properly formatted sources based on the citation rules"""
        sources = []
        seen_sources = set()  # To deduplicate identical {document, page} pairs
        
        # Group chunks by document to determine if we have multiple documents
        doc_chunks = {}
        for chunk in chunks:
            metadata = self._parse_chunk_metadata(chunk)
            filename = metadata.get("filename", "Unknown")
            if filename not in doc_chunks:
                doc_chunks[filename] = []
            doc_chunks[filename].append(chunk)
        
        multiple_docs = len(doc_chunks) > 1
        
        for chunk in chunks:
            metadata = self._parse_chunk_metadata(chunk)
            filename = metadata.get("filename", "Unknown")
            page_number = metadata.get("page_number")
            total_pages = metadata.get("total_pages") or 1  # Default to 1 if None
            start_page = metadata.get("start_page")
            end_page = metadata.get("end_page")
            
            # Clean filename for display (remove extension and clean up)
            display_name = filename
            if filename.endswith('.pdf'):
                display_name = filename[:-4]  # Remove .pdf extension
            
            # Apply citation rules
            if total_pages == 1:
                # Single-page PDF - no page number
                source_text = display_name
                source_key = (filename, None)
            elif page_number is not None:
                # Multi-page PDF with page number
                if start_page and end_page and start_page != end_page:
                    # Chunk spans multiple pages
                    source_text = f"{display_name} p.{start_page}–{end_page}"
                    source_key = (filename, f"{start_page}-{end_page}")
                else:
                    # Single page
                    source_text = f"{display_name} p.{page_number}"
                    source_key = (filename, page_number)
            elif total_pages and total_pages > 1:  # Added null check here
                # Multi-page PDF but missing page info
                source_text = f"{display_name} (page n/a)"
                source_key = (filename, "n/a")
            else:
                # Fallback
                source_text = display_name
                source_key = (filename, None)
            
            # Deduplicate identical {document, page} pairs
            if source_key not in seen_sources:
                sources.append(source_text)
                seen_sources.add(source_key)
        
        return sources
    
    def _prepare_messages(
        self,
        query: str,
        chunks: List[RetrievedChunk]
    ) -> List[Dict[str, str]]:
        """Prepare messages for the LLM"""
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        context = "\n\n".join([
            f"Document: {self._get_filename_from_chunk(chunk)}\n{chunk.content}"
            for chunk in chunks
        ])
        
        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        })
        
        return messages

    @lru_cache(maxsize=100)
    async def get_cached_response(
        self,
        query: str,
        context_hash: str
    ) -> Optional[ChatResponse]:
        """Get cached response if available"""
        try:
            with get_db_connection() as conn:
                # Add row factory for dictionary access
                conn.row_factory = lambda cursor, row: {
                    col[0]: row[idx] for idx, col in enumerate(cursor.description)
                }
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT response, created_at 
                    FROM chat_cache 
                    WHERE query_hash = ? AND context_hash = ?
                    AND created_at > datetime('now', '-24 hours')
                """, (hash(query), context_hash))
                result = cursor.fetchone()
                
                if result:
                    return ChatResponse.parse_raw(result['response'])
                return None
                
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {str(e)}")
            return None

    async def _save_to_cache(
        self,
        query: str,
        context_hash: str,
        response: ChatResponse
    ) -> None:
        """Save response to cache"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO chat_cache (
                        query_hash, context_hash, response, created_at
                    ) VALUES (?, ?, ?, datetime('now'))
                """, (hash(query), context_hash, response.json()))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to cache response: {str(e)}")

    @rate_limit()
    async def generate_response(
        self,
        query: ChatQuery
    ) -> ChatResponse:
        """Generate a response using RAG and LLM"""
        log_context = {"query": query.question[:100]}
        
        try:
            self._validate_query(query)
            logger.info("Processing chat query", **log_context)
            
            # Get relevant chunks using retrieval service
            chunks = await retrieval_service.get_relevant_chunks(
                query.question,
                query.parameters or QueryParameters(similarity_threshold=0.5)
            )
            
            if not chunks:
                raise HTTPException(
                    status_code=404,
                    detail="No relevant context found for the question"
                )
            
            context_hash = hash(tuple(chunk.content for chunk in chunks))
            cached = await self.get_cached_response(query.question, str(context_hash))
            if cached:
                logger.info("Cache hit", **log_context)
                return cached
            
            messages = self._prepare_messages(query.question, chunks)
            
            # Updated OpenAI API call for new version
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS
            )
            
            answer = response.choices[0].message.content
            confidence = sum(chunk.similarity for chunk in chunks) / len(chunks)
            
            # Create sources with proper formatting
            sources = self._generate_sources(chunks)
            
            chat_response = ChatResponse(
                answer=answer,
                sources=sources,
                confidence=confidence,
                retrieved_chunks=chunks if query.parameters and query.parameters.include_metadata else None
            )
            
            await self._save_to_cache(query.question, str(context_hash), chat_response)
            await self._save_to_history(query, chat_response)
            
            logger.info(
                "Generated response",
                confidence=confidence,
                sources_count=len(chunks),
                **log_context
            )
            
            return chat_response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                f"Failed to generate response: {str(e)}",
                error=str(e),
                error_type=type(e).__name__,
                **log_context,
                exc_info=True
            )
            # Print to console for immediate debugging
            print(f"CHAT SERVICE ERROR: {str(e)}")
            print(f"ERROR TYPE: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate response: {str(e)}"
            )

    async def _save_to_history(
        self,
        query: ChatQuery,
        response: ChatResponse
    ) -> None:
        """Save chat interaction to history"""
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO chat_history (
                        question, answer, confidence, sources, created_at
                    ) VALUES (?, ?, ?, ?, datetime('now'))
                """, (
                    query.question,
                    response.answer,
                    response.confidence,
                    json.dumps([str(s) for s in response.sources])
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to save to history: {str(e)}")

    async def get_chat_stats(self) -> Dict[str, Any]:
        """Get chat service statistics"""
        try:
            with get_db_connection() as conn:
                # Add row factory for dictionary access
                conn.row_factory = lambda cursor, row: {
                    col[0]: row[idx] for idx, col in enumerate(cursor.description)
                }
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_queries,
                        AVG(confidence) as avg_confidence,
                        COUNT(DISTINCT DATE(created_at)) as active_days,
                        (
                            SELECT COUNT(*) 
                            FROM chat_cache 
                            WHERE created_at > datetime('now', '-24 hours')
                        ) as cache_hits
                    FROM chat_history
                    WHERE created_at > datetime('now', '-30 days')
                """)
                stats = cursor.fetchone()
                
                return {
                    "total_queries": stats['total_queries'],
                    "average_confidence": round(stats['avg_confidence'] or 0.0, 2),
                    "active_days_last_month": stats['active_days'],
                    "cache_hit_rate": round(
                        (stats['cache_hits'] / stats['total_queries'] * 100)
                        if stats['total_queries'] > 0 else 0.0,
                        2
                    )
                }
        except Exception as e:
            logger.error(f"Error getting chat stats: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve chat statistics"
            )

# Create singleton instance
chat_service = ChatService()