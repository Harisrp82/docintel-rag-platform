import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path
from loguru import logger
from config import settings
from typing import List, Dict, Any, Optional
import hashlib

class VectorStore:
    def __init__(self):
        """Initialize vector store with ChromaDB"""
        try:
            # Create and resolve directory path
            vector_db_path = Path(settings.VECTOR_DB_PATH).resolve()
            vector_db_path.mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(vector_db_path)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Vector store initialized at {vector_db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> List[str]:
        """Add text chunks to the vector store"""
        try:
            if not ids:
                # Generate IDs based on text content and metadata
                ids = [self._generate_id(text, meta) for text, meta in zip(texts, metadatas)]
            
            # Add documents to collection
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(texts)} text chunks to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add texts to vector store: {str(e)}")
            raise

    def search(self, query: str, n_results: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_dict
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else 0.0,
                        'id': results['ids'][0][i] if results['ids'] and results['ids'][0] else None
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search vector store: {str(e)}")
            raise

    def delete(self, document_id: int) -> bool:
        """Delete documents by document ID"""
        try:
            # Fix: Use proper ChromaDB query syntax
            all_results = self.collection.get()
            
            # Filter results by document_id
            ids_to_delete = []
            if all_results['metadatas']:
                for i, metadata in enumerate(all_results['metadatas']):
                    if metadata and metadata.get('document_id') == document_id:
                        ids_to_delete.append(all_results['ids'][i])
            
            if ids_to_delete:
                # Delete all chunks for this document
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for document {document_id}")
                return True
            else:
                logger.warning(f"No chunks found for document {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete from vector store: {str(e)}")
            # Return True to avoid breaking the delete flow
            return True

    def _generate_id(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generate a unique ID for a text chunk"""
        # Create a hash from text content and metadata
        content_hash = hashlib.md5(f"{text}{str(metadata)}".encode()).hexdigest()
        return f"chunk_{content_hash[:8]}"

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection.name
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return {"error": str(e)}