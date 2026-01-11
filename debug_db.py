import asyncio
from services.embedding_service import embedding_service
from database import get_db_connection
import numpy as np

async def fix_missing_embeddings():
    """Generate embeddings for chunks that don't have them"""
    with get_db_connection() as conn:
        conn.row_factory = lambda cursor, row: {
            col[0]: row[idx] for idx, col in enumerate(cursor.description)
        }
        cursor = conn.cursor()
        
        # Get chunks without embeddings
        cursor.execute("""
            SELECT id, content, document_id
            FROM chunks 
            WHERE embedding IS NULL
            ORDER BY document_id, chunk_index
        """)
        chunks = cursor.fetchall()
        
        if not chunks:
            print("All chunks already have embeddings!")
            return
            
        print(f"Found {len(chunks)} chunks without embeddings. Generating...")
        
        # Get embeddings for all chunks
        texts = [chunk['content'] for chunk in chunks]
        embeddings = await embedding_service.get_embeddings(texts)
        
        # Update chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            if embedding is not None:
                cursor.execute("""
                    UPDATE chunks 
                    SET embedding = ? 
                    WHERE id = ?
                """, (
                    np.array(embedding, dtype=np.float32).tobytes(),
                    chunk['id']
                ))
                print(f"Added embedding for chunk {chunk['id']}")
        
        conn.commit()
        print(f"Successfully generated embeddings for {len(chunks)} chunks!")

if __name__ == "__main__":
    asyncio.run(fix_missing_embeddings())