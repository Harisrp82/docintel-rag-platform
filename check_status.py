import sqlite3
from database import get_db_connection

def check_database_status():
    """Check what's in the database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Check documents
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        print(f"📄 Documents: {doc_count}")
        
        # Check chunks
        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        print(f"📝 Chunks: {chunk_count}")
        
        # Check chunks with embeddings
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
        embedded_count = cursor.fetchone()[0]
        print(f"🔗 Chunks with embeddings: {embedded_count}")
        
        # Show some document names
        cursor.execute("SELECT id, filename FROM documents LIMIT 5")
        docs = cursor.fetchall()
        print("\n📋 Documents in database:")
        for doc_id, filename in docs:
            print(f"  - {doc_id}: {filename}")

if __name__ == "__main__":
    check_database_status()