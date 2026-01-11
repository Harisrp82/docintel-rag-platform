import sqlite3
from contextlib import contextmanager
from pathlib import Path
import os
import glob
from typing import Generator, Dict, Any
from datetime import datetime
from loguru import logger

DATABASE_PATH = "./data/documents.db"
SCHEMA_DIR = "./data/schemas"

@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for SQLite database connection."""
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        yield conn
    finally:
        conn.close()

def init_database() -> None:
    """Initialize SQLite database with required tables"""
    try:
        # Create data and schema directories
        os.makedirs("./data", exist_ok=True)
        os.makedirs(SCHEMA_DIR, exist_ok=True)
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Load and execute all SQL schema files
            schema_files = glob.glob(os.path.join(SCHEMA_DIR, "*.sql"))
            for schema_file in schema_files:
                logger.info(f"Loading schema from {schema_file}")
                with open(schema_file, 'r') as f:
                    schema_sql = f.read()
                    cursor.executescript(schema_sql)
            
            # Documents table - UPDATED with file_path column
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_path TEXT,
                    content TEXT NOT NULL,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_size INTEGER,
                    chunk_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    metadata JSON
                )
            """)
            
            # Check if file_path column exists, if not add it
            cursor.execute("PRAGMA table_info(documents)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'file_path' not in columns:
                cursor.execute("ALTER TABLE documents ADD COLUMN file_path TEXT")
                logger.info("Added file_path column to documents table")
            
            # Updated chunks table with embedding column
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER,
                    content TEXT NOT NULL,
                    chunk_index INTEGER,
                    page_number INTEGER,
                    total_pages INTEGER,
                    start_page INTEGER,
                    end_page INTEGER,
                    embedding BLOB,
                    embedding_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                )
            """)

            # Add migration for existing tables
            cursor.execute("PRAGMA table_info(chunks)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'page_number' not in columns:
                cursor.execute("ALTER TABLE chunks ADD COLUMN page_number INTEGER")
                logger.info("Added page_number column to chunks table")
            if 'total_pages' not in columns:
                cursor.execute("ALTER TABLE chunks ADD COLUMN total_pages INTEGER")
                logger.info("Added total_pages column to chunks table")
            if 'start_page' not in columns:
                cursor.execute("ALTER TABLE chunks ADD COLUMN start_page INTEGER")
                logger.info("Added start_page column to chunks table")
            if 'end_page' not in columns:
                cursor.execute("ALTER TABLE chunks ADD COLUMN end_page INTEGER")
                logger.info("Added end_page column to chunks table")
            
            # Add embeddings cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings_cache (
                    text_hash TEXT PRIMARY KEY,
                    text TEXT,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Add chat history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    confidence REAL,
                    sources TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Add chat cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_cache (
                    query_hash INTEGER,
                    context_hash TEXT,
                    response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (query_hash, context_hash)
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
                ON chunks(document_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_cache_hash 
                ON embeddings_cache(text_hash)
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")
            
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise