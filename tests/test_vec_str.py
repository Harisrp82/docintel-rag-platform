import pytest
from pathlib import Path
from utils.vector_store import VectorStore
import numpy as np
from config import settings
from loguru import logger 

# Fixtures

@pytest.fixture
def vector_store():
    """Fixture to provide a VectorStore instance"""
    return VectorStore()

@pytest.fixture
def test_embedding():
    """Fixture to provide a test embedding vector"""
    return np.random.rand(1536).tolist()  # Standard embedding dimension

@pytest.fixture
def sample_document():
    """Fixture to provide a sample document for testing"""
    return {
        "id": "test_doc_1",
        "content": "This is a test document",
        "metadata": {"source": "test", "page": 1}
    }

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test"""
    yield  # Run the test
    try:
        store = VectorStore()
        store.collection.delete(
            ids=store.collection.get()['ids']  # Delete all existing document IDs
        )
        logger.info("Test cleanup completed successfully")
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")

# Tests
def test_vector_store_initialization(vector_store):
    """Test VectorStore initialization"""
    assert vector_store.client is not None
    assert vector_store.collection is not None
    assert vector_store.collection.name == "documents"

def test_vector_store_path(vector_store):
    """Test if vector store path is correctly set and created"""
    vector_db_path = Path(settings.VECTOR_DB_PATH).resolve()
    assert vector_db_path.exists()
    assert vector_db_path.is_dir()

def test_collection_metadata(vector_store):
    """Test if collection has correct metadata"""
    metadata = vector_store.collection.metadata
    assert metadata is not None
    assert metadata.get("hnsw:space") == "cosine"

# Document operation tests

def test_add_and_query_document(vector_store, test_embedding, sample_document):
    """Test adding and querying documents"""
    # Add document
    vector_store.collection.add(
        embeddings=[test_embedding],
        documents=[sample_document["content"]],
        metadatas=[sample_document["metadata"]],
        ids=[sample_document["id"]]
    )
    
    # Query document
    results = vector_store.collection.get(
        ids=[sample_document["id"]]
    )
    
    assert len(results['ids']) == 1
    assert results['documents'][0] == sample_document["content"]
    assert results['metadatas'][0] == sample_document["metadata"]

def test_similarity_search(vector_store, test_embedding, sample_document):
    """Test similarity search functionality"""
    # Add document
    vector_store.collection.add(
        embeddings=[test_embedding],
        documents=[sample_document["content"]],
        metadatas=[sample_document["metadata"]],
        ids=[sample_document["id"]]
    )
    
    # Perform similarity search
    results = vector_store.collection.query(
        query_embeddings=[test_embedding],
        n_results=1
    )
    
    assert len(results['ids'][0]) == 1
    assert results['ids'][0][0] == sample_document["id"]

# Error handling tests

def test_invalid_document_id(vector_store):
    """Test querying with invalid document ID"""
    results = vector_store.collection.get(
        ids=["non_existent_id"]
    )
    # ChromaDB returns empty results for non-existent IDs
    assert len(results['ids']) == 0
    assert len(results['documents']) == 0
    assert len(results['metadatas']) == 0

def test_empty_embedding(vector_store, sample_document):
    """Test adding document with empty embedding"""
    with pytest.raises(ValueError):
        vector_store.collection.add(
            embeddings=[],
            documents=[sample_document["content"]],
            metadatas=[sample_document["metadata"]],
            ids=[sample_document["id"]]
        )
#Semantic Opeartion Tests 

def test_semantic_similarity(vector_store, test_embedding):
    """Test semantic similarity ranking"""
    docs = [
        {"id": "doc1", "content": "AI and machine learning basics"},
        {"id": "doc2", "content": "Weather forecast for tomorrow"}
    ]
    
    # Add documents with different semantic content
    for doc in docs:
        vector_store.collection.add(
            embeddings=[test_embedding],
            documents=[doc["content"]],
            metadatas=[{"source": "test"}],
            ids=[doc["id"]]
        )
    
    # Query should return semantically similar documents first
    query_results = vector_store.collection.query(
        query_embeddings=[test_embedding],
        n_results=2
    )
    
    assert len(query_results['distances'][0]) == 2
    # First result should have smaller distance (more similar)
    assert query_results['distances'][0][0] <= query_results['distances'][0][1]
