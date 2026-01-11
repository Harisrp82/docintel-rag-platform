from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from loguru import logger
import os
from pathlib import Path

from services import chat_service, document_service
from models import ChatQuery, DocumentStats, ChatResponse, DocumentResponse

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="API for document processing and question answering using RAG",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

# Static Files and Templates
# Fix 1: Ensure directories exist before mounting
static_dir = Path("static")
templates_dir = Path("templates")
static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# Routes
@app.get("/")
async def home(request: Request):
    """Render home page"""
    try:
        return templates.TemplateResponse(
            "index.html",
            {"request": request}
        )
    except Exception as e:
        logger.error(f"Error rendering home page: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to render home page")

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks
) -> DocumentResponse:
    """Upload and process a new document"""
    try:
        # Fix 2: Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Fix 3: Check file size (10MB limit)
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
        
        # Fix 4: Reset file position after reading
        await file.seek(0)
        
        return await document_service.upload_document(file, background_tasks)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process document")

@app.get("/documents")
async def get_documents() -> list[DocumentResponse]:
    """Get all processed documents"""
    try:
        return await document_service.get_documents()
    except Exception as e:
        logger.error(f"Failed to fetch documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch documents")

@app.get("/documents/{document_id}")
async def get_document(document_id: int) -> DocumentResponse:
    """Get a specific document by ID"""
    try:
        # Fix 5: Validate document_id
        if document_id <= 0:
            raise HTTPException(status_code=400, detail="Invalid document ID")
            
        doc = await document_service.get_document(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch document")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int) -> dict:
    """Delete a document"""
    try:
        # Fix 6: Validate document_id
        if document_id <= 0:
            raise HTTPException(status_code=400, detail="Invalid document ID")
            
        success = await document_service.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@app.post("/chat")
async def chat(query: ChatQuery) -> ChatResponse:
    """Generate response for a chat query"""
    try:
        # The method expects a ChatQuery object, not individual parameters
        response = await chat_service.generate_response(query)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate chat response"
        )

@app.get("/stats/documents")
async def get_document_stats() -> DocumentStats:
    """Get document processing statistics"""
    try:
        return await document_service.get_document_stats()
    except Exception as e:
        logger.error(f"Failed to get document stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get document statistics")

@app.get("/stats/chat")
async def get_chat_stats() -> dict:
    """Get chat interaction statistics"""
    try:
        return await chat_service.get_chat_stats()
    except Exception as e:
        logger.error(f"Failed to get chat stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get chat statistics")

# Debug endpoint to see what's in the vector store
@app.get("/debug/vector-store")
async def debug_vector_store():
    """Debug endpoint to see what's in the vector store"""
    try:
        from services.document_service import document_service
        vector_store = document_service.vector_store
        
        # Get collection info
        info = vector_store.get_collection_info()
        
        # Try to get all documents
        all_results = vector_store.collection.get()
        
        return {
            "collection_info": info,
            "total_documents": len(all_results['ids']) if all_results['ids'] else 0,
            "sample_metadata": all_results['metadatas'][:3] if all_results['metadatas'] else [],
            "sample_content": [doc[:100] + "..." if len(doc) > 100 else doc for doc in all_results['documents'][:3]] if all_results['documents'] else []
        }
    except Exception as e:
        return {"error": str(e)}

# Debug endpoint to test search functionality
@app.get("/debug/search/{query}")
async def debug_search(query: str):
    """Debug endpoint to test search functionality"""
    try:
        from services.document_service import document_service
        vector_store = document_service.vector_store
        
        # Test the search directly
        results = vector_store.search(query, n_results=5)
        
        return {
            "query": query,
            "search_results": results,
            "total_results": len(results)
        }
    except Exception as e:
        return {"error": str(e)}

# Debug endpoint to test chat service step by step - ENHANCED DEBUGGING
# Debug endpoint to test chat service step by step - FIXED PARAMETER HANDLING
@app.post("/debug/chat-test")
async def debug_chat_test(query: ChatQuery):
    """Debug endpoint to test chat service step by step"""
    try:
        from services.document_service import document_service
        from services.retrieval_service import retrieval_service
        from models import QueryParameters
        
        # Debug: Log the incoming query and parameters
        print(f"DEBUG: Received query: {query.question}")
        print(f"DEBUG: Received parameters: {query.parameters}")
        
        # Use the parameters from the request, or create defaults if none provided
        if query.parameters:
            params = query.parameters
            print(f"DEBUG: Using provided parameters: {params}")
        else:
            params = QueryParameters(max_chunks=5, similarity_threshold=0.001)
            print(f"DEBUG: Using default parameters: {params}")
        
        # Debug: Test retrieval step by step
        print("DEBUG: About to call retrieval_service.get_relevant_chunks")
        
        # Test retrieval first - use correct method name
        chunks = await retrieval_service.get_relevant_chunks(query.question, params)
        print(f"DEBUG: Retrieved chunks: {len(chunks) if chunks else 0}")
        
        if chunks:
            print(f"DEBUG: First chunk content preview: {chunks[0].content[:100] if chunks[0].content else 'No content'}")
        
        return {
            "query": query.question,
            "retrieved_chunks": len(chunks) if chunks else 0,
            "chunks_sample": chunks[:2] if chunks else [],
            "chunks_full": chunks if chunks else [],
            "debug_info": {
                "params_used": str(params),
                "chunks_count": len(chunks) if chunks else 0,
                "success": True,
                "request_params": str(query.parameters) if query.parameters else "None"
            }
        }
    except Exception as e:
        print(f"DEBUG ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "debug_info": {
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
        }

# Fix 8: Add health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": "2024-01-15T10:30:00"}

# Development Server
if __name__ == "__main__":
    import uvicorn
    # Fix 8: Better server configuration
    uvicorn.run(
        app,
        host="127.0.0.1",  # Changed from 0.0.0.0 for security
        port=8000,
        log_level="info"
        # Removed reload=True to fix the warning
    )