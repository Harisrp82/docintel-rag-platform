import asyncio
import json
from services.chat_service import chat_service
from models import ChatQuery, QueryParameters

async def test_chat():
    """Test the chat functionality directly"""
    
    # Test with a simple question about fluid dynamics
    query = ChatQuery(
        question="What is fluid dynamics?",
        parameters=QueryParameters(
            similarity_threshold=0.3,  # Lower threshold
            max_chunks=5,
            include_metadata=True
        )
    )
    
    print("🔍 Testing chat with query:", query.question)
    print("📊 Parameters:", {
        "similarity_threshold": query.parameters.similarity_threshold,
        "max_chunks": query.parameters.max_chunks
    })
    
    try:
        response = await chat_service.generate_response(query)
        print("\n✅ Chat Response:")
        print("Answer:", response.answer)
        print("Sources:", len(response.sources))
        for i, source in enumerate(response.sources):
            print(f"  Source {i+1}: {source.filename} (page {source.page_number})")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_chat())