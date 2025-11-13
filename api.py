import uvicorn
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File, Form

from config import Config

from rag_system_v2 import RAGSystem
from models.chat import ChatRequest, ChatResponse
from models.upload import UploadResponse

app_state = {}
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI startup and shutdown event handler.
    Initializes the RAGSystem on startup.
    """
    print("===================================")
    print(" API Server starting...")
    print("===================================")
    try:
        config = Config()
        # Store the initialized RAG system in the app_state
        app_state["rag_system"] = RAGSystem(config)
        print("RAG System initialized and ready.")
    except Exception as e:
        print(f"Failed to initialize RAGSystem: {e}")
        app_state["rag_system"] = None
    
    yield  # API is now running
    
    # Can add shutdown logic (if any)
    print("===================================")
    print(" API Server shutting down...")
    print("===================================")
    app_state["rag_system"] = None


app = FastAPI(title="RAG Pipeline API", lifespan=lifespan)

def get_rag_system() -> RAGSystem:
    """Helper function to get the RAG system from app_state."""
    rag_system = app_state.get("rag_system")
    if rag_system is None:
        raise HTTPException(
            status_code=503, 
            detail="RAGSystem is not initialized. Check server logs."
        )
    return rag_system

# --- API Endpoints ---

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to upload a single text file for ingestion.
    """
    if file.content_type != "text/plain":
        raise HTTPException(
            status_code=415, 
            detail="Unsupported file type. Please upload a .txt file."
        )
    
    try:
        # Read file content as bytes and decode
        content_bytes = await file.read()
        content = content_bytes.decode("utf-8")
        
        # Get the RAG system
        rag_system = get_rag_system()
      
        safe_filename = file.filename or "uploaded.txt"
        
        # Ingest the content
        await rag_system.add_document_from_text(
            text_content=content, 
            source_name=safe_filename
        )
        
        return UploadResponse(
            message="File ingested successfully", 
            filename=safe_filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_rag(request: ChatRequest):
    """
    Endpoint to ask a question.
    It takes a query and the previous chat history.
    It returns the answer and the updated chat history.
    """
    try:
        rag_system = get_rag_system()
        
        # Get the answer from the RAG system
        answer = await rag_system.answer_question(
            query_text=request.query, 
            chat_history=request.history
        )
        
        # Update the history
        updated_history = request.history + [(request.query, answer)]
        
        return ChatResponse(answer=answer, history=updated_history)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {e}")


if __name__ == "__main__":
    # Note: You'll need 'config.py' to be correct for this to run
    # For development, run with: uvicorn api:app --reload
    print("Starting server... Run with 'uvicorn api:app --reload'")
    uvicorn.run(app, host="0.0.0.0", port=8000)