# main.py
# This file sets up the FastAPI web application, defines API endpoints,
# and handles cross-origin resource sharing (CORS).

from fastapi import FastAPI, HTTPException, Request # Added Request
from fastapi.middleware.cors import CORSMiddleware # Import CORS middleware for cross-origin requests
from pydantic import BaseModel # Used to define data structures for API requests and responses
from fastapi.responses import HTMLResponse # NEW: Import HTMLResponse for serving HTML files
from fastapi.staticfiles import StaticFiles # NEW: Import StaticFiles for serving static assets
import chatbot_core # Import your core chatbot logic from the chatbot_core.py file
import admin_api # FIXED: Changed from relative 'from . import admin_api' to absolute 'import admin_api'
from typing import Optional # NEW: For Optional type hints

# --- FastAPI App Initialization ---
app = FastAPI(
    title="AIConverse Luna Chatbot API",
    description="A simple API for the Luna (Llama 3.1 8B) customer service chatbot with conversation history and multi-client knowledge base.",
    version="1.0.0"
)

# NEW: Include the admin_api router in the main app
app.include_router(admin_api.router)

# NEW: Serve static files (like your HTML widget, CSS, JS files, etc.)
# You might want to create a 'static' directory in your project root for these files.
app.mount("/static", StaticFiles(directory="static"), name="static")

# NEW: Endpoint to serve the Admin Dashboard HTML
@app.get("/admin", response_class=HTMLResponse, summary="Luna Admin Dashboard")
async def get_admin_dashboard(request: Request):
    """
    Serves the HTML file for the admin dashboard.
    """
    try:
        # IMPORTANT: Ensure the 'templates' directory exists in the same path as main.py
        with open("templates/admin_dashboard.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Admin dashboard HTML not found. Make sure 'templates/admin_dashboard.html' exists.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving admin dashboard: {str(e)}")


# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (domains) to access your API.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Event for Startup ---
@app.on_event("startup")
async def startup_event():
    """
    This function runs exactly once when the FastAPI application starts up.
    It's used to perform initialization tasks like loading the AI model and setting up the database.
    """
    print("Application startup: Initializing model and database...")
    try:
        chatbot_core.init_db() # Initialize the SQLite database (creates tables if not exists and populates KB)
        chatbot_core.load_llama_model() # Load the Llama 3.1 8B Instruct model
        # IMPORTANT: This call will now also generate embeddings for any *new* KB entries
        # added via the admin panel that might not have embeddings yet.
        chatbot_core.generate_and_store_embeddings() 
        print("Startup complete: Model loaded, database ready, and embeddings generated.")
    except Exception as e:
        print(f"CRITICAL ERROR during startup: {e}")
        raise

# --- Pydantic Models for Request/Response ---
class ChatRequest(BaseModel):
    user_id: str
    client_id: str # Added client_id to the chat request
    message: str

class ChatResponse(BaseModel):
    response: str
    debug_info: Optional[dict] = None # NEW: Optional field for RAG debug info

class HistoryEntry(BaseModel):
    message: str
    timestamp: str

class HistoryResponse(BaseModel):
    history: list[HistoryEntry]

class ResetResponse(BaseModel):
    message: str

# --- FastAPI Endpoints ---
@app.post("/chat", response_model=ChatResponse, summary="Send a message to the chatbot")
async def chat_endpoint(request: ChatRequest):
    """
    Receives a user message, passes it to the Llama 3.1 8B model (via chatbot_core),
    and returns the chatbot's response.
    """
    try:
        # Pass client_id to the core chatbot function
        # NEW: get_llama_response now returns (response_text, debug_info)
        response_text, debug_info = chatbot_core.get_llama_response(request.user_id, request.client_id, request.message)
        return {"response": response_text, "debug_info": debug_info}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/history/{user_id}/{client_id}", response_model=HistoryResponse, summary="Get conversation history for a user and client")
async def get_conversation_history_endpoint(user_id: str, client_id: str): # Added client_id to path
    """
    Retrieves the full conversation history for a given user ID and client ID from the database.
    """
    try:
        history = chatbot_core.get_conversation_history_from_db(user_id, client_id)
        return {"history": history}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset/{user_id}/{client_id}", response_model=ResetResponse, summary="Reset conversation history for a user and client")
async def reset_conversation_endpoint(user_id: str, client_id: str): # Added client_id to path
    """
    Clears all conversation history for a specific user ID and client ID from the database.
    """
    try:
        success = chatbot_core.reset_conversation_in_db(user_id, client_id)
        if success:
            return {"message": "Conversation history reset successfully."}
        else:
            raise HTTPException(status_code=500, detail="Failed to reset history.")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
