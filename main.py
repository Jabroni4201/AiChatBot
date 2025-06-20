# main.py
# This file sets up the FastAPI web application, defines API endpoints,
# and handles cross-origin resource sharing (CORS).

from fastapi import FastAPI, HTTPException, Request 
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel 
from fastapi.responses import HTMLResponse 
from starlette.responses import Response # For favicon.ico response type
# Removed: from fastapi.staticfiles import StaticFiles # THIS LINE IS REMOVED

import chatbot_core 
import admin_api 
from typing import Optional 

# NEW: For loading environment variables from .env file
from dotenv import load_dotenv

# Load environment variables from .env file (for local dev)
# Ensure your .env file is in the same directory as main.py
load_dotenv() 

# --- FastAPI App Initialization ---
app = FastAPI(
    title="AIConverse Luna Chatbot API",
    description="A simple API for the Luna (Llama 3.1 8B) customer service chatbot with conversation history and multi-client knowledge base.",
    version="1.0.0"
)

# NEW: Include the admin_api router in the main app
app.include_router(admin_api.router)

# Removed: app.mount("/static", StaticFiles(directory="static"), name="static") # THIS LINE IS REMOVED

# NEW: Endpoint to serve the Admin Dashboard HTML
@app.get("/admin", response_class=HTMLResponse, summary="Luna Admin Dashboard")
async def get_admin_dashboard(request: Request):
    """
    Serves the HTML file for the admin dashboard.
    """
    try:
        # IMPORTANT: This assumes admin_dashboard.html is in a 'templates' directory
        with open("templates/admin_dashboard.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Admin dashboard HTML not found. Make sure 'templates/admin_dashboard.html' exists.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving admin dashboard: {str(e)}")

# NEW: Endpoint to serve the Chat Widget HTML (main / endpoint)
@app.get("/", response_class=HTMLResponse, summary="Luna Chat Widget (Root)")
@app.get("/widget.html", response_class=HTMLResponse, summary="Luna Chat Widget")
async def get_widget_html():
    """
    Serves the HTML file for the main chat widget.
    It can be accessed via / or /widget.html.
    """
    try:
        # Assumes widget.html is in the root directory of the project
        with open("widget.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chat widget HTML not found. Make sure 'widget.html' exists in the root.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving chat widget: {str(e)}")

# NEW: Endpoint to serve favicon.ico (optional, but good practice)
@app.get("/favicon.ico", summary="Favicon")
async def get_favicon():
    """
    Serves the favicon.ico file.
    """
    try:
        # Assumes favicon.ico is in the root directory
        with open("favicon.ico", "rb") as f:
            return Response(content=f.read(), media_type="image/x-icon")
    except FileNotFoundError:
        # Return a 404 if not found, or a default empty favicon
        raise HTTPException(status_code=404, detail="Favicon not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving favicon: {str(e)}")


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
        # Ensure PostgreSQL is running locally via Docker before this.
        # This will test the connection pool setup.
        chatbot_core.init_db() 
        chatbot_core.load_llama_model() # Load the Llama 3.1 8B Instruct model
        print("Startup complete: Model loaded, database ready, and embeddings generated.")
    except Exception as e:
        print(f"CRITICAL ERROR during startup: {e}")
        raise

# NEW: Event for Shutdown to gracefully close DB connections
@app.on_event("shutdown")
async def shutdown_event():
    """
    This function runs when the FastAPI application is shutting down.
    It's used to perform cleanup tasks like closing database connections.
    """
    print("Application shutdown: Closing database connections...")
    chatbot_core.close_all_connections()
    print("Application shutdown complete.")

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
        # FIX: Removed extra '}' from here
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
    except Exception as e: # Added missing catch block
        # FIX: Removed extra '}' from here
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") 

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
    except Exception as e: # Added missing catch block
        # FIX: Removed extra '}' from here
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def serve_widget():
    """
    Serves the widget.html file when accessing the root URL.
    """
    try:
        with open("widget.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="widget.html not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving widget: {str(e)}")