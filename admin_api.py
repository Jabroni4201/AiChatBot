# admin_api.py
# This file defines FastAPI endpoints for managing clients and their knowledge bases,
# and now includes endpoints for analytics, all interacting with PostgreSQL.

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import psycopg2 
import psycopg2.extras 
from datetime import datetime, timedelta
import json
import numpy as np 
import time 
from typing import List, Optional 

# NEW: Import our database manager
from database_manager import db_manager

# Import chatbot_core functions (though we won't use its DB_NAME anymore)
import chatbot_core # Needed for sentence_model in bulk import

# --- API Router Initialization ---
router = APIRouter()

# --- Pydantic Models for Admin API Requests/Responses ---

class ClientCreate(BaseModel):
    """Model for creating a new client."""
    client_id: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-z0-9_]+$")
    business_name: str = Field(..., min_length=2, max_length=100)
    industry: str = Field(..., min_length=2, max_length=50)

class ClientInfo(ClientCreate):
    """Model for retrieving client information."""
    created_at: datetime # CRITICAL FIX: Changed from str to datetime
    strict_rag_mode: bool # Include strict_rag_mode in ClientInfo

class KnowledgeEntryCreate(BaseModel):
    """Model for adding a new Q&A pair to a client's knowledge base."""
    question: str = Field(..., min_length=5)
    answer: str = Field(..., min_length=5)

class KnowledgeEntryDisplay(KnowledgeEntryCreate):
    """Model for displaying a Q&A pair with its ID."""
    kb_id: int # The PostgreSQL SERIAL PRIMARY KEY, mapped to kb_id

# NEW: Pydantic models for analytics responses
class AnalyticsSummary(BaseModel):
    total_queries: int
    rag_hit_rate: float
    avg_response_time: float
    avg_follow_up_questions: float
    inappropriate_triggers: int

class DailyTrendData(BaseModel):
    date: str
    total_queries: int
    rag_hits: int
    avg_response_time: float

# NEW: Pydantic models for Bulk Import
class BulkFAQItem(BaseModel):
    question: str
    answer: str
    category: Optional[str] = "General"
    priority: Optional[str] = "Medium"

class BulkImportRequest(BaseModel):
    client_id: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-z0-9_]+$") # Changed from client_name
    business_name: str
    industry: str
    similarity_threshold: Optional[float] = 0.60 # Default matches chatbot_core
    strict_rag_mode: Optional[bool] = False
    custom_greeting: Optional[str] = None
    faqs: List[BulkFAQItem]

class BulkImportResponse(BaseModel):
    success: bool
    client_id: str
    message: str
    stats: dict
    errors: List[str] = []

# --- Admin API Endpoints ---

@router.post("/admin/clients", summary="Create a new client")
async def create_client(client_data: ClientCreate):
    """
    Creates a new client entry and initializes their configuration in PostgreSQL.
    """
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # Check if client_id already exists
                cursor.execute("SELECT client_id FROM client_config WHERE client_id = %s", (client_data.client_id,))
                if cursor.fetchone():
                    raise HTTPException(status_code=409, detail="Client ID already exists.")

                current_time = datetime.now() # Use datetime object for TIMESTAMP
                cursor.execute(
                    "INSERT INTO client_config (client_id, business_name, industry, created_at, strict_rag_mode) VALUES (%s, %s, %s, %s, %s)",
                    (client_data.client_id, client_data.business_name, client_data.industry, current_time, False) # Default strict_rag_mode to False
                )
                conn.commit()
                return {"message": f"Client '{client_data.client_id}' created successfully."}
    except psycopg2.errors.UniqueViolation: # Specific PostgreSQL unique constraint error
        raise HTTPException(status_code=409, detail="Client ID already exists.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create client: {str(e)}")

@router.get("/admin/clients", response_model=List[ClientInfo], summary="List all registered clients")
async def list_clients():
    """
    Retrieves a list of all registered clients from PostgreSQL.
    """
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor: # Use DictCursor
                cursor.execute("SELECT client_id, business_name, industry, created_at, strict_rag_mode FROM client_config")
                clients_raw = cursor.fetchall()
                clients = []
                for row in clients_raw:
                    client_dict = dict(row) # DictCursor already provides dict
                    # 'strict_rag_mode' from PostgreSQL is already a boolean, no conversion needed.
                    clients.append(client_dict)
                return clients
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve clients: {str(e)}")

@router.delete("/admin/clients/{client_id}", summary="Delete a client and their data")
async def delete_client(client_id: str):
    """
    Deletes a client, their conversation history, knowledge base entries, and analytics data from PostgreSQL.
    """
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # PostgreSQL handles foreign keys, so explicit deletes might not all be needed if FKs are set up
                # But for robustness, direct delete from each table.
                cursor.execute("DELETE FROM client_config WHERE client_id = %s", (client_id,))
                deleted_clients = cursor.rowcount # Check if client existed

                cursor.execute("DELETE FROM knowledge_base WHERE client_id = %s", (client_id,))
                cursor.execute("DELETE FROM chat_history WHERE client_id = %s", (client_id,)) # Corrected to chat_history
                cursor.execute("DELETE FROM analytics_events WHERE client_id = %s", (client_id,))
                
                conn.commit()
                
                if deleted_clients == 0:
                    raise HTTPException(status_code=404, detail="Client not found.")
                return {"message": f"Client '{client_id}' and all associated data deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete client: {str(e)}")

@router.get("/admin/clients/{client_id}/knowledge", response_model=List[KnowledgeEntryDisplay], summary="Get knowledge base entries for a client")
async def get_client_knowledge_base(client_id: str):
    """
    Retrieves all Q&A pairs for a specific client's knowledge base from PostgreSQL.
    """
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor: # Use DictCursor
                cursor.execute("SELECT kb_id, question, answer FROM knowledge_base WHERE client_id = %s ORDER BY kb_id ASC", (client_id,))
                kb_entries_raw = cursor.fetchall()
                # FIX: Explicitly convert DictRow objects to dict for Pydantic compatibility
                return [dict(row) for row in kb_entries_raw] # <--- CRITICAL FIX: Convert to list of dicts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve knowledge base: {str(e)}")

@router.post("/admin/clients/{client_id}/knowledge", summary="Add a Q&A entry to a client's knowledge base")
async def add_kb_entry(client_id: str, qa_pair: KnowledgeEntryCreate):
    """
    Adds a new Q&A pair to a client's knowledge base and generates its embedding in PostgreSQL.
    """
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # Check if the client exists
                cursor.execute("SELECT client_id FROM client_config WHERE client_id = %s", (client_id,))
                if not cursor.fetchone():
                    raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found.")

                # Check for duplicate question for this client
                cursor.execute("SELECT question FROM knowledge_base WHERE client_id = %s AND question = %s",
                               (client_id, qa_pair.question))
                if cursor.fetchone():
                    raise HTTPException(status_code=409, detail="Question already exists for this client.")

                new_kb_id = None
                # Generate and store embedding for the new question immediately
                if chatbot_core.sentence_model:
                    embedding = chatbot_core.sentence_model.encode(qa_pair.question, convert_to_numpy=True)
                    embedding_list = embedding.tolist() # Convert numpy array to list for PostgreSQL VECTOR type

                    cursor.execute(
                        "INSERT INTO knowledge_base (client_id, question, answer, question_embedding, created_at) VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP) RETURNING kb_id",
                        (client_id, qa_pair.question, qa_pair.answer, embedding_list)
                    )
                    new_kb_id = cursor.fetchone()[0] # Get the returned kb_id
                    conn.commit()
                    print(f"DEBUG: Embedding generated and stored for new KB ID {new_kb_id}")
                    return {"message": "Knowledge base entry added and embedding generated.", "kb_id": new_kb_id}
                else:
                    # If sentence_model not loaded, insert without embedding (will be re-embedded on startup)
                    cursor.execute(
                        "INSERT INTO knowledge_base (client_id, question, answer, created_at) VALUES (%s, %s, %s, CURRENT_TIMESTAMP) RETURNING kb_id",
                        (client_id, qa_pair.question, qa_pair.answer)
                    )
                    new_kb_id = cursor.fetchone()[0]
                    conn.commit()
                    print(f"WARNING: Sentence Transformer model not loaded. Embedding not generated for KB ID {new_kb_id}. This might lead to RAG misses for this entry until embeddings are generated.")
                    return {"message": "Knowledge base entry added, but embedding not generated (model not loaded or error).", "kb_id": new_kb_id}

    except psycopg2.errors.UniqueViolation:
        raise HTTPException(status_code=409, detail="Question already exists for this client.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add knowledge base entry: {str(e)}")

@router.put("/admin/clients/{client_id}/knowledge/{kb_id}", summary="Update a Q&A entry for a client")
async def update_kb_entry(client_id: str, kb_id: int, qa_pair: KnowledgeEntryCreate):
    """
    Updates an existing Q&A pair in a client's knowledge base in PostgreSQL.
    """
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # First, retrieve the existing question to check if it has changed
                cursor.execute("SELECT question FROM knowledge_base WHERE kb_id = %s AND client_id = %s", (kb_id, client_id))
                existing_entry = cursor.fetchone()
                if not existing_entry:
                    raise HTTPException(status_code=404, detail="Knowledge base entry not found for this client.")
                
                old_question = existing_entry[0] # Fetching by index as it's not a DictCursor here (default)
                
                # Update the entry
                cursor.execute(
                    "UPDATE knowledge_base SET question = %s, answer = %s, updated_at = CURRENT_TIMESTAMP WHERE kb_id = %s AND client_id = %s",
                    (qa_pair.question, qa_pair.answer, kb_id, client_id)
                )
                conn.commit()

                # If the question changed, regenerate the embedding
                if old_question != qa_pair.question and chatbot_core.sentence_model:
                    embedding = chatbot_core.sentence_model.encode(qa_pair.question, convert_to_numpy=True)
                    embedding_list = embedding.tolist()
                    cursor.execute("UPDATE knowledge_base SET question_embedding = %s WHERE kb_id = %s",
                                (embedding_list, kb_id))
                    conn.commit()
                    print(f"DEBUG: Embedding re-generated for updated KB ID {kb_id}")
                    return {"message": "Knowledge base entry updated and embedding re-generated."}
                else:
                    return {"message": "Knowledge base entry updated (question unchanged or model not loaded)."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update knowledge base entry: {str(e)}")

@router.delete("/admin/clients/{client_id}/knowledge/{kb_id}", summary="Delete a Q&A entry from a client's knowledge base")
async def delete_kb_entry(client_id: str, kb_id: int):
    """
    Deletes a specific Q&A pair from a client's knowledge base in PostgreSQL.
    """
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM knowledge_base WHERE kb_id = %s AND client_id = %s", (kb_id, client_id))
                conn.commit()
                if cursor.rowcount == 0:
                    raise HTTPException(status_code=404, detail="Knowledge base entry not found for this client.")
                return {"message": "Knowledge base entry deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete knowledge base entry: {str(e)}")


# --- NEW: Analytics Endpoints ---

@router.get("/admin/analytics/summary", response_model=AnalyticsSummary, summary="Get summary analytics for a client or all clients")
async def get_analytics_summary(
    client_id: str = Query(None, description="Optional client ID to filter analytics"),
    period: str = Query("all", description="Time period filter: '24h', '7d', '30d', 'all'")
):
    """
    Provides aggregated analytics metrics (RAG hit rate, avg response time, etc.)
    for a specified client or all clients over a given time period from PostgreSQL.
    """
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                # Build the WHERE clause for filtering
                where_clauses = ["event_type = %s"] # <--- FIX: Start with event_type filter
                params = ["query"] # <--- FIX: Add 'query' to params

                start_time = None
                if period == '24h':
                    start_time = datetime.now() - timedelta(hours=24)
                elif period == '7d':
                    start_time = datetime.now() - timedelta(days=7)
                elif period == '30d':
                    start_time = datetime.now() - timedelta(days=30)
                
                if client_id:
                    where_clauses.append("client_id = %s")
                    params.append(client_id)
                if start_time:
                    where_clauses.append("timestamp >= %s")
                    params.append(start_time)
                
                where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

                # Fetch all relevant analytics events
                # Use JSONB access operators '->' and '->>' for efficiency
                cursor.execute(f"""
                    SELECT event_data 
                    FROM analytics_events 
                    {where_sql}
                """, params) # Removed redundant WHERE clause here

                events = cursor.fetchall()

                total_queries = len(events)
                rag_hits = 0
                total_response_time = 0.0
                inappropriate_triggers = 0
                
                for event_row in events:
                    event_data = event_row['event_data'] # DictCursor handles JSONB directly as dict
                    
                    # Ensure debug_info and rag_data exist before accessing
                    debug_info = event_data.get('debug_info', {})
                    rag_data = debug_info.get('rag_data', {})

                    # Count RAG hits (rag_match_found AND rag_appropriate)
                    if rag_data.get('rag_match_found', False) and rag_data.get('rag_appropriate', False):
                        rag_hits += 1
                    
                    # Sum total response time
                    total_response_time += debug_info.get('total_request_time', 0.0)

                    # Count inappropriate triggers (rag_match_found AND NOT rag_appropriate)
                    if rag_data.get('rag_match_found', False) and not rag_data.get('rag_appropriate', True): # rag_appropriate defaults to True
                        inappropriate_triggers += 1

                rag_hit_rate = (rag_hits / total_queries) * 100 if total_queries > 0 else 0.0
                avg_response_time = total_response_time / total_queries if total_queries > 0 else 0.0

                return AnalyticsSummary(
                    total_queries=total_queries,
                    rag_hit_rate=round(rag_hit_rate, 2),
                    avg_response_time=round(avg_response_time, 2),
                    avg_follow_up_questions=0.0, # Placeholder, will be calculated later
                    inappropriate_triggers=inappropriate_triggers
                )
    except Exception as e:
        print(f"ERROR: Failed to get analytics summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve analytics summary: {str(e)}")

@router.get("/admin/analytics/daily_trend", response_model=List[DailyTrendData], summary="Get daily trend analytics")
async def get_analytics_daily_trend(
    client_id: str = Query(None, description="Optional client ID to filter analytics"),
    period: str = Query("7d", description="Time period filter: '7d', '30d'")
):
    """
    Provides daily aggregated trend data for analytics metrics from PostgreSQL.
    """
    try:
        if period not in ['7d', '30d']:
            raise HTTPException(status_code=400, detail="Invalid period. Must be '7d' or '30d'.")

        num_days = 7 if period == '7d' else 30
        start_time = datetime.now() - timedelta(days=num_days)

        with db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                where_clauses = ["event_type = %s"] # <--- FIX: Start with event_type filter
                params = ["query"] # <--- FIX: Add 'query' to params
                where_clauses.append("timestamp >= %s") # Timestamp is now always included in initial params
                params.append(start_time)
                
                if client_id:
                    where_clauses.append("client_id = %s")
                    params.append(client_id)
                
                where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

                # Fetch all relevant events and group by date
                query_sql = f"""
                    SELECT
                        TO_CHAR(timestamp, 'YYYY-MM-DD') AS event_date,
                        COUNT(*) AS total_queries_day,
                        SUM(CASE WHEN event_data->'debug_info'->'rag_data'->>'rag_match_found' = 'true' AND event_data->'debug_info'->'rag_data'->>'rag_appropriate' = 'true' THEN 1 ELSE 0 END) AS rag_hits_day,
                        AVG((event_data->'debug_info'->>'total_request_time')::float) AS avg_response_time_day
                    FROM analytics_events
                    {where_sql}
                    GROUP BY event_date
                    ORDER BY event_date ASC
                """
                cursor.execute(query_sql, params)
                
                daily_raw_data = cursor.fetchall()
                
                # Ensure all days in the period are present, even if no data
                all_dates_in_period = []
                for i in range(num_days + 1): # Include today
                    date = datetime.now() - timedelta(days=num_days - i)
                    all_dates_in_period.append(date.strftime('%Y-%m-%d'))
                
                # Create a dictionary for quick lookup of existing data
                existing_data_map = {entry['event_date']: entry for entry in daily_raw_data}
                
                # Populate final trend list, filling missing dates with zeros
                final_trend = []
                for date_str in all_dates_in_period:
                    if date_str in existing_data_map:
                        final_trend.append(DailyTrendData(
                            date=existing_data_map[date_str]['event_date'],
                            total_queries=existing_data_map[date_str]['total_queries_day'],
                            rag_hits=existing_data_map[date_str]['rag_hits_day'],
                            avg_response_time=round(existing_data_map[date_str]['avg_response_time_day'], 2) if existing_data_map[date_str]['avg_response_time_day'] is not None else 0.0
                        ))
                    else:
                        final_trend.append(DailyTrendData(date=date_str, total_queries=0, rag_hits=0, avg_response_time=0.0))

                return final_trend
    except Exception as e:
        print(f"ERROR: Failed to get analytics daily trend: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve analytics daily trend: {str(e)}")

# --- NEW: Bulk Import Endpoint ---
@router.post("/admin/bulk-import", response_model=BulkImportResponse)
async def bulk_import_client_data(import_data: BulkImportRequest):
    """
    Bulk import a complete client with all their FAQs into PostgreSQL.
    Handles client creation, KB population, and embedding generation.
    """
    errors = []
    stats = {
        "client_created": False,
        "faqs_imported": 0,
        "faqs_failed": 0,
        "embeddings_generated": 0,
        "total_processing_time": 0
    }
    
    start_time = time.time()
    
    try:
        # Validate input data first
        validation_errors = validate_bulk_import_data(import_data)
        if validation_errors:
            return BulkImportResponse(
                success=False,
                client_id=import_data.client_id, # Use provided client_id for error response
                message="Validation failed for bulk import",
                stats=stats,
                errors=validation_errors
            )

        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # 1. CREATE CLIENT
                try:
                    # Check if client_id already exists
                    cursor.execute("SELECT client_id FROM client_config WHERE client_id = %s", (import_data.client_id,))
                    if cursor.fetchone():
                        errors.append(f"Client ID '{import_data.client_id}' already exists.")
                        raise psycopg2.errors.UniqueViolation("Client ID already exists") # Raise specific error for handler

                    current_time = datetime.now() # Use datetime object for TIMESTAMP
                    cursor.execute(
                        "INSERT INTO client_config (client_id, business_name, industry, similarity_threshold, max_response_length, custom_greeting, created_at, strict_rag_mode) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            import_data.client_id,
                            import_data.business_name,
                            import_data.industry,
                            import_data.similarity_threshold,
                            80,  # default max_response_length if not provided in model (was 500 in old code)
                            import_data.custom_greeting,
                            current_time,
                            import_data.strict_rag_mode
                        )
                    )
                    conn.commit() # Commit client creation before FAQs to ensure client exists for FK
                    stats["client_created"] = True
                    
                except psycopg2.errors.UniqueViolation:
                    errors.append(f"Client creation failed: Client ID '{import_data.client_id}' already exists.")
                    conn.rollback() # Rollback any partial client creation if this was in a transaction
                    return BulkImportResponse(
                        success=False,
                        client_id=import_data.client_id,
                        message="Failed to create client (ID already exists)",
                        stats=stats,
                        errors=errors
                    )
                except Exception as e:
                    errors.append(f"Client creation failed: {str(e)}")
                    conn.rollback()
                    return BulkImportResponse(
                        success=False,
                        client_id=import_data.client_id,
                        message="Failed to create client",
                        stats=stats,
                        errors=errors
                    )
                
                # 2. BULK INSERT FAQs WITH EMBEDDINGS
                # We do this in a single transaction for efficiency and atomicity
                for i, faq in enumerate(import_data.faqs):
                    try:
                        # Generate embedding for this FAQ
                        if chatbot_core.sentence_model is None:
                            raise RuntimeError("Sentence Transformer model not loaded for embedding generation.")
                            
                        question_embedding = chatbot_core.sentence_model.encode(faq.question, convert_to_numpy=True)
                        embedding_list = question_embedding.tolist() # Convert numpy array to list for PostgreSQL VECTOR type
                        
                        cursor.execute(
                            "INSERT INTO knowledge_base (client_id, question, answer, question_embedding, created_at) VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)",
                            (import_data.client_id, faq.question, faq.answer, embedding_list)
                        )
                        
                        stats["faqs_imported"] += 1
                        stats["embeddings_generated"] += 1
                        
                    except Exception as e:
                        failed_faqs_msg = f"FAQ {i+1} ('{faq.question[:50]}...') failed: {str(e)}"
                        errors.append(failed_faqs_msg)
                        stats["faqs_failed"] += 1
                
                conn.commit() # Commit all FAQ insertions if client creation and all FAQs are part of one large transaction
                               # Or, if client creation is separate, commit FAQs here.
                               # For robustness, we commit client first, then all FAQs in one go.

        # 3. CALCULATE FINAL STATS
        stats["total_processing_time"] = round(time.time() - start_time, 2)
        
        # 4. DETERMINE SUCCESS MESSAGE
        if stats["faqs_imported"] > 0:
            success_message = f"Successfully imported {stats['faqs_imported']} FAQs for {import_data.business_name} (Client ID: {import_data.client_id})."
            if stats["faqs_failed"] > 0:
                success_message += f" ({stats['faqs_failed']} failed)."
            
            return BulkImportResponse(
                success=True,
                client_id=import_data.client_id,
                message=success_message,
                stats=stats,
                errors=errors
            )
        else:
            return BulkImportResponse(
                success=False,
                client_id=import_data.client_id,
                message="No FAQs were successfully imported. Check errors.",
                stats=stats,
                errors=errors
            )
            
    except Exception as e: # Catch any exceptions not handled by specific handlers
        errors.append(f"Critical error during bulk import: {str(e)}")
        # Check if client was created before rollback attempt
        if stats["client_created"]: # If client was committed, don't rollback client
             pass # FAQs might be partially committed if error occurred during their loop
        else: # If client wasn't created, rollback the entire transaction
            try:
                if 'conn' in locals() and conn: conn.rollback()
            except Exception as rb_e:
                errors.append(f"Rollback error: {rb_e}")
        return BulkImportResponse(
            success=False,
            client_id=import_data.client_id if import_data.client_id else "", # Provide client_id if available
            message="Bulk import failed with critical error.",
            stats=stats,
            errors=errors
        )

# --- NEW: Bulk Import Validation ---
def validate_bulk_import_data(import_data: BulkImportRequest) -> List[str]:
    """
    Pre-validates bulk import data before processing.
    Returns a list of validation errors.
    """
    errors = []
    
    # Client validation
    if not import_data.client_id or not re.match(r"^[a-z0-9_]+$", import_data.client_id):
        errors.append("Client ID is required and must be lowercase letters, numbers, and underscores only.")
    if len(import_data.business_name) < 2:
        errors.append("Business name must be at least 2 characters.")
    if not import_data.industry:
        errors.append("Industry is required.")
    
    # FAQ list validation
    if not import_data.faqs:
        errors.append("At least one FAQ is required for bulk import.")
    if len(import_data.faqs) > 500: # Max 500 FAQs per single bulk import request
        errors.append("Maximum 500 FAQs allowed per import request.")
    
    # Individual FAQ validation
    for i, faq in enumerate(import_data.faqs):
        if len(faq.question.strip()) < 5:
            errors.append(f"FAQ {i+1} ('{faq.question[:50]}...'): Question too short (min 5 chars).")
        if len(faq.answer.strip()) < 10:
            errors.append(f"FAQ {i+1} ('{faq.answer[:50]}...'): Answer too short (min 10 chars).")
        if len(faq.question) > 500:
            errors.append(f"FAQ {i+1} ('{faq.question[:50]}...'): Question too long (max 500 chars).")
        if len(faq.answer) > 2000:
            errors.append(f"FAQ {i+1} ('{faq.answer[:50]}...'): Answer too long (max 2000 chars).")
    
    return errors