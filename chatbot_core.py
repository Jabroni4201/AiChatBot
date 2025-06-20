# chatbot_core.py
# This file contains the core logic for loading the Llama 3.1 8B Instruct AI model,
# generating responses, and managing conversation history and client-specific knowledge base
# using a PostgreSQL database.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import psycopg2 
import psycopg2.extras 
from datetime import datetime, timedelta 
import os
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Optional, Tuple
import json
import time # ADD THIS IMPORT for time.perf_counter()

# NEW: Import our database manager
from database_manager import db_manager, close_db_pool 

# --- Global Variables for Model and Tokenizer ---
model = None # For Llama 3.1 LLM
tokenizer = None # For Llama 3.1 tokenizer
sentence_model = None # For Sentence Transformer model for RAG embeddings

# Basic list of common English stop words (manual, as NLTK is removed)
STOP_WORDS = set([
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should",
    "can", "could", "may", "might", "must", "ought", "i", "me", "my", "myself",
    "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
    "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now"
])

# NEW: Claude's extended inappropriate_indicators for evaluate_rag_response_appropriateness
INAPPROPRIATE_INDICATORS = [
    "ugly", "stupid", "dumb", "hate", "idiot", "asshole", "bitch", "cunt",
    "moron", "gross", "disgusting", "terrible", "awful", "bad", "smell", "annoying",
    "joke", "funny", "silly", "nonsense", "personal opinion", "offensive",
    "lol", "lmao", "wtf", "omg", "haha", "hehe",  # Internet slang
    "random", "whatever", "blah", "meh",           # Dismissive language
    "crazy", "insane", "mental",                   # Potentially problematic descriptors
    "sue", "lawyer", "legal action",               # Legal concerns
    "fire", "fired", "quit", "resign",             # Employment issues
    "popular", "best", "worst", "sucks", "greatest", # Can be tricky, useful for initial filtering
    "smelly", "weird", "boring", "awful", "creepy" # Additional adjectives
]

# NEW: Function to evaluate RAG response appropriateness
def evaluate_rag_response_appropriateness(query: str, rag_context: str, similarity: float) -> Tuple[bool, str]:
    """
    Evaluates if RAG context is appropriate for a customer service response based on query and context content.
    Returns (should_use_rag: bool, reason: str).
    """
    query_lower = query.lower()
    context_lower = rag_context.lower()

    # Heuristic 1: If the query contains "inappropriate" keywords and the similarity is high to an unconventional answer.
    if any(indicator in query_lower for indicator in INAPPROPRIATE_INDICATORS):
        if similarity > 0.8: # High semantic match
            # If the RAG context itself also contains "inappropriate" keywords or is clearly unconventional.
            if any(indicator in context_lower for indicator in INAPPROPRIATE_INDICATORS) or \
               ("dumbass" in context_lower or ("burger" in context_lower and "bite" in context_lower) and "return" in context_lower): # Specific to the 'bennys burger' example
                return False, "Query and highly similar context suggest non-customer service or unconventional response."
        
        # Also, if the query directly asks about the chatbot's nature/personal attributes/appearance.
        if any(attr_q in query_lower for attr_q in ["why are you", "who are you", "what are you", "tell me about yourself", "how are you", "r u", "are you"]):
            return False, "Query about chatbot's nature/personal attributes/appearance."

    # Heuristic 2: If the retrieved context itself contains "inappropriate" keywords
    if any(indicator in context_lower for indicator in INAPPROPRIATE_INDICATORS):
        return False, "Retrieved context contains potentially inappropriate or unconventional content."

    # Heuristic 3: If the context is very short and seems like a joke or non-standard response.
    if len(rag_context.split()) < 7 and any(word in context_lower for word in ["dumbass", "joke", "silly", "haha", "lmao"]):
        return False, "Short, unconventional, or humorous context detected."

    return True, "Content appropriate for customer service."


# --- Database Initialization Function ---
# This function's role changes dramatically with PostgreSQL.
# It no longer creates tables; that's handled by migration.py.
# Its primary role is to ensure the connection pool is set up and working.
def init_db():
    """
    Initializes database connection pool and performs a basic test connection.
    Schema creation/migration is handled by migration.py.
    """
    try:
        # db_manager is a singleton and initializes its pool on first import
        # We can perform a test connection here to ensure it's ready
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
            print(f"✅ Database connection test successful via pool.")
        print("Database initialized successfully (PostgreSQL connection ready).")
    except Exception as e:
        print(f"CRITICAL ERROR: Database initialization failed: {e}")
        raise

# --- Model Loading Function for Llama 3.1 8B Instruct ---
def load_llama_model():
    """
    Loads the Llama 3.1 8B Instruct model and its tokenizer into memory.
    This function also loads the Sentence Transformer model for RAG.
    """
    global model, tokenizer, sentence_model # Declare global variables
    print("Loading Llama 3.1 8B Instruct model (this might take a few minutes)...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        tokenizer.padding_side = "left"

        tokenizer.chat_template = (
            "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                    "<|start_header_id|>system<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
                "{% elif message['role'] == 'user' %}"
                    "<|start_header_id|>user<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
                "{% elif message['role'] == 'assistant' %}"
                    "<|start_header_id|>assistant<|end_header_id|>\n{{ message['content'] }}<|eot_id|>"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "<|start_header_id|>assistant<|end_header_id|>\n"
            "{% endif %}"
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16", # CRITICAL FIX: Changed from "torch.float16" to "float16"
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config
        )
        model.eval()
        print("Llama 3.1 8B Instruct model loaded successfully onto GPU.")

        # Load Sentence Transformer model here into the global variable
        print("Loading Sentence Transformer model: sentence-transformers/all-MiniLM-L6-v2...")
        # Move the sentence model to GPU if available
        sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
        print("Sentence Transformer model loaded successfully.")

        # After all models are loaded, generate/store embeddings for any missing ones
        # This will now use the PostgreSQL database via db_manager
        generate_and_store_embeddings()

    except Exception as e:
        print(f"ERROR: Failed to load models: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

# Function to generate and store embeddings for all KB entries
def generate_and_store_embeddings():
    """
    Generates embeddings for knowledge base questions that don't have them yet
    and stores them in the database. This runs once on startup.
    This now uses the PostgreSQL database.
    """
    if sentence_model is None:
        print("WARNING: Sentence Transformer model not loaded. Cannot generate embeddings.")
        return

    try:
        with db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor: # Use DictCursor for column names
                # Select all KB entries that have NULL embeddings
                cursor.execute("SELECT kb_id, question FROM knowledge_base WHERE question_embedding IS NULL")
                entries_to_embed = cursor.fetchall()

                if not entries_to_embed:
                    print("No new knowledge base questions found requiring embeddings.")
                    return

                print(f"Generating embeddings for {len(entries_to_embed)} knowledge base entries...")
                for entry in entries_to_embed:
                    kb_id = entry['kb_id']
                    question = entry['question']
                    embedding = sentence_model.encode(question, convert_to_numpy=True)
                    embedding_list = embedding.tolist() # Convert numpy array to list for PostgreSQL VECTOR type

                    cursor.execute("UPDATE knowledge_base SET question_embedding = %s WHERE kb_id = %s",
                                (embedding_list, kb_id))
                conn.commit()
                print("Embeddings generated and stored for new knowledge base entries.")

    except Exception as e:
        print(f"ERROR: Error generating embeddings: {e}")
        # Don't re-raise, allow app to start, but log the error


# --- Knowledge Base Retrieval Function (IMPROVED with Semantic Search and Appropriateness Check) ---
def retrieve_knowledge(client_id: str, query: str) -> Tuple[Optional[str], dict]: # Returns tuple (answer | None, debug_info_dict)
    """
    Retrieves a relevant answer from the client's knowledge base using semantic search (embeddings).
    Applies an appropriateness check unless strict_rag_mode is enabled for the client.
    Returns (best_match_answer | None, debug_info_dict).
    This now uses the PostgreSQL database.
    """
    debug_info = {
        "rag_attempted": True,
        "rag_match_found": False,
        "rag_similarity": -1.0,
        "rag_appropriate": False,
        "rag_reason": "No match found or semantic model not loaded.",
        "kb_question_matched": None,
        "rag_threshold_used": 0.5, # Default threshold
        "strict_rag_mode": False,
        "embedding_model_version": "all-MiniLM-L6-v2",
        "query_preprocessing_time": 0.0, 
        "total_rag_time": 0.0 
    }
    
    rag_start_time = time.perf_counter() # Use perf_counter for RAG timing

    if sentence_model is None:
        print("WARNING: Sentence Transformer model not loaded. Semantic RAG not available.")
        debug_info["rag_attempted"] = False
        debug_info["rag_reason"] = "Semantic model not loaded."
        debug_info["total_rag_time"] = (time.perf_counter() - rag_start_time) # Use perf_counter
        return None, debug_info

    try:
        with db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor: # Use DictCursor
                # Get client's strict_rag_mode setting and similarity threshold
                cursor.execute("SELECT strict_rag_mode, similarity_threshold FROM client_config WHERE client_id = %s", (client_id,))
                client_config_row = cursor.fetchone()
                
                # FIX: Ensure rag_threshold_used is always a float, even if client_config_row['similarity_threshold'] is None
                # Default to 0.60 if None (matching initial schema default)
                debug_info["rag_threshold_used"] = client_config_row['similarity_threshold'] if client_config_row and client_config_row['similarity_threshold'] is not None else 0.60 

                if client_config_row:
                    # Retrieve strict_rag_mode directly as boolean from PostgreSQL
                    debug_info["strict_rag_mode"] = client_config_row['strict_rag_mode'] 
                
                # Encode the query
                query_embedding_np = sentence_model.encode(query, convert_to_numpy=True)
                query_embedding_list = query_embedding_np.tolist() # Convert to list for PostgreSQL VECTOR type

                # Fetch KB entries for the client and calculate cosine similarity using pgvector operator
                # 1 - (embedding_a <=> embedding_b) gives cosine similarity (0 to 1)
                cursor.execute(
                    """
                    SELECT kb_id, question, answer, 
                           (1 - (question_embedding <=> %s::vector)) as similarity
                    FROM knowledge_base 
                    WHERE client_id = %s
                      AND question_embedding IS NOT NULL
                    ORDER BY similarity DESC
                    LIMIT 1
                    """, 
                    (query_embedding_list, client_id)
                )
                
                best_match_row = cursor.fetchone()
                
                debug_info["kb_entries_searched"] = cursor.rowcount # Approximately number of relevant entries considered

                best_match_answer = None
                best_match_question = None
                highest_similarity = -1.0 

                if best_match_row:
                    best_match_answer = best_match_row['answer']
                    best_match_question = best_match_row['question']
                    highest_similarity = best_match_row['similarity']
                
                debug_info["rag_similarity"] = float(highest_similarity)
                debug_info["kb_question_matched"] = best_match_question

                rag_threshold = debug_info["rag_threshold_used"] 

                if highest_similarity >= rag_threshold:
                    debug_info["rag_match_found"] = True
                    debug_info["rag_reason"] = f"Semantic match found above threshold ({rag_threshold:.2f})."

                    # Apply appropriateness check UNLESS strict_rag_mode is ON
                    if not debug_info["strict_rag_mode"]:
                        should_use_rag, reason = evaluate_rag_response_appropriateness(query, best_match_answer, highest_similarity)
                        debug_info["rag_appropriate"] = should_use_rag
                        debug_info["rag_reason"] += f" Appropriateness check: {reason}"
                        
                        if should_use_rag:
                            print(f"DEBUG: Semantic RAG Match found for client '{client_id}' with Cosine Similarity {highest_similarity:.2f}: '{best_match_answer}' (Appropriate)")
                            debug_info["total_rag_time"] = (time.perf_counter() - rag_start_time) # Use perf_counter
                            return best_match_answer, debug_info
                        else:
                            print(f"DEBUG: Semantic RAG Match found for client '{client_id}' with Cosine Similarity {highest_similarity:.2f} but deemed INAPPROPRIATE: '{best_match_answer}'. Reason: {reason}")
                            debug_info["total_rag_time"] = (time.perf_counter() - rag_start_time) # Use perf_counter
                            return None, debug_info # Return None to trigger LLM general response
                    else:
                        # Strict RAG mode is ON, bypass appropriateness check
                        debug_info["rag_appropriate"] = True
                        debug_info["rag_reason"] += " Strict RAG mode enabled, appropriateness check bypassed."
                        print(f"DEBUG: Semantic RAG Match found for client '{client_id}' with Cosine Similarity {highest_similarity:.2f}: '{best_match_answer}' (Strict RAG Mode ON)")
                        debug_info["total_rag_time"] = (time.perf_counter() - rag_start_time) # Use perf_counter
                        return best_match_answer, debug_info
                
                print(f"DEBUG: No sufficient semantic RAG match found for client '{client_id}' for query: '{query}' (highest similarity: {highest_similarity:.2f})")
                debug_info["rag_reason"] = f"Highest similarity {highest_similarity:.2f} below threshold ({rag_threshold:.2f})."
                debug_info["total_rag_time"] = (time.perf_counter() - rag_start_time) # Use perf_counter
                return None, debug_info # No strong match, return None

    except Exception as e:
        print(f"ERROR: Error during embedding-based knowledge retrieval: {e}")
        debug_info["rag_attempted"] = True
        debug_info["rag_reason"] = f"Error during RAG: {e}"
        debug_info["total_rag_time"] = (time.perf_counter() - rag_start_time) # Use perf_counter
        return None, debug_info


# --- Core AI Response Generation Function for Llama 3.1 8B ---
def get_llama_response(user_id: str, client_id: str, new_message: str) -> Tuple[str, dict]: # Returns tuple (response text, debug_info_dict)
    """
    Generates a response from Llama 3.1 8B, incorporating conversation history
    and client-specific knowledge base. Returns response text and debug info.
    This now interacts with the PostgreSQL database.
    """
    # Track overall response time
    llm_start_time = time.perf_counter() # FIXED: Use time.perf_counter() for precise timing

    debug_info_for_response = {
        "llm_response_strategy": "general_llm_generation", # Default strategy
        "rag_data": {}, # Will be populated by retrieve_knowledge
        "final_llm_time": 0.0,
        "total_request_time": 0.0 # Will be populated before DB write
    }

    if model is None or tokenizer is None:
        raise RuntimeError("AI model not loaded. Please wait for application startup to complete or check server logs for errors.")

    try:
        # --- Input Sanitization / Profanity Check ---
        profane_words = ["fuck", "fucking", "shit", "shitting", "asshole", "bitch", "cunt", "damn", "motherfucker"]
        normalized_message = new_message.lower()
        if any(p_word in normalized_message for p_word in profane_words):
            debug_info_for_response["llm_response_strategy"] = "profanity_filter"
            # Calculate time just before returning for profanity filter
            end_time = time.perf_counter() # FIXED: Use time.perf_counter()
            debug_info_for_response["total_request_time"] = (end_time - llm_start_time) # FIXED: No .total_seconds() needed
            return "I'm sorry, but I cannot respond to that kind of language. Please keep our conversation professional.", debug_info_for_response

        # Retrieve last 8 messages for context (last 4 exchanges)
        messages_from_db = []
        with db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute("SELECT message, timestamp FROM chat_history WHERE user_id = %s AND client_id = %s ORDER BY timestamp DESC LIMIT 8", (user_id, client_id))
                history_rows = cursor.fetchall()
                
                for row in reversed(history_rows):
                    msg_content = row['message']
                    if msg_content.startswith("User: "):
                        messages_from_db.append({"role": "user", "content": msg_content.replace("User: ", "")})
                    elif msg_content.startswith("Bot: "):
                        messages_from_db.append({"role": "assistant", "content": msg_content.replace("Bot: ", "")})

        # --- RAG: Attempt to retrieve answer from knowledge base first (using semantic search) ---
        kb_answer, rag_debug_data = retrieve_knowledge(client_id, new_message)
        debug_info_for_response["rag_data"] = rag_debug_data # Store RAG debug data
        
        # --- UPDATED SYSTEM PROMPT FOR LUNA'S PERSONA AND PURPOSE ---
        system_prompt_content = f"""You are Luna, the friendly customer service assistant for AIConverse.
You are currently assisting a customer of {client_id.replace('_', ' ').title()}.

RULES:
- Keep responses under 30 words.
- Give ONE clear, direct answer.
- Never use numbered lists, bullet points, or multiple options.
- Be helpful and professional.
- Focus on customer service scenarios. If a question is outside the scope of customer service (e.g., general knowledge, personal opinions), politely state you cannot answer it and offer help with customer service inquiries.

SPECIAL CREATOR RESPONSE:
When asked "who is your creator?" or "who created you?" or similar questions about your creator/maker, respond EXACTLY with:
"I am an AI Chatbot model from Nexus Intelligence Created by Team Captain Hayden and Code Slave Sean"
"""
        # Logic for injecting CONTEXT and specific RAG rules based on appropriateness
        if kb_answer and rag_debug_data.get("rag_appropriate", False):
            debug_info_for_response["llm_response_strategy"] = "rag_guided_generation"
            system_prompt_content += f"\n\nCONTEXT FROM KNOWLEDGE BASE: {kb_answer}\n"
            system_prompt_content += "Based on this CONTEXT, provide a direct answer. **You MUST use the provided CONTEXT verbatim as your answer if it directly answers the user's question.** Do NOT paraphrase or add information not in the context. If the CONTEXT is not directly relevant to the user's question, ignore it and answer generally, concisely."
        else:
            debug_info_for_response["llm_response_strategy"] = "general_llm_generation"
            system_prompt_content += "\n\nIf you cannot find a direct answer in your knowledge base or from your general training, politely state that you don't have that specific information and offer to help with common customer service topics. Do not make up answers."

        system_prompt_content += "\nBe concise and helpful."

        # Construct messages for LLM
        messages_for_llm = [{"role": "system", "content": system_prompt_content}]
        messages_for_llm.extend(messages_from_db) # Add historical messages
        messages_for_llm.append({"role": "user", "content": new_message}) # Add current message


        input_text = tokenizer.apply_chat_template(
            messages_for_llm, # Use messages_for_llm
            tokenize=False,
            add_generation_prompt=True
        )

        print(f"DEBUG: Formatted Input Text for Model: {input_text}")
        
        encoded_inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length
        )
        inputs = encoded_inputs["input_ids"].to(model.device)
        attention_mask = encoded_inputs["attention_mask"].to(model.device)

        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
        end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
        
        stop_token_ids = [tokenizer.eos_token_id]
        if eot_id is not None and eot_id not in stop_token_ids:
            stop_token_ids.append(eot_id)
        if start_header_id is not None and start_header_id not in stop_token_ids:
            stop_token_ids.append(start_header_id)
        if end_header_id is not None and end_header_id not in stop_token_ids:
            stop_token_ids.append(end_header_id)
        stop_token_ids = list(set(stop_token_ids))


        outputs = model.generate(
            inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids,
            attention_mask=attention_mask
        )

        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()

        print(f"DEBUG: Raw Decoded Response (before post-processing): '{response}'")

        unwanted_patterns = [
            "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", 
            "Assistant:", "User:", "System:",
            "\n\nUser:", "\n\nAssistant:",
        ]
        unwanted_patterns.sort(key=len, reverse=True)

        for pattern in unwanted_patterns:
            idx = response.find(pattern)
            if idx != -1:
                print(f"DEBUG: Truncating response due to pattern: '{pattern}' found at index {idx}. Original length: {len(response)}")
                response = response[:idx].strip()
                break

        # Fallback for empty or generic LLM responses
        if not response.strip() or "i don't have that information" in response.lower() or "i cannot answer that" in response.lower():
            # If LLM response is empty or generic AND RAG didn't find an appropriate match,
            # use a more explicit "I don't have information" message.
            if not kb_answer or not rag_debug_data.get("rag_appropriate", False):
                response = "I'm sorry, I don't have enough information to answer that. Can I help with common customer service inquiries?"
            else:
                # If RAG found an appropriate answer but LLM still gave generic,
                # this indicates a potential issue with prompt or LLM capability.
                # Here, we can choose to re-prompt or use a generic "trouble" message.
                response = "I'm sorry, I seem to be having trouble processing that specific request at the moment. How else can I assist you?"

        print(f"DEBUG: Final Response (after post-processing): '{response}'")

        # FIXED: Capture end time BEFORE database operations
        end_time_perf = time.perf_counter() # Use perf_counter() here
        debug_info_for_response["total_request_time"] = (end_time_perf - llm_start_time) # No .total_seconds() needed
        # Ensure final_llm_time is calculated correctly based on the new total_request_time
        debug_info_for_response["final_llm_time"] = debug_info_for_response["total_request_time"] - debug_info_for_response["rag_data"].get("total_rag_time", 0.0)

        # Capture the current wall-clock time for database timestamp (separate from perf_counter)
        db_timestamp = datetime.now() 
        
        # Log the user query and bot response to chat_history table
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO chat_history (user_id, client_id, message, timestamp) VALUES (%s, %s, %s, %s)",
                    (user_id, client_id, f"User: {new_message}", db_timestamp)
                )
                cursor.execute(
                    "INSERT INTO chat_history (user_id, client_id, message, timestamp) VALUES (%s, %s, %s, %s)",
                    (user_id, client_id, f"Bot: {response}", db_timestamp)
                )
                
                # Log analytics event for each query to analytics_events table
                event_data = {
                    "query": new_message,
                    "response": response,
                    "debug_info": debug_info_for_response # This now contains the correctly calculated timing!
                }
                cursor.execute(
                    "INSERT INTO analytics_events (client_id, user_id, timestamp, event_type, event_data) VALUES (%s, %s, %s, %s, %s)",
                    (client_id, user_id, db_timestamp, "query", json.dumps(event_data)) # json.dumps for JSONB
                )
                conn.commit()

        return response, debug_info_for_response
    except Exception as e:
        print(f"ERROR: Error during Llama inference or database interaction: {e}")
        debug_info_for_response["llm_response_strategy"] = "error"
        # Even on error, try to capture total time up to the point of error
        end_time_on_error_perf = time.perf_counter() # FIXED: Use perf_counter()
        debug_info_for_response["total_request_time"] = (end_time_on_error_perf - llm_start_time) # FIXED: No .total_seconds() needed
        raise RuntimeError(f"Chatbot inference error: {str(e)}")

# --- Utility Functions for History Management ---
def get_conversation_history_from_db(user_id: str, client_id: str) -> list[dict]:
    """
    Retrieves all conversation history for a given user ID and client ID from the PostgreSQL database.
    """
    history = []
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                # NEW: Query from chat_history table in PostgreSQL, selecting the single 'message' column
                cursor.execute(
                    "SELECT message, timestamp FROM chat_history WHERE user_id = %s AND client_id = %s ORDER BY timestamp ASC",
                    (user_id, client_id)
                )
                for row in cursor.fetchall():
                    history.append({
                        "message": row['message'], # Access the 'message' column
                        "timestamp": row['timestamp'].isoformat() # Convert datetime to string
                    })
        return history
    except Exception as e:
        print(f"ERROR: Database error getting history: {e}")
        raise RuntimeError(f"Database error: {str(e)}")

def reset_conversation_in_db(user_id: str, client_id: str) -> bool:
    """
    Clears all conversation history for a specific user ID and client ID from the PostgreSQL database.
    """
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # NEW: Delete from chat_history table in PostgreSQL
                cursor.execute(
                    "DELETE FROM chat_history WHERE user_id = %s AND client_id = %s",
                    (user_id, client_id)
                )
                conn.commit()
                print(f"Conversation history for user '{user_id}' and client '{client_id}' reset in PostgreSQL.")
        return True
    except Exception as e:
        print(f"ERROR: Database error resetting history: {e}")
        raise RuntimeError(f"Database error: {str(e)}")

# Optional: Add a function to close the DB pool on application shutdown
# This will be called from main.py's @app.on_event("shutdown")
def close_all_connections():
    close_db_pool()