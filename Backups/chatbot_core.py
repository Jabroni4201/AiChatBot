# chatbot_core.py
# This file contains the core logic for loading the Llama 3.1 8B Instruct AI model,
# generating responses, and managing conversation history and client-specific knowledge base in a SQLite database.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sqlite3
from datetime import datetime
import os
import re # Import regex for basic word tokenization and cleaning
from sentence_transformers import SentenceTransformer # NEW: Import SentenceTransformer for RAG embeddings
import numpy as np # NEW: Import numpy for handling embeddings in RAG

# --- Global Variables for Model and Tokenizer ---
model = None # For Llama 3.1 LLM
tokenizer = None # For Llama 3.1 tokenizer
sentence_model = None # NEW: Global variable for Sentence Transformer model for RAG embeddings
DB_NAME = "chat_history.db"

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


# --- Database Initialization Function ---
def init_db():
    """
    Initializes the SQLite database tables for conversation history and client knowledge bases.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # Create history table (if not exists)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                user_id TEXT NOT NULL,
                client_id TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)

        # Create knowledge_base table (if not exists)
        # NEW: Added question_embedding column for semantic RAG
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                client_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                question_embedding BLOB, -- NEW: Store embeddings as BLOB
                UNIQUE(client_id, question)
            )
        """)
        
        # NEW: Ensure client_config table exists (copied from admin_api for startup init)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS client_config (
                client_id TEXT PRIMARY KEY,
                business_name TEXT NOT NULL,
                industry TEXT NOT NULL,
                similarity_threshold REAL DEFAULT 0.60,
                max_response_length INTEGER DEFAULT 80,
                custom_greeting TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        print(f"Database '{DB_NAME}' initialized successfully.")

        # --- Populate initial knowledge base data for fake clients (for testing) ---
        initial_knowledge_data = {
            "acme_tech": {
                "business_name": "Acme Tech Solutions",
                "industry": "tech_support",
                "faqs": [
                    {"question": "What are your business hours?", "answer": "Acme Tech Solutions is open Monday to Friday, 9 AM to 5 PM PST. We are closed on weekends and public holidays."},
                    {"question": "How do I reset my password for your software?", "answer": "You can reset your password by visiting our login page and clicking the 'Forgot Password' link. Follow the instructions sent to your registered email."},
                    {"question": "Do you offer remote support?", "answer": "Yes, we offer comprehensive remote support for all our software and IT services. Please contact our support line for assistance."}
                ]
            },
            "petal_boutique": {
                "business_name": "The Petal Boutique",
                "industry": "floral_shop",
                "faqs": [
                    {"question": "Do you deliver on weekends?", "answer": "The Petal Boutique delivers Monday through Saturday. We do not offer deliveries on Sundays."},
                    {"question": "What is your refund policy?", "answer": "We offer refunds for damaged flowers within 24 hours of delivery. Please provide photos of the damage when contacting us."},
                    {"question": "Can I customize a bouquet?", "answer": "Absolutely! You can customize any bouquet to your preferences. Please call us directly to discuss your specific floral needs."}
                ]
            },
            "gourmet_grub": {
                "business_name": "Gourmet Grub Bistro",
                "industry": "restaurant",
                "faqs": [
                    {"question": "Do you take reservations?", "answer": "Yes, Gourmet Grub Bistro highly recommends reservations, especially on weekends. You can book online through our website or call us during business hours."},
                    {"question": "Are you pet-friendly?", "answer": "We welcome well-behaved pets on our outdoor patio only. Please ensure they are on a leash."},
                    {"question": "What are your catering options?", "answer": "We offer full-service catering for events of all sizes, from intimate gatherings to large corporate events. Please visit our website's catering section for menu options and pricing."}
                ]
            }
        }

        for client_id, client_data in initial_knowledge_data.items():
            # NEW: Insert into client_config table if not exists
            cursor.execute("SELECT client_id FROM client_config WHERE client_id = ?", (client_id,))
            if cursor.fetchone() is None: # Only insert if client doesn't exist
                current_time = datetime.now().isoformat()
                cursor.execute(
                    "INSERT INTO client_config (client_id, business_name, industry, created_at) VALUES (?, ?, ?, ?)",
                    (client_id, client_data["business_name"], client_data["industry"], current_time)
                )
                print(f"Client '{client_id}' added to client_config.")
            else:
                print(f"Client '{client_id}' already exists in client_config. Skipping insertion.")

            # Insert into knowledge_base table if not exists for this client
            cursor.execute("SELECT COUNT(*) FROM knowledge_base WHERE client_id = ?", (client_id,))
            if cursor.fetchone()[0] == 0:
                for faq in client_data["faqs"]:
                    cursor.execute("INSERT INTO knowledge_base (client_id, question, answer, question_embedding) VALUES (?, ?, ?, ?)",
                                   (client_id, faq["question"], faq["answer"], None))
                conn.commit()
                print(f"Initial knowledge base data populated for client: {client_id}")
            else:
                print(f"Knowledge base data already exists for client: {client_id}. Skipping insertion.")
        
        # --- Add Indexes for Performance ---
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_history_user_client ON history(user_id, client_id);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_kb_client ON knowledge_base(client_id);
        """)
        conn.commit()
        print("Database indexes added/verified successfully.")
        
    except sqlite3.Error as e:
        print(f"Database error during initialization: {e}")
    finally:
        if conn:
            conn.close()

# --- Model Loading Function for Llama 3.1 8B Instruct ---
def load_llama_model():
    """
    Loads the Llama 3.1 8B Instruct model and its tokenizer into memory.
    This function also loads the Sentence Transformer model for RAG.
    """
    global model, tokenizer, sentence_model # NEW: Declare sentence_model global
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
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config
        )
        model.eval()
        print("Llama 3.1 8B Instruct model loaded successfully onto GPU.")

        # NEW: Load Sentence Transformer model here into the global variable
        print("Loading Sentence Transformer model: sentence-transformers/all-MiniLM-L6-v2...")
        # Move the sentence model to GPU if available
        sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
        print("Sentence Transformer model loaded successfully.")

        # NEW: After all models are loaded, generate/store embeddings for any missing ones
        generate_and_store_embeddings()

    except Exception as e:
        print(f"ERROR: Failed to load models: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

# NEW: Function to generate and store embeddings for all KB entries
def generate_and_store_embeddings():
    """
    Generates embeddings for knowledge base questions that don't have them yet
    and stores them in the database. This runs once on startup.
    """
    if sentence_model is None:
        print("WARNING: Sentence Transformer model not loaded. Cannot generate embeddings.")
        return

    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # Select all KB entries that have NULL embeddings
        cursor.execute("SELECT rowid, question FROM knowledge_base WHERE question_embedding IS NULL")
        entries_to_embed = cursor.fetchall()

        if not entries_to_embed:
            print("No new knowledge base questions found requiring embeddings.")
            return

        print(f"Generating embeddings for {len(entries_to_embed)} knowledge base entries...")
        for rowid, question in entries_to_embed:
            embedding = sentence_model.encode(question, convert_to_numpy=True)
            embedding_blob = embedding.tobytes() # Convert numpy array to bytes for SQLite BLOB
            cursor.execute("UPDATE knowledge_base SET question_embedding = ? WHERE rowid = ?",
                           (embedding_blob, rowid))
        conn.commit()
        print("Embeddings generated and stored for new knowledge base entries.")

    except sqlite3.Error as e:
        print(f"ERROR: Database error during embedding generation: {e}")
    except Exception as e:
        print(f"ERROR: Error generating embeddings: {e}")
    finally:
        if conn:
            conn.close()

# --- Knowledge Base Retrieval Function (IMPROVED with Semantic Search) ---
def retrieve_knowledge(client_id: str, query: str) -> str | None:
    """
    Retrieves a relevant answer from the client's knowledge base using semantic search (embeddings).
    """
    if sentence_model is None:
        print("WARNING: Sentence Transformer model not loaded. Falling back to basic keyword matching for RAG.")
        # Fallback to simple keyword matching if embedding model isn't available
        return retrieve_knowledge_basic_keyword(client_id, query)

    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Encode the query
        query_embedding = sentence_model.encode(query, convert_to_numpy=True)

        cursor.execute("SELECT question, answer, question_embedding FROM knowledge_base WHERE client_id = ?", (client_id,))
        kb_entries = cursor.fetchall()

        best_match_answer = None
        highest_similarity = -1.0 # Cosine similarity ranges from -1 to 1

        for kb_question, kb_answer, kb_embedding_blob in kb_entries:
            if kb_embedding_blob is None:
                # This should ideally not happen after generate_and_store_embeddings on startup
                print(f"WARNING: Skipping KB entry '{kb_question}' due to missing embedding. Ensure generate_and_store_embeddings ran.")
                continue

            kb_embedding = np.frombuffer(kb_embedding_blob, dtype=np.float32) # Assuming float32 embeddings

            # Calculate cosine similarity
            # Use dot product since embeddings are typically normalized to unit length
            similarity = np.dot(query_embedding, kb_embedding) # This assumes embeddings are normalized.
                                                              # Sentence-transformers usually outputs normalized embeddings.

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_answer = kb_answer
        
        # Define a threshold for semantic similarity
        if highest_similarity >= 0.5: # Tunable threshold
            print(f"DEBUG: Semantic RAG Match found for client '{client_id}' with Cosine Similarity {highest_similarity:.2f}: '{best_match_answer}'")
            return best_match_answer
        
        print(f"DEBUG: No sufficient semantic RAG match found for client '{client_id}' for query: '{query}' (highest similarity: {highest_similarity:.2f})")
        return None

    except sqlite3.Error as e:
        print(f"ERROR: Database error during knowledge retrieval: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Error during embedding-based knowledge retrieval: {e}")
        return None
    finally:
        if conn:
            conn.close()

# Fallback function for basic keyword matching (if semantic model not available or embedding failed)
def retrieve_knowledge_basic_keyword(client_id: str, query: str) -> str | None:
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        def preprocess_text_simple(text):
            words = re.findall(r'\b\w+\b', text.lower())
            return set(word for word in words if word not in STOP_WORDS)

        processed_query_words = preprocess_text_simple(query)
        
        cursor.execute("SELECT question, answer FROM knowledge_base WHERE client_id = ?", (client_id,))
        kb_entries = cursor.fetchall()

        best_match_answer = None
        highest_jaccard_similarity = 0.0
        
        for kb_question, kb_answer in kb_entries:
            if query.lower().strip() == kb_question.lower().strip():
                print(f"DEBUG: Exact RAG Match (fallback) found for client '{client_id}': '{kb_question}' -> '{kb_answer}'")
                return kb_answer

            processed_kb_question_words = preprocess_text_simple(kb_question)
            
            intersection = len(processed_query_words.intersection(processed_kb_question_words))
            union = len(processed_query_words.union(processed_kb_question_words))
            
            jaccard_similarity = intersection / union if union != 0 else 0.0
            
            if jaccard_similarity > highest_jaccard_similarity:
                highest_jaccard_similarity = jaccard_similarity
                best_match_answer = kb_answer
        
        if highest_jaccard_similarity >= 0.3:
            print(f"DEBUG: Best RAG Match (fallback) found for client '{client_id}' with Jaccard Similarity {highest_jaccard_similarity:.2f}: '{best_match_answer}'")
            return best_match_answer
        
        print(f"DEBUG: No sufficient RAG match (fallback) found for client '{client_id}' for query: '{query}'")
        return None

    except sqlite3.Error as e:
        print(f"ERROR: Database error during fallback knowledge retrieval: {e}")
        return None
    finally:
        if conn:
            conn.close()

# --- Core AI Response Generation Function for Llama 3.1 8B ---
def get_llama_response(user_id: str, client_id: str, new_message: str) -> str:
    """
    Generates a response from Llama 3.1 8B, incorporating conversation history
    and client-specific knowledge base.
    """
    if model is None or tokenizer is None:
        raise RuntimeError("AI model not loaded. Please wait for application startup to complete or check server logs for errors.")

    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # --- Input Sanitization / Profanity Check ---
        profane_words = ["fuck", "fucking", "shit", "shitting", "asshole", "bitch", "cunt", "damn", "motherfucker"]
        normalized_message = new_message.lower()
        if any(p_word in normalized_message for p_word in profane_words):
            return "I'm sorry, but I cannot respond to that kind of language. Please keep our conversation professional."

        # Retrieve last 8 messages for context (last 4 exchanges)
        cursor.execute("SELECT message FROM history WHERE user_id = ? AND client_id = ? ORDER BY timestamp DESC LIMIT 8", (user_id, client_id))
        history_rows = cursor.fetchall()
        messages = []
        
        # --- RAG: Attempt to retrieve answer from knowledge base first (using semantic search) ---
        kb_answer = retrieve_knowledge(client_id, new_message)
        
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
        if kb_answer:
            # Emphasize strict adherence to context
            system_prompt_content += f"\n\nCONTEXT FROM KNOWLEDGE BASE: {kb_answer}\n"
            system_prompt_content += "Based on this CONTEXT, provide a direct answer. **You MUST use the provided CONTEXT verbatim as your answer if it directly answers the user's question.** Do NOT paraphrase or add information not in the context. If the CONTEXT is not directly relevant to the user's question, ignore it and answer generally, concisely."
        else:
            system_prompt_content += "\n\nIf you cannot find a direct answer in your knowledge base or from your general training, politely state that you don't have that specific information and offer to help with common customer service topics. Do not make up answers."

        system_prompt_content += "\nBe concise and helpful." # Keep this general instruction at the end

        messages.append({"role": "system", "content": system_prompt_content})

        for msg_text_tuple in reversed(history_rows):
            msg = msg_text_tuple[0]
            if msg.startswith("User: "):
                messages.append({"role": "user", "content": msg.replace("User: ", "")})
            elif msg.startswith("Bot: "):
                messages.append({"role": "assistant", "content": msg.replace("Bot: ", "")})

        messages.append({"role": "user", "content": new_message})

        input_text = tokenizer.apply_chat_template(
            messages,
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

        if not response.strip() or "i don't have that information" in response.lower() or "i cannot answer that" in response.lower():
            if not kb_answer:
                response = "I'm sorry, I don't have enough information to answer that. Can I help with common customer service inquiries?"
            else:
                response = "I'm sorry, I seem to be having trouble with that request. How else can I assist you?"

        print(f"DEBUG: Final Response (after post-processing): '{response}'")

        current_time = datetime.now().isoformat()
        cursor.execute("INSERT INTO history (user_id, client_id, message, timestamp) VALUES (?, ?, ?, ?)",
                       (user_id, client_id, f"User: {new_message}", current_time))
        cursor.execute("INSERT INTO history (user_id, client_id, message, timestamp) VALUES (?, ?, ?, ?)",
                       (user_id, client_id, f"Bot: {response}", current_time))
        conn.commit()

        return response
    except sqlite3.Error as e:
        print(f"ERROR: Database error during chat interaction: {e}")
        raise RuntimeError(f"Database error: {str(e)}")
    except Exception as e:
        print(f"ERROR: Error during Llama inference: {e}")
        raise RuntimeError(f"Chatbot inference error: {str(e)}")
    finally:
        if conn:
            conn.close()

# --- Utility Functions for History Management ---
def get_conversation_history_from_db(user_id: str, client_id: str):
    """
    Retrieves all conversation history for a given user ID and client ID from the database.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT message, timestamp FROM history WHERE user_id = ? AND client_id = ? ORDER BY timestamp ASC", (user_id, client_id))
        history = [{"message": msg, "timestamp": ts} for msg, ts in cursor.fetchall()]
        return history
    except sqlite3.Error as e:
        print(f"ERROR: Database error getting history: {e}")
        raise RuntimeError(f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()

def reset_conversation_in_db(user_id: str, client_id: str):
    """
    Clears all conversation history for a specific user ID and client ID from the database.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM history WHERE user_id = ? AND client_id = ?", (user_id, client_id))
        conn.commit()
        print(f"Conversation history for user '{user_id}' and client '{client_id}' reset.")
        return True
    except sqlite3.Error as e:
        print(f"ERROR: Database error resetting history: {e}")
        raise RuntimeError(f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()
