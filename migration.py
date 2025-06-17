# =============================================================================
# LUNA AI SCHEMA-ACCURATE POSTGRESQL MIGRATION
# Based on actual SQLite schema discovered: 2024
# =============================================================================

import psycopg2
import sqlite3
import numpy as np
import json
from typing import List, Tuple
import logging
import time 

class LunaSchemaManagerCorrected:
    """
    Creates PostgreSQL schema matching Hayden's actual Luna SQLite structure
    """
    
    def __init__(self, pg_connection_string: str):
        self.pg_conn_str = pg_connection_string
        
    def create_postgresql_schema(self):
        """
        Create PostgreSQL schema based on ACTUAL Luna SQLite structure
        """
        
        schema_sql = """
        -- Enable pgvector extension
        CREATE EXTENSION IF NOT EXISTS vector;
        
        -- Chat history table (matches actual SQLite structure)
        CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,  -- Added auto-increment for PostgreSQL
            user_id VARCHAR(255),   -- Matches SQLite: user_id TEXT NULL
            client_id VARCHAR(255), -- Matches SQLite: client_id TEXT NULL  
            message TEXT,           -- Matches SQLite: message TEXT NULL
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- Convert TEXT to proper TIMESTAMP
        );
        
        -- Knowledge base (matches actual SQLite + PostgreSQL enhancements)
        CREATE TABLE IF NOT EXISTS knowledge_base (
            kb_id SERIAL PRIMARY KEY,        -- Added for PostgreSQL best practices
            client_id VARCHAR(255) NOT NULL, -- Matches SQLite: client_id TEXT NULL
            question TEXT NOT NULL,          -- Matches SQLite: question TEXT NULL
            answer TEXT NOT NULL,            -- Matches SQLite: answer TEXT NULL
            question_embedding VECTOR(384),  -- Convert BLOB to VECTOR(384)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Added for tracking
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Added for tracking
        );
        
        -- Client configuration (perfect match with SQLite)
        CREATE TABLE IF NOT EXISTS client_config (
            client_id VARCHAR(255) PRIMARY KEY, -- Matches: client_id TEXT NULL
            business_name VARCHAR(255),         -- Matches: business_name TEXT NULL
            industry VARCHAR(100),              -- Matches: industry TEXT NULL  
            similarity_threshold FLOAT,         -- Matches: similarity_threshold REAL NULL
            max_response_length INTEGER,        -- Matches: max_response_length INTEGER NULL
            custom_greeting TEXT,               -- Matches: custom_greeting TEXT NULL
            created_at TIMESTAMP,               -- Convert TEXT timestamp to proper TIMESTAMP
            strict_rag_mode BOOLEAN DEFAULT FALSE -- Matches: strict_rag_mode BOOLEAN NULL
        );
        
        -- Analytics events (matches actual SQLite structure)
        CREATE TABLE IF NOT EXISTS analytics_events (
            id SERIAL PRIMARY KEY,           -- Enhanced from event_id INTEGER
            event_id INTEGER,                -- Keep original event_id for compatibility
            client_id VARCHAR(255),          -- Matches: client_id TEXT NULL
            user_id VARCHAR(255),            -- Matches: user_id TEXT NULL (your addition)
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Convert TEXT to TIMESTAMP
            event_type VARCHAR(50),          -- Matches: event_type TEXT NULL
            event_data JSONB                 -- Enhanced: JSON -> JSONB for better performance
        );
        
        -- PERFORMANCE INDEXES
        CREATE INDEX IF NOT EXISTS idx_chat_history_client_id ON chat_history(client_id);
        CREATE INDEX IF NOT EXISTS idx_chat_history_timestamp ON chat_history(timestamp);
        CREATE INDEX IF NOT EXISTS idx_chat_history_user_id ON chat_history(user_id);
        
        CREATE INDEX IF NOT EXISTS idx_knowledge_base_client_id ON knowledge_base(client_id);
        
        -- CRITICAL: Vector similarity index for semantic search
        CREATE INDEX IF NOT EXISTS idx_knowledge_embeddings_cosine 
        ON knowledge_base USING ivfflat (question_embedding vector_cosine_ops) 
        WITH (lists = 100);
        
        CREATE INDEX IF NOT EXISTS idx_analytics_client_id ON analytics_events(client_id);
        CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON analytics_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_analytics_event_type ON analytics_events(event_type);
        
        -- Row Level Security (Multi-tenant foundation)
        ALTER TABLE chat_history ENABLE ROW LEVEL SECURITY;
        ALTER TABLE knowledge_base ENABLE ROW LEVEL SECURITY;
        ALTER TABLE analytics_events ENABLE ROW LEVEL SECURITY;
        
        -- Permissive policies for now (will restrict in Phase 1.2)
        CREATE POLICY chat_history_policy ON chat_history FOR ALL USING (true);
        CREATE POLICY knowledge_base_policy ON knowledge_base FOR ALL USING (true);
        CREATE POLICY analytics_events_policy ON analytics_events FOR ALL USING (true);
        """
        
        try:
            with psycopg2.connect(self.pg_conn_str) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(schema_sql)
                    conn.commit()
                    print("✅ PostgreSQL schema created successfully (Luna-specific)")
                    
        except Exception as e:
            print(f"❌ Schema creation failed: {e}")
            raise

class LunaDataMigratorCorrected:
    """
    Migrates data from Hayden's actual Luna SQLite structure to PostgreSQL
    """
    
    def __init__(self, pg_connection_string: str, sqlite_path: str):
        self.pg_conn_str = pg_connection_string
        self.sqlite_path = sqlite_path
        
    def migrate_chat_history(self):
        """Migrate chat history with actual SQLite schema"""
        sqlite_conn = sqlite3.connect(self.sqlite_path)
        pg_conn = psycopg2.connect(self.pg_conn_str)
        
        try:
            sqlite_cursor = sqlite_conn.cursor()
            # CORRECTED: Querying from 'history' table as per your SQLite schema
            sqlite_cursor.execute("SELECT user_id, client_id, message, timestamp FROM history") 
            rows = sqlite_cursor.fetchall()
            
            pg_cursor = pg_conn.cursor()
            
            insert_sql = """
            INSERT INTO chat_history (user_id, client_id, message, timestamp)
            VALUES (%s, %s, %s, %s)
            """
            
            successful_migrations = 0
            failed_migrations = 0
            
            for row in rows:
                try:
                    # Convert timestamp if needed (SQLite TEXT -> PostgreSQL TIMESTAMP)
                    timestamp_value = row[3] if row[3] else 'now()'
                    
                    pg_cursor.execute(insert_sql, (
                        row[0],  # user_id
                        row[1],  # client_id
                        row[2],  # message
                        timestamp_value  # timestamp
                    ))
                    successful_migrations += 1
                    
                except Exception as e:
                    print(f"⚠️ Failed to migrate chat history record: {e}")
                    failed_migrations += 1
                    continue
            
            pg_conn.commit()
            print(f"✅ Migrated {successful_migrations} chat history records")
            if failed_migrations > 0:
                print(f"⚠️ {failed_migrations} chat records failed migration")
            
        except Exception as e:
            print(f"❌ Chat history migration failed: {e}")
            pg_conn.rollback()
            raise
        finally:
            sqlite_conn.close()
            pg_conn.close()
    
    def migrate_knowledge_base_with_embeddings(self):
        """
        Migrate KB with actual schema: client_id, question, answer, question_embedding
        """
        sqlite_conn = sqlite3.connect(self.sqlite_path)
        pg_conn = psycopg2.connect(self.pg_conn_str)
        
        try:
            sqlite_cursor = sqlite_conn.cursor()
            sqlite_cursor.execute("SELECT client_id, question, answer, question_embedding FROM knowledge_base")
            rows = sqlite_cursor.fetchall()
            
            pg_cursor = pg_conn.cursor()
            
            insert_sql = """
            INSERT INTO knowledge_base (client_id, question, answer, question_embedding, created_at)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            """
            
            successful_migrations = 0
            failed_migrations = 0
            
            for row in rows:
                try:
                    # Handle embedding conversion: BLOB -> VECTOR
                    if row[3]:  # question_embedding BLOB exists
                        embedding_blob = row[3]
                        embedding_array = np.frombuffer(embedding_blob, dtype=np.float32)
                        embedding_list = embedding_array.tolist()
                    else:
                        print(f"⚠️ Missing embedding for question: {row[1][:50]}...")
                        failed_migrations += 1
                        continue
                    
                    pg_cursor.execute(insert_sql, (
                        row[0],  # client_id
                        row[1],  # question
                        row[2],  # answer
                        embedding_list  # PostgreSQL VECTOR format
                    ))
                    
                    successful_migrations += 1
                    
                except Exception as e:
                    print(f"⚠️ Failed to migrate KB entry: {e}")
                    failed_migrations += 1
                    continue
            
            pg_conn.commit()
            print(f"✅ Migrated {successful_migrations} knowledge base entries")
            if failed_migrations > 0:
                print(f"⚠️ {failed_migrations} KB entries failed migration")
            
        except Exception as e:
            print(f"❌ Knowledge base migration failed: {e}")
            pg_conn.rollback()
            raise
        finally:
            sqlite_conn.close()
            pg_conn.close()
    
    def migrate_client_config(self):
        """Migrate client configuration with boolean conversion fix"""
        sqlite_conn = sqlite3.connect(self.sqlite_path)
        pg_conn = psycopg2.connect(self.pg_conn_str)
        
        try:
            sqlite_cursor = sqlite_conn.cursor()
            sqlite_cursor.execute("""
                SELECT client_id, business_name, industry, similarity_threshold, 
                       max_response_length, custom_greeting, created_at, strict_rag_mode 
                FROM client_config
            """)
            rows = sqlite_cursor.fetchall()
            
            pg_cursor = pg_conn.cursor()
            
            insert_sql = """
            INSERT INTO client_config 
            (client_id, business_name, industry, similarity_threshold, max_response_length, 
             custom_greeting, created_at, strict_rag_mode)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            successful_migrations = 0
            failed_migrations = 0
            
            for row in rows:
                try:
                    # Convert created_at timestamp if needed
                    created_at = row[6] if row[6] else 'now()'
                    
                    # FIX: Convert integer to boolean for strict_rag_mode 
                    strict_rag_mode = bool(row[7]) if row[7] is not None else False [cite: 92]
                    
                    pg_cursor.execute(insert_sql, (
                        row[0],  # client_id
                        row[1],  # business_name
                        row[2],  # industry
                        row[3],  # similarity_threshold
                        row[4],  # max_response_length
                        row[5],  # custom_greeting
                        created_at,  # created_at
                        strict_rag_mode   # strict_rag_mode (converted to boolean)
                    ))
                    
                    successful_migrations += 1
                    
                except Exception as e:
                    print(f"⚠️ Failed to migrate client config record: {e}")
                    failed_migrations += 1
                    continue
            
            pg_conn.commit()
            print(f"✅ Migrated {successful_migrations} client configurations")
            if failed_migrations > 0:
                print(f"⚠️ {failed_migrations} client config records failed migration")
            
        except Exception as e:
            print(f"❌ Client config migration failed: {e}")
            pg_conn.rollback()
            raise
        finally:
            sqlite_conn.close()
            pg_conn.close()
    
    def migrate_analytics_events(self):
        """Migrate analytics events with actual schema"""
        sqlite_conn = sqlite3.connect(self.sqlite_path)
        pg_conn = psycopg2.connect(self.pg_conn_str)
        
        try:
            sqlite_cursor = sqlite_conn.cursor()
            sqlite_cursor.execute("""
                SELECT event_id, client_id, user_id, timestamp, event_type, event_data 
                FROM analytics_events
            """)
            rows = sqlite_cursor.fetchall()
            
            pg_cursor = pg_conn.cursor()
            
            insert_sql = """
            INSERT INTO analytics_events 
            (event_id, client_id, user_id, timestamp, event_type, event_data)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            successful_migrations = 0
            failed_migrations = 0
            
            for row in rows:
                try:
                    # Convert JSON string to JSONB
                    event_data = row[5]
                    if isinstance(event_data, str):
                        try:
                            event_data = json.loads(event_data)
                        except json.JSONDecodeError:
                            event_data = {}
                    
                    timestamp_value = row[3] if row[3] else 'now()'
                    
                    pg_cursor.execute(insert_sql, (
                        row[0],  # event_id
                        row[1],  # client_id
                        row[2],  # user_id
                        timestamp_value,  # timestamp
                        row[4],  # event_type
                        json.dumps(event_data) if event_data else '{}'  # event_data as JSONB
                    ))
                    
                    successful_migrations += 1
                    
                except Exception as e:
                    print(f"⚠️ Failed to migrate analytics event: {e}")
                    failed_migrations += 1
                    continue
            
            pg_conn.commit()
            print(f"✅ Migrated {successful_migrations} analytics events")
            if failed_migrations > 0:
                print(f"⚠️ {failed_migrations} analytics events failed migration")
            
        except Exception as e:
            print(f"❌ Analytics events migration failed: {e}")
            pg_conn.rollback()
            raise
        finally:
            sqlite_conn.close()
            pg_conn.close()

# =============================================================================
# UPDATED MIGRATION EXECUTION
# =============================================================================

def execute_luna_specific_migration():
    """
    Execute migration based on Hayden's actual Luna SQLite schema
    """
    
    print("🚀 Starting Luna PostgreSQL Migration (Schema-Accurate)...")
    time.sleep(10) 

    # Configuration
    POSTGRES_CONNECTION = "postgresql://luna_user:luna_dev_password@localhost:5432/luna_dev"
    SQLITE_PATH = "chat_history.db"
    
    try:
        # Step 1: Create schema
        print("\n📋 Step 1: Creating PostgreSQL schema (Luna-specific)...")
        schema_manager = LunaSchemaManagerCorrected(POSTGRES_CONNECTION)
        schema_manager.create_postgresql_schema()
        
        # Step 2: Migrate data
        print("\n📦 Step 2: Migrating data...")
        migrator = LunaDataMigratorCorrected(POSTGRES_CONNECTION, SQLITE_PATH)
        
        # Migrate each table with correct schema
        print("  → Migrating chat_history...")
        migrator.migrate_chat_history()
        
        print("  → Migrating knowledge_base...")
        migrator.migrate_knowledge_base_with_embeddings()
        
        print("  → Migrating client_config...")
        migrator.migrate_client_config()
        
        print("  → Migrating analytics_events...")
        migrator.migrate_analytics_events()
        
        # Step 3: Verify migration
        print("\n✅ Step 3: Verifying migration...")
        with psycopg2.connect(POSTGRES_CONNECTION) as conn:
            with conn.cursor() as cursor:
                # Check record counts
                tables = ['chat_history', 'knowledge_base', 'client_config', 'analytics_events']
                
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"   {table}: {count} records")
                
                # Test vector similarity search
                cursor.execute("""
                SELECT question, 1 - (question_embedding <=> %s::vector) as similarity 
                FROM knowledge_base 
                WHERE question_embedding IS NOT NULL
                LIMIT 1
                """, ([0.1] * 384,))  # Test vector with correct 384 dimensions
                
                test_result = cursor.fetchone()
                if test_result:
                    print("   ✅ Vector similarity search working")
                else:
                    print("   ⚠️ No embeddings found for testing")
                
        print("\n🎉 Luna-specific migration completed successfully!")
        print("\n🔄 Next Steps:")
        print("   1. Update chatbot_core.py database calls")
        print("   2. Update admin_api.py database calls") 
        print("   3. Test semantic RAG with PostgreSQL")
        print("   4. Test admin dashboard functionality")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        print("Review error details and retry if needed")

if __name__ == "__main__":
    execute_luna_specific_migration()