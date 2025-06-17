# =============================================================================
# LUNA POSTGRESQL CONNECTION DEBUG & FIX
# Solves "Connection Refused" issues with proper health checking
# =============================================================================

import psycopg2
import time
import subprocess
import os

def wait_for_postgresql(connection_string: str, max_retries: int = 30, delay: int = 2):
    """
    Wait for PostgreSQL to be ready with proper health checking
    """
    print("🔍 Waiting for PostgreSQL to be ready...")
    
    for attempt in range(max_retries):
        try:
            # Try to connect
            conn = psycopg2.connect(connection_string)
            conn.close()
            print(f"✅ PostgreSQL ready after {attempt * delay} seconds")
            return True
        except psycopg2.OperationalError as e:
            print(f"⏳ Attempt {attempt + 1}/{max_retries}: PostgreSQL not ready yet...")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                print(f"❌ PostgreSQL failed to start after {max_retries * delay} seconds")
                print(f"Error: {e}")
                return False
    
    return False

def diagnose_docker_container():
    """
    Comprehensive Docker container diagnostics
    """
    print("\n🔍 Docker Container Diagnostics:")
    
    try:
        # Check container status
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        print("📋 Container Status:")
        print(result.stdout)
        
        # Check container logs
        print("\n📝 Container Logs (last 20 lines):")
        result = subprocess.run(['docker', 'logs', '--tail', '20', 'koreailocal-postgres-1'], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        
        # Check port mapping
        print("\n🔌 Port Mapping Check:")
        result = subprocess.run(['docker', 'port', 'koreailocal-postgres-1'], 
                              capture_output=True, text=True)
        print(result.stdout)
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")

def test_postgresql_connection_methods():
    """
    Test multiple connection methods to identify the issue
    """
    print("\n🧪 Testing Connection Methods:")
    
    connection_configs = [
        {
            "name": "localhost:5432",
            "conn_str": "postgresql://luna_user:luna_dev_password@localhost:5432/luna_dev"
        },
        {
            "name": "127.0.0.1:5432",
            "conn_str": "postgresql://luna_user:luna_dev_password@127.0.0.1:5432/luna_dev"
        },
        {
            "name": "host.docker.internal:5432",
            "conn_str": "postgresql://luna_user:luna_dev_password@host.docker.internal:5432/luna_dev"
        }
    ]
    
    for config in connection_configs:
        print(f"Testing {config['name']}...")
        try:
            conn = psycopg2.connect(config['conn_str'])
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            print(f"✅ {config['name']}: Connected! PostgreSQL version: {version[:50]}...")
            conn.close()
            return config['conn_str']
        except Exception as e:
            print(f"❌ {config['name']}: {e}")
    
    return None

def verify_docker_setup():
    """
    Verify Docker setup and container health
    """
    print("\n🏥 Container Health Check:")
    
    try:
        # Check if container is actually running
        result = subprocess.run(['docker', 'exec', 'koreailocal-postgres-1', 'pg_isready', 
                               '-h', 'localhost', '-p', '5432'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PostgreSQL service is ready inside container")
        else:
            print(f"❌ PostgreSQL service not ready: {result.stdout} {result.stderr}")
            
        # Test pgvector extension
        result = subprocess.run(['docker', 'exec', 'koreailocal-postgres-1', 'psql', 
                               '-U', 'luna_user', '-d', 'luna_dev', 
                               '-c', 'SELECT * FROM pg_extension WHERE extname = \'vector\';'], 
                              capture_output=True, text=True)
        
        if 'vector' in result.stdout:
            print("✅ pgvector extension loaded")
        else:
            print("❌ pgvector extension not found")
            print(f"Output: {result.stdout}")
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")

def enhanced_migration_with_diagnostics():
    """
    Enhanced migration function with comprehensive diagnostics
    """
    print("🚀 Enhanced Luna PostgreSQL Migration with Diagnostics")
    
    # Step 1: Diagnose container
    diagnose_docker_container()
    
    # Step 2: Verify Docker setup
    verify_docker_setup()
    
    # Step 3: Test connection methods
    working_connection = test_postgresql_connection_methods()
    
    if not working_connection:
        print("\n❌ No working connection found. Container may not be ready.")
        print("\n🔧 Recommended Actions:")
        print("1. Check container logs: docker logs koreailocal-postgres-1")
        print("2. Restart container: docker compose down && docker compose up -d")
        print("3. Wait longer (PostgreSQL + pgvector can take 30+ seconds)")
        return False
    
    # Step 4: Wait for PostgreSQL with health checking
    if not wait_for_postgresql(working_connection):
        print("❌ PostgreSQL failed to become ready")
        return False
    
    # Step 5: Proceed with migration using working connection
    print(f"\n✅ Using working connection: {working_connection}")
    
    try:
        # Import from your existing migration.py file
        import sys
        sys.path.append('.')
        from migration import LunaSchemaManagerCorrected, LunaDataMigratorCorrected
        
        # Create schema
        print("\n📋 Creating PostgreSQL schema...")
        schema_manager = LunaSchemaManagerCorrected(working_connection)
        schema_manager.create_postgresql_schema()
        
        # Migrate data  
        print("\n📦 Migrating data...")
        migrator = LunaDataMigratorCorrected(working_connection, "chat_history.db")
        
        print("  → Migrating chat history...")
        migrator.migrate_chat_history()
        
        print("  → Migrating knowledge base...")
        migrator.migrate_knowledge_base_with_embeddings()
        
        print("  → Migrating client config...")
        migrator.migrate_client_config()
        
        print("  → Migrating analytics events...")
        migrator.migrate_analytics_events()
        
        print("\n🎉 Migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        return False

# =============================================================================
# IMPROVED DOCKER COMPOSE CONFIGURATION
# =============================================================================

improved_docker_compose = """
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg15
    container_name: luna-postgres
    environment:
      POSTGRES_DB: luna_dev
      POSTGRES_USER: luna_user
      POSTGRES_PASSWORD: luna_dev_password
      POSTGRES_INITDB_ARGS: "--auth-host=scram-sha-256"
    ports:
      - "5432:5432"
    volumes:
      - luna_postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U luna_user -d luna_dev"]
      interval: 5s
      timeout: 5s
      retries: 12
    restart: unless-stopped

volumes:
  luna_postgres_data:
"""

init_sql_content = """
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE luna_dev TO luna_user;
GRANT ALL ON SCHEMA public TO luna_user;
GRANT ALL ON ALL TABLES IN SCHEMA public TO luna_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO luna_user;

-- Verify pgvector installation
SELECT 'pgvector extension loaded' as status, extversion FROM pg_extension WHERE extname = 'vector';
"""

# =============================================================================
# MANUAL DIAGNOSTIC COMMANDS
# =============================================================================

def print_manual_diagnostic_commands():
    """
    Print manual commands for debugging
    """
    print("\n🔧 Manual Diagnostic Commands:")
    print("=" * 50)
    
    print("\n1. Check container status:")
    print("   docker ps")
    
    print("\n2. Check container logs:")
    print("   docker logs koreailocal-postgres-1")
    
    print("\n3. Test PostgreSQL inside container:")
    print("   docker exec -it koreailocal-postgres-1 pg_isready -h localhost -p 5432")
    
    print("\n4. Connect to PostgreSQL directly:")
    print("   docker exec -it koreailocal-postgres-1 psql -U luna_user -d luna_dev")
    
    print("\n5. Check port mapping:")
    print("   docker port koreailocal-postgres-1")
    
    print("\n6. Test network connectivity from host:")
    print("   telnet localhost 5432")
    
    print("\n7. Restart with fresh data:")
    print("   docker compose down -v")
    print("   docker compose up -d")
    
    print("\n8. Check Windows firewall (if applicable):")
    print("   netstat -an | findstr 5432")

if __name__ == "__main__":
    print("🔍 Luna PostgreSQL Connection Diagnostics")
    print("=" * 50)
    
    # Run comprehensive diagnostics
    enhanced_migration_with_diagnostics()
    
    # Print manual commands for reference
    print_manual_diagnostic_commands()
