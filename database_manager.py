# database_manager.py
# Centralized PostgreSQL connection pooling and management for Luna AI.

import psycopg2
from psycopg2 import pool
import os
from contextlib import contextmanager
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LunaDatabaseManager:
    """
    Manages PostgreSQL connections using a connection pool for efficiency.
    Ensures connections are properly acquired and released.
    """
    
    _instance = None # Singleton instance
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LunaDatabaseManager, cls).__new__(cls)
            # Initialize connection_pool to None for the new instance before calling _initialize_connection_pool
            cls._instance.connection_pool = None # <--- ADD THIS LINE
            cls._instance._initialize_connection_pool()
        return cls._instance
        
    def _initialize_connection_pool(self):
        """
        Initializes the PostgreSQL connection pool.
        Uses environment variables for configuration.
        """
        # The check 'if self.connection_pool is not None:' is now redundant on the first call,
        # but harmless as 'connection_pool' is explicitly set to None.
        if self.connection_pool is not None:
            logging.info("Connection pool already initialized.")
            return

        # Load environment variables for DB connection
        pg_host = os.getenv('POSTGRES_HOST', 'localhost')
        pg_port = os.getenv('POSTGRES_PORT', '5432')
        pg_db = os.getenv('POSTGRES_DB', 'luna_dev')
        pg_user = os.getenv('POSTGRES_USER', 'luna_user')
        pg_password = os.getenv('POSTGRES_PASSWORD', 'luna_dev_password')
        
        # Connection pool settings
        min_conn = int(os.getenv('POSTGRES_MIN_CONNECTIONS', 1))
        max_conn = int(os.getenv('POSTGRES_MAX_CONNECTIONS', 20)) # Adjust max_conn based on expected load

        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=min_conn,
                maxconn=max_conn,
                host=pg_host,
                port=pg_port,
                database=pg_db,
                user=pg_user,
                password=pg_password
            )
            logging.info(f"✅ PostgreSQL connection pool initialized: {min_conn}-{max_conn} connections to {pg_db}@{pg_host}:{pg_port}")
            
        except Exception as e:
            logging.error(f"❌ PostgreSQL connection pool initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize database connection pool: {e}")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager to acquire a connection from the pool and ensure its release.
        Usage: with db_manager.get_connection() as conn: ...
        """
        connection = None
        try:
            connection = self.connection_pool.getconn()
            yield connection
        except Exception as e:
            if connection:
                # If an error occurs, ensure the connection is rolled back before returning to pool
                connection.rollback()
            logging.error(f"Error acquiring or using database connection: {e}")
            raise
        finally:
            if connection:
                # Ensure connection is always returned to the pool
                self.connection_pool.putconn(connection)

# Create a singleton instance of the database manager
db_manager = LunaDatabaseManager()

# Optional: Function to gracefully close the pool on application shutdown
def close_db_pool():
    if db_manager.connection_pool:
        db_manager.connection_pool.closeall()
        logging.info("PostgreSQL connection pool closed.")