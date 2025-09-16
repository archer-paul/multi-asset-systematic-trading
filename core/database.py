"""
Database management for Trading Bot
Handles PostgreSQL connections and operations
"""

import os
import logging
import asyncio
import subprocess
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
from core.config import config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, config=None):
        if config is None:
            from core.config import config as default_config
            config = default_config
        
        self.config = config
        self.db_config = self._parse_database_url(config.database_url or os.getenv('DATABASE_URL'))
        self.redis_config = self._parse_redis_url(config.redis_url or os.getenv('REDIS_URL'))
        self.connection = None
        self.redis_client = None
        self.redis_process = None
    
    async def initialize(self):
        """Initialize database connections"""
        postgres_connected = False
        redis_connected = False
        
        try:
            postgres_connected = await asyncio.to_thread(self.connect_postgres)
            if postgres_connected:
                self.initialize_tables()
                logger.info("PostgreSQL connected and initialized")
            else:
                logger.warning("PostgreSQL not available - continuing without it")
            
            redis_connected = self.connect_redis()
            if redis_connected:
                logger.info("Redis connected")
            else:
                logger.warning("Redis not available - continuing without it")
            
            logger.info(f"Database manager initialized - PostgreSQL: {postgres_connected}, Redis: {redis_connected}")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            logger.warning("Continuing without database connections")
            return True

    async def cleanup(self):
        """Cleanup database connections"""
        try:
            self.close_connections()
            logger.info("Database cleanup completed")
        except Exception as e:
            logger.error(f"Database cleanup failed: {e}")

    def _parse_database_url(self, url: str) -> Dict[str, str]:
        """Parse PostgreSQL connection URL for local development."""
        if not url or os.getenv("K_SERVICE"):
            return {}
        try:
            from urllib.parse import urlparse
            result = urlparse(url)
            return {
                'user': result.username,
                'password': result.password,
                'host': result.hostname,
                'port': result.port or 5432,
                'database': result.path[1:]
            }
        except Exception as e:
            logger.error(f"Failed to parse database URL for local dev: {e}")
            return {}

    def _parse_redis_url(self, url: str) -> Dict[str, Any]:
        """Parse Redis connection URL"""
        if not url: return {}
        try:
            from urllib.parse import urlparse
            result = urlparse(url)
            return {
                'host': result.hostname,
                'port': result.port or 6379,
                'decode_responses': True
            }
        except Exception as e:
            logger.error(f"Failed to parse Redis URL: {e}")
            return {}

    def connect_postgres(self) -> bool:
        """Connect to PostgreSQL, handling both local and Cloud Run environments."""
        if os.getenv("K_SERVICE") and self.config.database_url and "/" in self.config.database_url:
            logger.info("Cloud Run environment detected. Using Cloud SQL Connector.")
            return self._connect_postgres_cloud_run()
        else:
            logger.info("Local environment detected. Using standard psycopg2 connection.")
            return self._connect_postgres_local()

    def _connect_postgres_cloud_run(self) -> bool:
        """Connect to Cloud SQL using the Python Connector."""
        try:
            from google.cloud.sql.connector import Connector
            import pg8000

            parts = self.config.database_url.split('/')
            instance_connection_name = parts[3]
            db_name = parts[4]
            db_user = "postgres"

            connector = Connector()

            def getconn() -> psycopg2.extensions.connection:
                conn = connector.connect(
                    instance_connection_name,
                    "psycopg2",
                    user=db_user,
                    db=db_name,
                    enable_iam_auth=True,
                    cursor_factory=RealDictCursor
                )
                return conn

            self.connection = getconn()
            self.connection.autocommit = True
            self.connection.set_client_encoding('UTF8')
            logger.info("Connected to Cloud SQL for PostgreSQL successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Cloud SQL: {e}", exc_info=True)
            return False

    def _connect_postgres_local(self) -> bool:
        """Connect to a local or standard PostgreSQL instance."""
        if not self.db_config:
            logger.warning("No database configuration found for local connection")
            return False
        try:
            self.connection = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
            self.connection.autocommit = True
            self.connection.set_client_encoding('UTF8')
            logger.info("Connected to local PostgreSQL database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to local PostgreSQL: {e}")
            return False

    def connect_redis(self) -> bool:
        """Connect to Redis cache"""
        if not self.redis_config:
            logger.warning("No Redis configuration found")
            return False
        try:
            self.redis_client = redis.Redis(**self.redis_config)
            self.redis_client.ping()
            logger.info("Connected to Redis cache")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False

    def initialize_tables(self):
        # ... (The rest of the file remains the same)
        if not self.connection:
            logger.error("No database connection")
            return False
        
        tables = {
            'trading_signals': '''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    signal_type VARCHAR(10) NOT NULL,
                    confidence FLOAT NOT NULL,
                    price FLOAT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    strategy VARCHAR(50),
                    metadata JSONB
                )
            ''',
            'market_data': '''
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    price FLOAT NOT NULL,
                    volume BIGINT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source VARCHAR(50),
                    metadata JSONB
                )
            ''',
            'news_sentiment': '''
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT,
                    sentiment_score FLOAT,
                    symbols TEXT[],
                    source VARCHAR(50),
                    published_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'bot_performance': '''
                CREATE TABLE IF NOT EXISTS bot_performance (
                    id SERIAL PRIMARY KEY,
                    cycle_id VARCHAR(50) NOT NULL,
                    total_return FLOAT,
                    trades_count INTEGER,
                    success_rate FLOAT,
                    max_drawdown FLOAT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    metadata JSONB
                )
            ''',
            'portfolio_value_history': '''
                CREATE TABLE IF NOT EXISTS portfolio_value_history (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    portfolio_value FLOAT NOT NULL,
                    cash_balance FLOAT,
                    positions_value FLOAT,
                    unrealized_pnl FLOAT,
                    metadata JSONB
                )
            '''
        }
        
        try:
            cursor = self.connection.cursor()
            for table_name, query in tables.items():
                cursor.execute(query)
                logger.info(f"Table {table_name} created/verified")
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            return False

    def close_connections(self):
        if self.connection:
            self.connection.close()
            logger.info("PostgreSQL connection closed")
        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis connection closed")

    # ... other methods like save_trading_signal, etc.
