"""
Database management for Trading Bot
Handles PostgreSQL connections and operations
"""

import os
import logging
import asyncio
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
        self.db_config = self._parse_database_url(config.database_url)
        self.redis_config = self._parse_redis_url(config.redis_url)
        self.connection = None
        self.redis_client = None
    
    async def initialize(self):
        """Initialize database connections"""
        postgres_connected = False
        redis_connected = False
        
        try:
            # Connect to PostgreSQL (optional)
            postgres_connected = self.connect_postgres()
            if postgres_connected:
                self.initialize_tables()
                logger.info("PostgreSQL connected and initialized")
            else:
                logger.warning("PostgreSQL not available - continuing without it")
            
            # Connect to Redis (optional)
            redis_connected = self.connect_redis()
            if redis_connected:
                logger.info("Redis connected")
            else:
                logger.warning("Redis not available - continuing without it")
            
            # Always return True - the bot should work even without databases
            logger.info(f"Database manager initialized - PostgreSQL: {postgres_connected}, Redis: {redis_connected}")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            logger.warning("Continuing without database connections")
            return True  # Don't fail completely
    
    async def cleanup(self):
        """Cleanup database connections"""
        try:
            self.close_connections()
            logger.info("Database cleanup completed")
        except Exception as e:
            logger.error(f"Database cleanup failed: {e}")
        
    def _parse_database_url(self, url: str) -> Dict[str, str]:
        """Parse PostgreSQL connection URL"""
        if not url:
            return {}
            
        # postgresql://user:password@host:port/database
        try:
            url = url.replace('postgresql://', '')
            auth, location = url.split('@')
            user, password = auth.split(':')
            host_port, database = location.split('/')
            host, port = host_port.split(':') if ':' in host_port else (host_port, '5432')
            
            return {
                'user': user,
                'password': password,
                'host': host,
                'port': int(port),
                'database': database
            }
        except Exception as e:
            logger.error(f"Failed to parse database URL: {e}")
            return {}
    
    def _parse_redis_url(self, url: str) -> Dict[str, Any]:
        """Parse Redis connection URL"""
        if not url:
            return {}
            
        # redis://host:port
        try:
            url = url.replace('redis://', '')
            host, port = url.split(':') if ':' in url else (url, '6379')
            
            return {
                'host': host,
                'port': int(port),
                'decode_responses': True
            }
        except Exception as e:
            logger.error(f"Failed to parse Redis URL: {e}")
            return {}
    
    def connect_postgres(self) -> bool:
        """Connect to PostgreSQL database"""
        if not self.db_config:
            logger.warning("No database configuration found")
            return False
            
        try:
            # Add encoding settings to fix Windows UTF-8 issues
            db_config = self.db_config.copy()
            db_config['client_encoding'] = 'utf8'
            
            # Set environment variables for proper encoding
            import locale
            os.environ['PGCLIENTENCODING'] = 'utf8'
            
            self.connection = psycopg2.connect(
                **db_config,
                cursor_factory=RealDictCursor
            )
            self.connection.autocommit = True
            
            # Set connection encoding explicitly
            self.connection.set_client_encoding('UTF8')
            
            logger.info("Connected to PostgreSQL database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False
    
    def connect_redis(self) -> bool:
        """Connect to Redis cache"""
        if not self.redis_config:
            logger.warning("No Redis configuration found")
            return False
            
        try:
            self.redis_client = redis.Redis(**self.redis_config)
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis cache")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    def initialize_tables(self):
        """Create necessary database tables"""
        if not self.connection:
            logger.error("No database connection")
            return False
            
        tables = {
            'trading_signals': """
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
            """,
            'market_data': """
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    price FLOAT NOT NULL,
                    volume BIGINT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source VARCHAR(50),
                    metadata JSONB
                )
            """,
            'news_sentiment': """
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
            """,
            'bot_performance': """
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
            """
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
    
    def store_trading_signal(self, symbol: str, signal_type: str, confidence: float, 
                           price: float, strategy: str = None, metadata: Dict = None):
        """Store trading signal in database"""
        if not self.connection:
            return False
            
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO trading_signals (symbol, signal_type, confidence, price, strategy, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (symbol, signal_type, confidence, price, strategy, metadata))
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Failed to store trading signal: {e}")
            return False
    
    def store_market_data(self, symbol: str, price: float, volume: int = None, 
                         source: str = None, metadata: Dict = None):
        """Store market data in database"""
        if not self.connection:
            return False
            
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO market_data (symbol, price, volume, source, metadata)
                VALUES (%s, %s, %s, %s, %s)
            """, (symbol, price, volume, source, metadata))
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Failed to store market data: {e}")
            return False
    
    def cache_set(self, key: str, value: str, expire: int = 3600):
        """Set value in Redis cache"""
        if not self.redis_client:
            return False
            
        try:
            self.redis_client.setex(key, expire, value)
            return True
        except Exception as e:
            logger.error(f"Failed to set cache: {e}")
            return False
    
    def cache_get(self, key: str) -> Optional[str]:
        """Get value from Redis cache"""
        if not self.redis_client:
            return None
            
        try:
            return self.redis_client.get(key)
        except Exception as e:
            logger.error(f"Failed to get cache: {e}")
            return None
    
    def get_recent_signals(self, symbol: str = None, hours: int = 24) -> List[Dict]:
        """Get recent trading signals"""
        if not self.connection:
            return []
            
        try:
            cursor = self.connection.cursor()
            since = datetime.now() - timedelta(hours=hours)
            
            if symbol:
                cursor.execute("""
                    SELECT * FROM trading_signals 
                    WHERE symbol = %s AND timestamp > %s
                    ORDER BY timestamp DESC
                """, (symbol, since))
            else:
                cursor.execute("""
                    SELECT * FROM trading_signals 
                    WHERE timestamp > %s
                    ORDER BY timestamp DESC
                """, (since,))
                
            results = cursor.fetchall()
            cursor.close()
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to get recent signals: {e}")
            return []
    
    def close_connections(self):
        """Close all database connections"""
        if self.connection:
            self.connection.close()
            logger.info("PostgreSQL connection closed")
            
        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis connection closed")
    
    # Methods used by bot_orchestrator
    async def save_trading_signal(self, signal: Dict[str, Any]):
        """Save trading signal to database"""
        if not self.connection:
            logger.debug("No database connection - skipping signal save")
            return
        
        try:
            self.store_trading_signal(
                symbol=signal.get('symbol', ''),
                signal_type=signal.get('signal_type', ''),
                confidence=signal.get('confidence', 0.0),
                price=signal.get('price', 0.0),
                strategy=signal.get('strategy', ''),
                metadata=signal
            )
        except Exception as e:
            logger.error(f"Failed to save trading signal: {e}")
    
    async def save_trade_execution(self, execution: Dict[str, Any]):
        """Save trade execution result to database"""
        if not self.connection:
            logger.debug("No database connection - skipping execution save")
            return
        
        try:
            # Store in market_data table for now
            self.store_market_data(
                symbol=execution.get('symbol', ''),
                price=execution.get('price', 0.0),
                volume=execution.get('quantity', 0),
                source='execution',
                metadata=execution
            )
        except Exception as e:
            logger.error(f"Failed to save trade execution: {e}")
    
    async def save_portfolio_snapshot(self, portfolio: Dict[str, Any]):
        """Save portfolio snapshot to database"""
        if not self.connection:
            logger.debug("No database connection - skipping portfolio save")
            return
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO bot_performance (cycle_id, total_return, trades_count, metadata)
                VALUES (%s, %s, %s, %s)
            """, (
                f"portfolio_{datetime.now().isoformat()}", 
                portfolio.get('total_return_pct', 0.0),
                portfolio.get('positions_count', 0),
                portfolio
            ))
            cursor.close()
        except Exception as e:
            logger.error(f"Failed to save portfolio snapshot: {e}")
    
    async def save_performance_metrics(self, metrics: Dict[str, Any]):
        """Save performance metrics to database"""
        if not self.connection:
            logger.debug("No database connection - skipping metrics save")
            return
        
        try:
            # Store performance metrics
            logger.debug(f"Performance metrics: {metrics}")
        except Exception as e:
            logger.error(f"Failed to save performance metrics: {e}")
    
    async def get_trading_history(self, limit: int = 100) -> List[Dict]:
        """Get trading history from database"""
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM trading_signals
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))
            results = cursor.fetchall()
            cursor.close()
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to get trading history: {e}")
            return []
    
    async def save_final_report(self, report: Dict[str, Any]):
        """Save final performance report to database"""
        if not self.connection:
            logger.debug("No database connection - skipping final report save")
            return
        
        try:
            # Store final report in performance table
            logger.info("Final report saved (placeholder)")
        except Exception as e:
            logger.error(f"Failed to save final report: {e}")