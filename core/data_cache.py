"""
Advanced Data Caching System
Manages caching of historical data, models, and analysis results
"""

import logging
import pickle
import json
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import sqlite3
import threading

logger = logging.getLogger(__name__)

class DataCacheManager:
    """Manages data caching with SQLite backend and file-based storage"""
    
    def __init__(self, config, cache_dir: str = "cache"):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize SQLite cache database
        self.db_path = self.cache_dir / "cache_metadata.db"
        self.connection = None
        self.lock = threading.Lock()
        
        # Cache settings
        self.default_ttl = timedelta(hours=6)  # Default cache time-to-live
        self.cache_settings = {
            'historical_market_data': timedelta(hours=12),
            'historical_news': timedelta(hours=2),
            'social_sentiment': timedelta(minutes=30),
            'ml_models': timedelta(days=1),
            'analysis_results': timedelta(hours=1)
        }
        
        self._initialize_cache_db()
        logger.info(f"Data cache manager initialized at {cache_dir}")
    
    def _initialize_cache_db(self):
        """Initialize SQLite cache metadata database"""
        try:
            self.connection = sqlite3.connect(
                self.db_path, 
                check_same_thread=False,
                timeout=30.0
            )
            
            # Create cache metadata table
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    cache_key TEXT PRIMARY KEY,
                    cache_type TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    file_path TEXT,
                    data_hash TEXT,
                    metadata TEXT
                )
            """)
            
            # Create index for performance
            self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_type_expires 
                ON cache_metadata(cache_type, expires_at)
            """)
            
            self.connection.commit()
            logger.info("Cache database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache database: {e}")
            self.connection = None
    
    def _generate_cache_key(self, cache_type: str, identifier: str, params: Dict = None) -> str:
        """Generate unique cache key"""
        key_data = f"{cache_type}:{identifier}"
        if params:
            params_str = json.dumps(params, sort_keys=True)
            key_data += f":{params_str}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if not self.connection:
            return False
        
        try:
            with self.lock:
                cursor = self.connection.execute(
                    "SELECT expires_at FROM cache_metadata WHERE cache_key = ?",
                    (cache_key,)
                )
                result = cursor.fetchone()
                
                if result:
                    expires_at = datetime.fromisoformat(result[0])
                    return datetime.now() < expires_at
                
                return False
                
        except Exception as e:
            logger.error(f"Error checking cache validity: {e}")
            return False
    
    def _save_to_cache(self, cache_key: str, data: Any, cache_type: str, 
                      metadata: Dict = None, ttl: timedelta = None) -> bool:
        """Save data to cache"""
        try:
            ttl = ttl or self.cache_settings.get(cache_type, self.default_ttl)
            created_at = datetime.now()
            expires_at = created_at + ttl
            
            # Generate file path
            file_path = self.cache_dir / f"{cache_key}.pkl"
            
            # Save data to file
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Calculate data hash for integrity
            data_hash = hashlib.md5(str(data).encode()).hexdigest()
            
            # Save metadata to database
            if self.connection:
                with self.lock:
                    self.connection.execute("""
                        INSERT OR REPLACE INTO cache_metadata 
                        (cache_key, cache_type, created_at, expires_at, file_path, data_hash, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        cache_key,
                        cache_type,
                        created_at.isoformat(),
                        expires_at.isoformat(),
                        str(file_path),
                        data_hash,
                        json.dumps(metadata or {})
                    ))
                    self.connection.commit()
            
            logger.debug(f"Data cached with key {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save to cache: {e}")
            return False
    
    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """Load data from cache"""
        try:
            if not self._is_cache_valid(cache_key):
                return None
            
            # Get file path from database
            if self.connection:
                with self.lock:
                    cursor = self.connection.execute(
                        "SELECT file_path FROM cache_metadata WHERE cache_key = ?",
                        (cache_key,)
                    )
                    result = cursor.fetchone()
                    
                    if result:
                        file_path = Path(result[0])
                        if file_path.exists():
                            with open(file_path, 'rb') as f:
                                data = pickle.load(f)
                            
                            logger.debug(f"Data loaded from cache with key {cache_key}")
                            return data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load from cache: {e}")
            return None
    
    def get_historical_market_data(self, symbols: List[str], days: int = 180) -> Optional[Dict[str, pd.DataFrame]]:
        """Get historical market data from cache"""
        cache_key = self._generate_cache_key(
            'historical_market_data', 
            ','.join(sorted(symbols)),
            {'days': days}
        )
        
        return self._load_from_cache(cache_key)
    
    def save_historical_market_data(self, data: Dict[str, pd.DataFrame], 
                                  symbols: List[str], days: int = 180) -> bool:
        """Save historical market data to cache"""
        cache_key = self._generate_cache_key(
            'historical_market_data',
            ','.join(sorted(symbols)),
            {'days': days}
        )
        
        metadata = {
            'symbols_count': len(symbols),
            'total_records': sum(len(df) for df in data.values()),
            'symbols': symbols[:10]  # Store first 10 symbols for reference
        }
        
        return self._save_to_cache(cache_key, data, 'historical_market_data', metadata)
    
    def get_historical_news(self, symbols: List[str], days: int = 30) -> Optional[List[Dict]]:
        """Get historical news data from cache"""
        cache_key = self._generate_cache_key(
            'historical_news',
            ','.join(sorted(symbols)),
            {'days': days}
        )
        
        return self._load_from_cache(cache_key)
    
    def save_historical_news(self, news_data: List[Dict], 
                           symbols: List[str], days: int = 30) -> bool:
        """Save historical news data to cache"""
        cache_key = self._generate_cache_key(
            'historical_news',
            ','.join(sorted(symbols)),
            {'days': days}
        )
        
        metadata = {
            'news_count': len(news_data),
            'symbols_count': len(symbols),
            'date_range': days
        }
        
        return self._save_to_cache(cache_key, news_data, 'historical_news', metadata)
    
    def get_social_sentiment(self, symbols: List[str], days: int = 7) -> Optional[List[Dict]]:
        """Get social sentiment data from cache"""
        cache_key = self._generate_cache_key(
            'social_sentiment',
            ','.join(sorted(symbols)),
            {'days': days}
        )
        
        return self._load_from_cache(cache_key)
    
    def save_social_sentiment(self, sentiment_data: List[Dict], 
                            symbols: List[str], days: int = 7) -> bool:
        """Save social sentiment data to cache"""
        cache_key = self._generate_cache_key(
            'social_sentiment',
            ','.join(sorted(symbols)),
            {'days': days}
        )
        
        metadata = {
            'sentiment_count': len(sentiment_data),
            'symbols_count': len(symbols)
        }
        
        return self._save_to_cache(
            cache_key, sentiment_data, 'social_sentiment', metadata,
            ttl=timedelta(minutes=30)  # Short TTL for social sentiment
        )
    
    def get_ml_model(self, model_type: str, symbol: str = None) -> Optional[Any]:
        """Get trained ML model from cache"""
        identifier = symbol or 'global'
        cache_key = self._generate_cache_key('ml_models', f"{model_type}_{identifier}")
        
        return self._load_from_cache(cache_key)
    
    def save_ml_model(self, model: Any, model_type: str, symbol: str = None,
                     training_metadata: Dict = None) -> bool:
        """Save trained ML model to cache"""
        identifier = symbol or 'global'
        cache_key = self._generate_cache_key('ml_models', f"{model_type}_{identifier}")
        
        metadata = training_metadata or {}
        metadata.update({
            'model_type': model_type,
            'symbol': symbol,
            'cached_at': datetime.now().isoformat()
        })
        
        return self._save_to_cache(
            cache_key, model, 'ml_models', metadata,
            ttl=timedelta(days=1)  # Models cache for 1 day
        )
    
    def cleanup_expired_cache(self):
        """Remove expired cache entries"""
        try:
            if not self.connection:
                return
            
            current_time = datetime.now()
            
            with self.lock:
                # Get expired entries
                cursor = self.connection.execute(
                    "SELECT cache_key, file_path FROM cache_metadata WHERE expires_at < ?",
                    (current_time.isoformat(),)
                )
                expired_entries = cursor.fetchall()
                
                # Delete expired files and database entries
                deleted_count = 0
                for cache_key, file_path in expired_entries:
                    try:
                        # Delete file
                        if file_path and Path(file_path).exists():
                            Path(file_path).unlink()
                        
                        # Delete database entry
                        self.connection.execute(
                            "DELETE FROM cache_metadata WHERE cache_key = ?",
                            (cache_key,)
                        )
                        deleted_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error deleting cache entry {cache_key}: {e}")
                
                self.connection.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} expired cache entries")
                    
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            if not self.connection:
                return {'error': 'Cache database not available'}
            
            with self.lock:
                # Total cache entries
                cursor = self.connection.execute("SELECT COUNT(*) FROM cache_metadata")
                total_entries = cursor.fetchone()[0]
                
                # Entries by type
                cursor = self.connection.execute("""
                    SELECT cache_type, COUNT(*) 
                    FROM cache_metadata 
                    GROUP BY cache_type
                """)
                entries_by_type = dict(cursor.fetchall())
                
                # Expired entries
                current_time = datetime.now()
                cursor = self.connection.execute(
                    "SELECT COUNT(*) FROM cache_metadata WHERE expires_at < ?",
                    (current_time.isoformat(),)
                )
                expired_entries = cursor.fetchone()[0]
                
                # Cache directory size
                cache_size = sum(f.stat().st_size for f in self.cache_dir.glob('*') if f.is_file())
                cache_size_mb = cache_size / (1024 * 1024)
                
                return {
                    'total_entries': total_entries,
                    'entries_by_type': entries_by_type,
                    'expired_entries': expired_entries,
                    'cache_size_mb': round(cache_size_mb, 2),
                    'cache_directory': str(self.cache_dir)
                }
                
        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return {'error': str(e)}
    
    def clear_cache(self, cache_type: str = None):
        """Clear cache entries"""
        try:
            if not self.connection:
                return
            
            with self.lock:
                if cache_type:
                    # Clear specific cache type
                    cursor = self.connection.execute(
                        "SELECT cache_key, file_path FROM cache_metadata WHERE cache_type = ?",
                        (cache_type,)
                    )
                else:
                    # Clear all cache
                    cursor = self.connection.execute(
                        "SELECT cache_key, file_path FROM cache_metadata"
                    )
                
                entries_to_delete = cursor.fetchall()
                
                # Delete files and database entries
                for cache_key, file_path in entries_to_delete:
                    try:
                        if file_path and Path(file_path).exists():
                            Path(file_path).unlink()
                    except Exception as e:
                        logger.error(f"Error deleting cache file {file_path}: {e}")
                
                # Clear database entries
                if cache_type:
                    self.connection.execute(
                        "DELETE FROM cache_metadata WHERE cache_type = ?",
                        (cache_type,)
                    )
                else:
                    self.connection.execute("DELETE FROM cache_metadata")
                
                self.connection.commit()
                
                logger.info(f"Cache cleared: {len(entries_to_delete)} entries removed")
                
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def close(self):
        """Close cache manager"""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
            logger.info("Cache manager closed")
        except Exception as e:
            logger.error(f"Error closing cache manager: {e}")