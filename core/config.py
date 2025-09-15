"""
Configuration management for Trading Bot
Handles environment variables and configuration settings
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging

class Config:
    """Configuration manager for the trading bot"""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # API Keys
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY')
        self.finnhub_key = os.getenv('FINNHUB_KEY')
        self.gcs_models_bucket = os.getenv('GCS_MODELS_BUCKET')
        
        # Social Media APIs
        self.enable_twitter = os.getenv('ENABLE_TWITTER', 'True').lower() == 'true'
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN') if self.enable_twitter else None
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'TradingBot/1.0')
        
        # Database
        self.database_url = os.getenv('DATABASE_URL')
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        
        # Bot Configuration
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.debug_mode = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
        self.max_cycles = int(os.getenv('MAX_CYCLES', '0'))
        self.backtest_mode = os.getenv('BACKTEST_MODE', 'False').lower() == 'true'
        
        # Trading Parameters
        self.initial_capital = float(os.getenv('INITIAL_CAPITAL', '10000.0'))
        self.INITIAL_CAPITAL = self.initial_capital  # Alias for backward compatibility
        self.enable_social_sentiment = os.getenv('ENABLE_SOCIAL_SENTIMENT', 'False').lower() == 'true'
        
        # ML Configuration
        self.ENABLE_TRADITIONAL_ML = os.getenv('ENABLE_TRADITIONAL_ML', 'True').lower() == 'true'
        self.ENABLE_TRANSFORMER_ML = os.getenv('ENABLE_TRANSFORMER_ML', 'True').lower() == 'true'
        self.SKIP_ML_TRAINING = os.getenv('SKIP_ML_TRAINING', 'False').lower() == 'true'
        self.ENABLE_SOCIAL_SENTIMENT = self.enable_social_sentiment
        
        # Trading Symbols - Import from universe configuration
        try:
            from data.universe_symbols import get_high_priority_symbols, get_all_symbols, REGIONS_CONFIG
            
            # Mode de trading configurable
            trading_mode = os.getenv('TRADING_MODE', 'fast_mode')  # fast_mode, normal_mode, comprehensive_mode
            
            if trading_mode == 'fast_mode':
                self.ALL_SYMBOLS = get_high_priority_symbols()
            elif trading_mode == 'normal_mode':
                # Combine US top + EU + UK principales
                from data.universe_symbols import SP500_TOP100, CAC40_SYMBOLS, FTSE100_SYMBOLS
                self.ALL_SYMBOLS = SP500_TOP100[:100] + CAC40_SYMBOLS[:20] + FTSE100_SYMBOLS[:20]
            else:  # comprehensive_mode
                self.ALL_SYMBOLS = get_all_symbols()
            
            # Séparer par région pour compatibilité
            self.US_SYMBOLS = [s for s in self.ALL_SYMBOLS if not ('.' in s)]  # Pas de suffixe = US
            self.EU_SYMBOLS = [s for s in self.ALL_SYMBOLS if '.PA' in s or '.DE' in s]  # France + Allemagne
            self.UK_SYMBOLS = [s for s in self.ALL_SYMBOLS if '.L' in s]  # UK
            self.ASIA_SYMBOLS = [s for s in self.ALL_SYMBOLS if any(x in s for x in ['.T', '.HK', '.KS'])]  # Asie
            
            self.REGIONS_CONFIG = REGIONS_CONFIG
            
            logging.info(f"Trading mode: {trading_mode}")
            logging.info(f"Total symbols loaded: {len(self.ALL_SYMBOLS)}")
            logging.info(f"US: {len(self.US_SYMBOLS)}, EU: {len(self.EU_SYMBOLS)}, UK: {len(self.UK_SYMBOLS)}, Asia: {len(self.ASIA_SYMBOLS)}")
            
        except ImportError:
            logging.warning("Could not import universe_symbols, using basic symbols")
            # Fallback vers la configuration basique
            us_symbols_str = os.getenv('US_SYMBOLS', 'AAPL,GOOGL,MSFT,AMZN,TSLA,NVDA,META,NFLX')
            eu_symbols_str = os.getenv('EU_SYMBOLS', 'ASML,SAP,LVMH,NESN')
            
            self.US_SYMBOLS = [symbol.strip() for symbol in us_symbols_str.split(',') if symbol.strip()]
            self.EU_SYMBOLS = [symbol.strip() for symbol in eu_symbols_str.split(',') if symbol.strip()]
            self.UK_SYMBOLS = []
            self.ASIA_SYMBOLS = []
            self.ALL_SYMBOLS = self.US_SYMBOLS + self.EU_SYMBOLS
        
        # Analysis Configuration - Enhanced for better ML training
        self.ANALYSIS_LOOKBACK_DAYS = int(os.getenv('ANALYSIS_LOOKBACK_DAYS', '90'))  # 3 months default
        self.ML_TRAINING_LOOKBACK_DAYS = int(os.getenv('ML_TRAINING_LOOKBACK_DAYS', '3650'))  # 10 years for ML training  
        self.NEWS_LOOKBACK_DAYS = int(os.getenv('NEWS_LOOKBACK_DAYS', '60'))  # 2 months for news analysis
        self.MIN_CONFIDENCE_THRESHOLD = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.6'))
        
        # Validate required configuration
        self._validate_config()
    
    def get(self, key: str, default=None):
        """Dict-like get method for backward compatibility"""
        return getattr(self, key, default)
    
    def _validate_config(self):
        """Validate required configuration parameters"""
        required_keys = ['GEMINI_API_KEY']
        missing_keys = []
        
        for key in required_keys:
            if not getattr(self, key.lower()):
                missing_keys.append(key)
        
        if missing_keys:
            logging.warning(f"Missing required configuration keys: {missing_keys}")
    
    def get_api_keys(self) -> Dict[str, Optional[str]]:
        """Get all API keys as a dictionary"""
        return {
            'gemini': self.gemini_api_key,
            'news_api': self.news_api_key,
            'alpha_vantage': self.alpha_vantage_key,
            'finnhub': self.finnhub_key,
            'twitter': self.twitter_bearer_token,
            'reddit_client_id': self.reddit_client_id,
            'reddit_client_secret': self.reddit_client_secret,
        }
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading-specific configuration"""
        return {
            'initial_capital': self.initial_capital,
            'enable_social_sentiment': self.enable_social_sentiment,
            'backtest_mode': self.backtest_mode,
            'max_cycles': self.max_cycles,
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            'level': self.log_level,
            'debug_mode': self.debug_mode,
        }
    
    def get_symbol_region(self, symbol: str) -> str:
        """Get the region for a given symbol"""
        if symbol in self.US_SYMBOLS:
            return 'US'
        elif symbol in self.EU_SYMBOLS:
            return 'EU'
        else:
            return 'US'  # Default to US

# Global configuration instance
config = Config()