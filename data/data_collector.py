'''
Data collection module for Trading Bot
HHandles various data sources and collection strategies
'''

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp
import yfinance as yf
import pandas as pd
from core.config import config

logger = logging.getLogger(__name__)

class DataCollector:
    """Collects market data from various sources"""
    
    def __init__(self, config=None):
        self.config = config
        self.session = None
        self.data_sources = {
            'yahoo': self._fetch_yahoo_data,
            'alpha_vantage': self._fetch_alpha_vantage_data,
            'finnhub': self._fetch_finnhub_data
        }
    
    async def initialize(self):
        """Initialize the data collector"""
        try:
            self.session = aiohttp.ClientSession()
            
            # Test data sources availability
            health_status = await self.health_check()
            available_sources = [source for source, status in health_status.items() if status]
            
            logger.info(f"DataCollector initialized. Available sources: {available_sources}")
            return True
            
        except Exception as e:
            logger.error(f"DataCollector initialization failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            logger.info("DataCollector cleanup completed")
        except Exception as e:
            logger.error(f"DataCollector cleanup failed: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def collect_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Collect market data for given symbols"""
        market_data = {}
        
        for symbol in symbols:
            try:
                # Primary source: Yahoo Finance
                data = await self._fetch_yahoo_data(symbol)
                if data:
                    market_data[symbol] = data
                else:
                    # Fallback to Alpha Vantage
                    data = await self._fetch_alpha_vantage_data(symbol)
                    if data:
                        market_data[symbol] = data
                
            except Exception as e:
                logger.error(f"Failed to collect data for {symbol}: {e}")
        
        return market_data
    
    async def _fetch_yahoo_data(self, symbol: str) -> Optional[Dict]:
        # ... (implementation details)
        pass

    async def _fetch_alpha_vantage_data(self, symbol: str) -> Optional[Dict]:
        # ... (implementation details)
        pass

    async def _fetch_finnhub_data(self, symbol: str) -> Optional[Dict]:
        # ... (implementation details)
        pass

    async def collect_news_data(self, symbols: List[str] = None) -> List[Dict]:
        # ... (implementation details)
        pass

    async def _fetch_news_api_data(self, symbols: List[str] = None) -> List[Dict]:
        # ... (implementation details)
        pass

    def get_historical_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        # ... (implementation details)
        pass

    async def collect_historical_market_data(self, symbols: List[str], days: int = 30) -> Dict[str, pd.DataFrame]:
        """Collect historical market data"""
        logger.info(f"Collecting historical market data for {len(symbols)} symbols, {days} days")

        # Use yahoo finance for now
        historical_data = {}
        for symbol in symbols[:10]:  # Limit to 10 symbols for startup
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=f"{days}d")
                if not data.empty:
                    historical_data[symbol] = data
            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {e}")

        return historical_data

    async def collect_historical_news(self, symbols: List[str] = None, days: int = 7) -> List[Dict]:
        """Collect historical news data"""
        # Return empty list for now - news collection is handled by enhanced_sentiment
        logger.info(f"Historical news collection requested for {len(symbols) if symbols else 'all'} symbols, {days} days")
        return []

    async def health_check(self) -> Dict[str, bool]:
        """Check health of data sources"""
        health_status = {}

        # Check Yahoo Finance (always available, no API key required)
        health_status['yahoo'] = True

        # Check Alpha Vantage
        health_status['alpha_vantage'] = hasattr(self.config, 'ALPHA_VANTAGE_KEY') and self.config.ALPHA_VANTAGE_KEY is not None

        # Check Finnhub
        health_status['finnhub'] = hasattr(self.config, 'FINNHUB_KEY') and self.config.FINNHUB_KEY is not None

        return health_status

    async def collect_current_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Alias for collect_market_data to match bot_orchestrator expectations"""
        return await self.collect_market_data(symbols)

    async def collect_macro_economic_data(self, analyzer) -> Dict[str, Any]:
        """Collects and analyzes macro-economic and geopolitical data."""
        logger.info("Collecting macro-economic and geopolitical data...")
        try:
            return await analyzer.get_full_macro_analysis()
        except Exception as e:
            logger.error(f"Error collecting macro-economic data: {e}")
            return {'error': str(e)}