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
        """Fetch current market data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)

            # Try to get info first
            try:
                info = ticker.info
            except Exception as e:
                logger.warning(f"Failed to get info for {symbol}: {e}")
                info = {}

            # Try to get historical data
            try:
                hist = ticker.history(period="1d", interval="1h")
                if hist.empty:
                    # Fallback to daily data
                    hist = ticker.history(period="2d", interval="1d")
            except Exception as e:
                logger.warning(f"Failed to get history for {symbol}: {e}")
                hist = pd.DataFrame()

            if hist.empty and not info:
                logger.warning(f"No data available for {symbol}")
                return None

            # Get current price from available source
            current_price = 0
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
            elif info.get('currentPrice'):
                current_price = info.get('currentPrice')
            elif info.get('regularMarketPrice'):
                current_price = info.get('regularMarketPrice')

            if current_price == 0:
                logger.warning(f"No valid price found for {symbol}")
                return None

            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'open': float(hist['Open'].iloc[-1]) if not hist.empty else info.get('open', current_price),
                'high': float(hist['High'].iloc[-1]) if not hist.empty else info.get('dayHigh', current_price),
                'low': float(hist['Low'].iloc[-1]) if not hist.empty else info.get('dayLow', current_price),
                'volume': int(hist['Volume'].iloc[-1]) if not hist.empty else info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'sector': info.get('sector', 'Unknown'),
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.warning(f"Error fetching Yahoo data for {symbol}: {e}")
            return None

    async def _fetch_alpha_vantage_data(self, symbol: str) -> Optional[Dict]:
        """Fetch current market data from Alpha Vantage"""
        if not hasattr(self.config, 'alpha_vantage_key') or not self.config.alpha_vantage_key:
            return None

        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.config.alpha_vantage_key
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    quote = data.get('Global Quote', {})

                    if quote:
                        return {
                            'symbol': symbol,
                            'current_price': float(quote.get('05. price', 0)),
                            'open': float(quote.get('02. open', 0)),
                            'high': float(quote.get('03. high', 0)),
                            'low': float(quote.get('04. low', 0)),
                            'volume': int(quote.get('06. volume', 0)),
                            'change_percent': float(quote.get('10. change percent', '0%').replace('%', '')),
                            'timestamp': datetime.now()
                        }

        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")

        return None

    async def _fetch_finnhub_data(self, symbol: str) -> Optional[Dict]:
        # ... (implementation details)
        pass

    async def collect_news_data(self, symbols: List[str] = None) -> List[Dict]:
        """Collect news data from available sources"""
        news_data = []

        try:
            # Try News API if available
            if hasattr(self.config, 'news_api_key') and self.config.news_api_key:
                news_api_data = await self._fetch_news_api_data(symbols)
                news_data.extend(news_api_data)
                logger.info(f"Collected {len(news_api_data)} articles from News API")

            # Try Alpha Vantage News if available
            if hasattr(self.config, 'alpha_vantage_key') and self.config.alpha_vantage_key:
                av_news_data = await self._fetch_alpha_vantage_news(symbols)
                news_data.extend(av_news_data)
                logger.info(f"Collected {len(av_news_data)} articles from Alpha Vantage")

            # Try Finnhub News if available
            if hasattr(self.config, 'finnhub_key') and self.config.finnhub_key:
                finnhub_news_data = await self._fetch_finnhub_news(symbols)
                news_data.extend(finnhub_news_data)
                logger.info(f"Collected {len(finnhub_news_data)} articles from Finnhub")

            logger.info(f"Total news articles collected: {len(news_data)}")
            return news_data

        except Exception as e:
            logger.error(f"Error collecting news data: {e}")
            return []

    async def _fetch_news_api_data(self, symbols: List[str] = None) -> List[Dict]:
        """Fetch news from NewsAPI.org"""
        if not hasattr(self.config, 'news_api_key') or not self.config.news_api_key:
            return []

        news_articles = []
        try:
            url = "https://newsapi.org/v2/everything"
            headers = {'X-API-Key': self.config.news_api_key}

            # Search for general financial news
            params = {
                'q': 'stocks OR market OR trading OR finance',
                'domains': 'reuters.com,bloomberg.com,cnbc.com,marketwatch.com,yahoo.com',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 50
            }

            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get('articles', [])

                    for article in articles:
                        # Extract company mentions from title/description
                        companies_mentioned = []
                        title_desc = f"{article.get('title', '')} {article.get('description', '')}".upper()

                        if symbols:
                            for symbol in symbols[:20]:  # Check first 20 symbols
                                if symbol in title_desc:
                                    companies_mentioned.append(symbol)

                        news_articles.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'content': article.get('content', ''),
                            'url': article.get('url', ''),
                            'published_at': article.get('publishedAt', ''),
                            'source': 'newsapi',
                            'companies_mentioned': companies_mentioned
                        })

        except Exception as e:
            logger.error(f"Error fetching NewsAPI data: {e}")

        return news_articles

    async def _fetch_alpha_vantage_news(self, symbols: List[str] = None) -> List[Dict]:
        """Fetch news from Alpha Vantage"""
        if not hasattr(self.config, 'alpha_vantage_key') or not self.config.alpha_vantage_key:
            return []

        news_articles = []
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'apikey': self.config.alpha_vantage_key,
                'limit': 50
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get('feed', [])

                    for article in articles:
                        # Extract ticker relevance
                        companies_mentioned = []
                        ticker_sentiment = article.get('ticker_sentiment', [])
                        for ticker_info in ticker_sentiment:
                            ticker = ticker_info.get('ticker', '')
                            if ticker and symbols and ticker in symbols:
                                companies_mentioned.append(ticker)

                        news_articles.append({
                            'title': article.get('title', ''),
                            'description': article.get('summary', ''),
                            'content': article.get('summary', ''),
                            'url': article.get('url', ''),
                            'published_at': article.get('time_published', ''),
                            'source': 'alpha_vantage',
                            'companies_mentioned': companies_mentioned,
                            'overall_sentiment_score': article.get('overall_sentiment_score', 0)
                        })

        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")

        return news_articles

    async def _fetch_finnhub_news(self, symbols: List[str] = None) -> List[Dict]:
        """Fetch news from Finnhub"""
        if not hasattr(self.config, 'finnhub_key') or not self.config.finnhub_key:
            return []

        news_articles = []
        try:
            # General market news
            url = "https://finnhub.io/api/v1/news"
            headers = {'X-Finnhub-Token': self.config.finnhub_key}
            params = {'category': 'general'}

            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    articles = await response.json()

                    for article in articles[:30]:  # Limit to 30 articles
                        # Simple company mention extraction
                        companies_mentioned = []
                        headline_summary = f"{article.get('headline', '')} {article.get('summary', '')}".upper()

                        if symbols:
                            for symbol in symbols[:20]:  # Check first 20 symbols
                                if symbol in headline_summary:
                                    companies_mentioned.append(symbol)

                        news_articles.append({
                            'title': article.get('headline', ''),
                            'description': article.get('summary', ''),
                            'content': article.get('summary', ''),
                            'url': article.get('url', ''),
                            'published_at': article.get('datetime', ''),
                            'source': 'finnhub',
                            'companies_mentioned': companies_mentioned
                        })

        except Exception as e:
            logger.error(f"Error fetching Finnhub news: {e}")

        return news_articles

    def get_historical_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        # ... (implementation details)
        pass

    async def collect_historical_market_data(self, symbols: List[str], days: int = 30) -> Dict[str, pd.DataFrame]:
        """Collect historical market data"""
        logger.info(f"Collecting historical market data for {len(symbols)} symbols, {days} days")

        # Use yahoo finance for now
        historical_data = {}
        for symbol in symbols:  # Process all symbols
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