"""
Enhanced Multi-Source Sentiment Analysis
Aggregates sentiment from news, social media, press releases, SEC filings, and other sources
"""

import logging
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import feedparser
import re
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class SentimentData:
    """Structured sentiment data"""
    source: str
    content: str
    title: str
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    market_impact: float  # 0 to 1
    urgency: float  # 0 to 1
    symbols_mentioned: List[str]
    timestamp: datetime
    url: str = None
    author: str = None
    
class EnhancedSentimentAnalyzer:
    """Multi-source sentiment analyzer with extensive coverage"""
    
    def __init__(self, config, gemini_sentiment_analyzer):
        self.config = config
        self.gemini_analyzer = gemini_sentiment_analyzer
        self.session = None
        
        # News sources (RSS feeds and APIs) - with fallback URLs
        self.news_sources = {
            # Financial news RSS feeds (primary)
            'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/marketpulse/',
            'investing_news': 'https://www.investing.com/rss/news.rss',
            'benzinga': 'https://feeds.benzinga.com/benzinga',
            'zacks_stock_news': 'https://www.zacks.com/rss/rss_news_stock.php',
            'google_finance': 'https://news.google.com/rss/search?q=stocks&hl=en-US&gl=US&ceid=US:en',
            'nasdaq_news': 'https://www.nasdaq.com/feed/rssoutbound?category=stocks',
            'finviz_news': 'https://finviz.com/news.ashx',
            # Alternative sources
            'reuters_business': 'https://feeds.reuters.com/reuters/businessNews',
            'reuters_markets': 'https://feeds.reuters.com/reuters/marketsNews',
            'bloomberg_markets': 'https://feeds.bloomberg.com/markets/news.rss',
            'cnbc_finance': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664',
        }
        
        # Press release sources
        self.press_release_sources = {
            'prnewswire': 'https://www.prnewswire.com/rss/news-releases-list.rss',
            'businesswire': 'https://www.businesswire.com/portal/site/home/news/',
            'globe_newswire': 'https://www.globenewswire.com/RssFeed/subjectcode/02-02/feedTitle/GlobeNewswire%20-%20Financial%20Services',
        }
        
        # Economic indicators sources
        self.economic_sources = {
            'fred_indicators': 'https://fred.stlouisfed.org/rss/releases',
            'bls_news': 'https://www.bls.gov/rss/rss_news_releases.htm',
            'census_econ': 'https://www.census.gov/economic-indicators/',
        }
        
        # Social media sources (beyond Reddit/Twitter)
        self.social_sources = {
            'stocktwits_trending': 'https://api.stocktwits.com/api/2/trending/symbols.json',
            'reddit_wallstreetbets': 'wallstreetbets',
            'reddit_investing': 'investing',
            'reddit_stocks': 'stocks',
            'reddit_stockmarket': 'StockMarket',
            'reddit_securityanalysis': 'SecurityAnalysis',
            'reddit_valueinvesting': 'ValueInvesting',
        }
        
        # Corporate communication sources
        self.corporate_sources = {
            'sec_filings': 'https://www.sec.gov/cgi-bin/browse-edgar',
            'earnings_calls': [],  # To be populated with earnings calendar
        }
        
        # Analyst sources
        self.analyst_sources = {
            'tipranks': 'https://www.tipranks.com',
            'morningstar': 'https://www.morningstar.com',
            'seeking_alpha_analysis': 'https://seekingalpha.com',
        }
        
        # Rate limiting
        self.rate_limits = {
            'rss': 2.0,  # seconds between requests
            'api': 1.0,
            'web_scraping': 5.0
        }
        
        logger.info(f"Enhanced sentiment analyzer initialized with {len(self.news_sources)} news sources")
    
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'TradingBot/1.0 (Financial Analysis)'
            }
        )
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()

    async def analyze_financial_sentiment(self, text: str, company: str = None, region: str = 'US') -> Dict[str, Any]:
        """Analyze financial sentiment for a given text"""
        try:
            # Use Gemini analyzer if available
            if self.gemini_analyzer and hasattr(self.gemini_analyzer, 'analyze_sentiment'):
                result = await self.gemini_analyzer.analyze_sentiment(text, company, region)
                return result

            # Fallback to simple sentiment analysis
            positive_words = ['gain', 'rise', 'up', 'bull', 'positive', 'growth', 'profit', 'buy', 'strong', 'beat', 'exceed']
            negative_words = ['loss', 'fall', 'down', 'bear', 'negative', 'decline', 'sell', 'weak', 'miss', 'below']

            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)

            total_words = len(text.split())
            if total_words == 0:
                return {'sentiment_score': 0.0, 'confidence': 0.0}

            sentiment_score = (positive_count - negative_count) / max(total_words, 1)
            sentiment_score = max(-1.0, min(1.0, sentiment_score * 10))  # Scale and clamp

            confidence = min(1.0, (positive_count + negative_count) / max(total_words, 1) * 5)

            return {
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'positive_indicators': positive_count,
                'negative_indicators': negative_count,
                'text_length': len(text)
            }

        except Exception as e:
            logger.error(f"Error in financial sentiment analysis: {e}")
            return {'sentiment_score': 0.0, 'confidence': 0.0}
    
    async def _fetch_rss_feed(self, url: str, source_name: str) -> List[Dict[str, Any]]:
        """Fetch and parse RSS feed with improved error handling"""
        try:
            await asyncio.sleep(self.rate_limits['rss'])  # Rate limiting
            
            if not self.session:
                return []
            
            # Use shorter timeout for testing
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            
            async with self.session.get(url, timeout=timeout, ssl=False) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    articles = []
                    # Handle both entries and items (different RSS formats)
                    entries = getattr(feed, 'entries', []) or getattr(feed, 'items', [])
                    
                    for entry in entries[:50]:  # Limit to 50 most recent
                        # Handle different RSS field names
                        title = (entry.get('title', '') or 
                                entry.get('title_detail', {}).get('value', '') or 
                                'Untitled')
                        
                        content = (entry.get('summary', '') or 
                                 entry.get('description', '') or
                                 entry.get('content', [{}])[0].get('value', '') if entry.get('content') else '' or
                                 title)  # Fallback to title if no content
                        
                        url_link = (entry.get('link', '') or 
                                   entry.get('id', '') or
                                   entry.get('href', ''))
                        
                        if title and len(title.strip()) > 5:  # Only include articles with meaningful titles
                            article = {
                                'source': source_name,
                                'title': title.strip(),
                                'content': content.strip()[:1000],  # Limit content length
                                'url': url_link,
                                'published': entry.get('published_parsed'),
                                'timestamp': datetime.now()
                            }
                            
                            # Extract symbols mentioned
                            article['symbols_mentioned'] = self._extract_symbols_from_text(
                                f"{article['title']} {article['content']}"
                            )
                            
                            articles.append(article)
                    
                    if articles:
                        logger.debug(f"✅ {source_name}: {len(articles)} articles")
                    else:
                        logger.debug(f"⚠️ {source_name}: RSS parsed but no articles found")
                    return articles
                    
                else:
                    logger.debug(f"❌ {source_name}: HTTP {response.status}")
                    
        except asyncio.TimeoutError:
            logger.debug(f"⏰ {source_name}: Timeout")
        except aiohttp.ClientConnectorError:
            logger.debug(f"🔌 {source_name}: Connection failed")
        except Exception as e:
            logger.debug(f"❌ {source_name}: {type(e).__name__}")
        
        return []
    
    def _extract_symbols_from_text(self, text: str) -> List[str]:
        """Extract stock symbols from text content"""
        symbols = []
        
        # Look for ticker patterns
        ticker_pattern = r'\\b([A-Z]{1,5})\\b'
        potential_tickers = re.findall(ticker_pattern, text)
        
        # Filter against known symbols
        for ticker in potential_tickers:
            if ticker in self.config.ALL_SYMBOLS:
                symbols.append(ticker)
        
        # Look for company names and map to symbols
        for symbol in self.config.ALL_SYMBOLS[:20]:  # Check top symbols
            # Simple company name matching (would need enhancement)
            symbol_lower = symbol.lower()
            if symbol_lower in text.lower():
                symbols.append(symbol)
        
        return list(set(symbols))  # Remove duplicates
    
    async def _analyze_sentiment_batch(self, articles: List[Dict[str, Any]]) -> List[SentimentData]:
        """Analyze sentiment for a batch of articles"""
        sentiment_results = []
        
        # Process in batches to avoid overwhelming the sentiment analyzer
        batch_size = 10
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            
            # Analyze each article in the batch
            for article in batch:
                try:
                    # Use Gemini for sentiment analysis
                    text_to_analyze = f"{article['title']}. {article['content']}"
                    
                    if len(text_to_analyze.strip()) < 20:  # Skip very short content
                        continue
                    
                    # Primary company for analysis
                    companies_mentioned = article.get('symbols_mentioned', [])
                    primary_company = companies_mentioned[0] if companies_mentioned else None
                    
                    # Get region for the primary symbol
                    region = 'US'
                    if primary_company and hasattr(self.config, 'get_symbol_region'):
                        region = self.config.get_symbol_region(primary_company)
                    
                    # Analyze sentiment
                    sentiment_result = await self.gemini_analyzer.analyze_financial_sentiment(
                        text=text_to_analyze,
                        company=primary_company,
                        region=region
                    )
                    
                    # Create structured sentiment data
                    sentiment_data = SentimentData(
                        source=article['source'],
                        content=article['content'][:500],  # Truncate for storage
                        title=article['title'],
                        sentiment_score=sentiment_result.get('sentiment_score', 0.0),
                        confidence=sentiment_result.get('confidence', 0.5),
                        market_impact=sentiment_result.get('market_impact', 0.5),
                        urgency=sentiment_result.get('urgency', 0.5),
                        symbols_mentioned=companies_mentioned,
                        timestamp=article['timestamp'],
                        url=article.get('url'),
                        author=article.get('author')
                    )
                    
                    sentiment_results.append(sentiment_data)
                    
                except Exception as e:
                    logger.error(f"Error analyzing sentiment for article from {article['source']}: {e}")
                    continue
            
            # Small delay between batches
            await asyncio.sleep(0.5)
        
        return sentiment_results
    
    async def _fetch_press_releases(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Fetch recent press releases for given symbols"""
        press_releases = []
        
        for source_name, url in self.press_release_sources.items():
            try:
                articles = await self._fetch_rss_feed(url, f"press_release_{source_name}")
                
                # Filter for relevant symbols
                relevant_articles = []
                for article in articles:
                    if any(symbol in article.get('symbols_mentioned', []) for symbol in symbols):
                        relevant_articles.append(article)
                
                press_releases.extend(relevant_articles)
                
            except Exception as e:
                logger.error(f"Error fetching press releases from {source_name}: {e}")
        
        return press_releases
    
    async def _fetch_economic_indicators(self) -> List[Dict[str, Any]]:
        """Fetch economic indicators that could affect markets"""
        economic_news = []
        
        for source_name, url in self.economic_sources.items():
            try:
                if url.endswith('.rss') or 'rss' in url:
                    articles = await self._fetch_rss_feed(url, f"economic_{source_name}")
                    economic_news.extend(articles)
            except Exception as e:
                logger.error(f"Error fetching economic data from {source_name}: {e}")
        
        return economic_news
    
    async def _get_social_sentiment_extended(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Get sentiment from extended social media sources"""
        social_sentiment = []
        
        # StockTwits trending (if API available)
        try:
            if self.session:
                async with self.session.get('https://api.stocktwits.com/api/2/trending/symbols.json') as response:
                    if response.status == 200:
                        data = await response.json()
                        trending_symbols = data.get('symbols', [])
                        
                        for symbol_data in trending_symbols[:20]:  # Top 20 trending
                            symbol = symbol_data.get('symbol')
                            if symbol in symbols:
                                social_sentiment.append({
                                    'source': 'stocktwits_trending',
                                    'symbol': symbol,
                                    'title': f"Trending: {symbol}",
                                    'content': f"Symbol {symbol} is trending on StockTwits",
                                    'symbols_mentioned': [symbol],
                                    'timestamp': datetime.now(),
                                    'sentiment_score': 0.1,  # Mild positive for trending
                                    'confidence': 0.6
                                })
        except Exception as e:
            logger.error(f"Error fetching StockTwits data: {e}")
        
        return social_sentiment
    
    async def collect_comprehensive_sentiment(self, symbols: List[str], 
                                           lookback_hours: int = 24) -> List[SentimentData]:
        """Collect sentiment from all available sources"""
        
        logger.info(f"Collecting comprehensive sentiment for {len(symbols)} symbols (last {lookback_hours}h)")
        
        all_sentiment_data = []
        
        try:
            # 1. Fetch from all news sources (prioritize reliable ones)
            logger.info("Fetching from news sources...")
            
            # Prioritize sources (most reliable first)
            priority_sources = ['yahoo_finance', 'marketwatch', 'investing_news', 'google_finance']
            backup_sources = [src for src in self.news_sources.keys() if src not in priority_sources]
            
            news_results = []
            successful_sources = []
            
            # Try priority sources first
            logger.info("Trying priority sources...")
            for source_name in priority_sources:
                if source_name in self.news_sources:
                    try:
                        url = self.news_sources[source_name]
                        articles = await self._fetch_rss_feed(url, source_name)
                        if articles:
                            news_results.extend(articles)
                            successful_sources.append(source_name)
                            logger.info(f"✅ {source_name}: {len(articles)} articles")
                        
                        # Stop if we have enough articles from reliable sources
                        if len(news_results) >= 20:
                            break
                            
                    except Exception as e:
                        logger.debug(f"Priority source {source_name} failed: {e}")
            
            # Try backup sources if needed
            if len(news_results) < 10:
                logger.info("Trying backup sources...")
                for source_name in backup_sources[:3]:  # Try max 3 backup sources
                    try:
                        url = self.news_sources[source_name]
                        articles = await self._fetch_rss_feed(url, source_name)
                        if articles:
                            news_results.extend(articles)
                            successful_sources.append(source_name)
                            logger.info(f"✅ {source_name}: {len(articles)} articles")
                    except Exception as e:
                        logger.debug(f"Backup source {source_name} failed: {e}")
            
            logger.info(f"Collected {len(news_results)} news articles from {len(successful_sources)} sources")
            
            # 2. Fetch press releases
            logger.info("Fetching press releases...")
            press_releases = await self._fetch_press_releases(symbols)
            logger.info(f"Collected {len(press_releases)} press releases")
            
            # 3. Fetch economic indicators
            logger.info("Fetching economic indicators...")
            economic_news = await self._fetch_economic_indicators()
            logger.info(f"Collected {len(economic_news)} economic indicators")
            
            # 4. Extended social media
            logger.info("Fetching extended social media sentiment...")
            social_data = await self._get_social_sentiment_extended(symbols)
            logger.info(f"Collected {len(social_data)} social media data points")
            
            # Combine all sources
            all_articles = news_results + press_releases + economic_news + social_data
            
            # Filter for relevance to our symbols
            relevant_articles = []
            for article in all_articles:
                # Check if article mentions any of our symbols
                mentions_symbol = False
                article_text = f"{article.get('title', '')} {article.get('content', '')}"
                
                for symbol in symbols:
                    if (symbol in article.get('symbols_mentioned', []) or 
                        symbol.lower() in article_text.lower()):
                        mentions_symbol = True
                        break
                
                # Also include general market news (no specific symbol mentioned)
                if not mentions_symbol and not article.get('symbols_mentioned'):
                    if any(word in article_text.lower() for word in 
                          ['market', 'stocks', 'trading', 'economy', 'federal reserve', 'inflation']):
                        mentions_symbol = True
                
                if mentions_symbol:
                    relevant_articles.append(article)
            
            logger.info(f"Filtered to {len(relevant_articles)} relevant articles")
            
            # 5. Analyze sentiment for all relevant articles
            logger.info("Analyzing sentiment for all articles...")
            sentiment_results = await self._analyze_sentiment_batch(relevant_articles)
            
            logger.info(f"Completed sentiment analysis for {len(sentiment_results)} articles")
            
            # Filter by time window
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            recent_sentiment = [
                s for s in sentiment_results 
                if s.timestamp >= cutoff_time
            ]
            
            logger.info(f"Final result: {len(recent_sentiment)} sentiment data points in last {lookback_hours}h")
            
            return recent_sentiment
            
        except Exception as e:
            logger.error(f"Error in comprehensive sentiment collection: {e}")
            return []
    
    def aggregate_sentiment_by_symbol(self, sentiment_data: List[SentimentData]) -> Dict[str, Dict[str, Any]]:
        """Aggregate sentiment data by symbol"""
        
        symbol_sentiment = {}
        
        for data in sentiment_data:
            for symbol in data.symbols_mentioned:
                if symbol not in symbol_sentiment:
                    symbol_sentiment[symbol] = {
                        'sentiment_scores': [],
                        'confidence_scores': [],
                        'market_impact_scores': [],
                        'urgency_scores': [],
                        'sources': set(),
                        'article_count': 0,
                        'recent_articles': []
                    }
                
                symbol_sentiment[symbol]['sentiment_scores'].append(data.sentiment_score)
                symbol_sentiment[symbol]['confidence_scores'].append(data.confidence)
                symbol_sentiment[symbol]['market_impact_scores'].append(data.market_impact)
                symbol_sentiment[symbol]['urgency_scores'].append(data.urgency)
                symbol_sentiment[symbol]['sources'].add(data.source)
                symbol_sentiment[symbol]['article_count'] += 1
                
                # Store recent articles for reference
                symbol_sentiment[symbol]['recent_articles'].append({
                    'title': data.title,
                    'source': data.source,
                    'timestamp': data.timestamp,
                    'sentiment_score': data.sentiment_score,
                    'url': data.url
                })
        
        # Calculate aggregated metrics
        aggregated_sentiment = {}
        for symbol, data in symbol_sentiment.items():
            if data['sentiment_scores']:
                import numpy as np
                
                aggregated_sentiment[symbol] = {
                    'avg_sentiment': np.mean(data['sentiment_scores']),
                    'sentiment_std': np.std(data['sentiment_scores']),
                    'avg_confidence': np.mean(data['confidence_scores']),
                    'avg_market_impact': np.mean(data['market_impact_scores']),
                    'avg_urgency': np.mean(data['urgency_scores']),
                    'source_diversity': len(data['sources']),
                    'total_articles': data['article_count'],
                    'sources_used': list(data['sources']),
                    'recent_articles': data['recent_articles'][-5:],  # Last 5 articles
                    'sentiment_trend': 'positive' if np.mean(data['sentiment_scores']) > 0.1 else 
                                    'negative' if np.mean(data['sentiment_scores']) < -0.1 else 'neutral'
                }
        
        return aggregated_sentiment
    
    async def get_enhanced_sentiment_analysis(self, symbols: List[str], 
                                           lookback_hours: int = 6) -> Dict[str, Any]:
        """Main method to get enhanced sentiment analysis"""
        
        # Initialize if not already done
        if not self.session:
            await self.initialize()
        
        try:
            # Collect comprehensive sentiment
            sentiment_data = await self.collect_comprehensive_sentiment(symbols, lookback_hours)
            
            # Aggregate by symbol
            aggregated_sentiment = self.aggregate_sentiment_by_symbol(sentiment_data)
            
            # Overall market sentiment
            if sentiment_data:
                import numpy as np
                all_scores = [d.sentiment_score for d in sentiment_data]
                market_sentiment = {
                    'overall_sentiment': np.mean(all_scores),
                    'sentiment_volatility': np.std(all_scores),
                    'total_data_points': len(sentiment_data),
                    'sources_count': len(set(d.source for d in sentiment_data)),
                    'time_range_hours': lookback_hours
                }
            else:
                market_sentiment = {
                    'overall_sentiment': 0.0,
                    'sentiment_volatility': 0.0,
                    'total_data_points': 0,
                    'sources_count': 0,
                    'time_range_hours': lookback_hours
                }
            
            return {
                'symbol_sentiment': aggregated_sentiment,
                'market_sentiment': market_sentiment,
                'raw_sentiment_data': sentiment_data,
                'collection_timestamp': datetime.now(),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Enhanced sentiment analysis failed: {e}")
            return {
                'symbol_sentiment': {},
                'market_sentiment': {},
                'raw_sentiment_data': [],
                'error': str(e),
                'success': False
            }