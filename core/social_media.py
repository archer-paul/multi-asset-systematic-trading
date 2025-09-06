"""
Social Media Sentiment Analyzer
Modular component for analyzing sentiment from Twitter/X and Reddit
"""

import logging
import asyncio
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Optional imports - gracefully handle missing dependencies
try:
    import tweepy
    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False
    tweepy = None

try:
    import praw
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False
    praw = None

from core.config import Config

class SocialMediaAnalyzer:
    """
    Modular social media sentiment analyzer
    Supports Twitter/X and Reddit with extensible architecture
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize API clients
        self.twitter_client = None
        self.reddit_client = None
        
        self._initialize_clients()
        
        # Cache for avoiding duplicate processing
        self.processed_tweets = set()
        self.processed_posts = set()
        
        # Rate limiting
        self.last_twitter_request = None
        self.last_reddit_request = None
        self.twitter_request_count = 0
        self.reddit_request_count = 0
    
    def _initialize_clients(self):
        """Initialize social media API clients"""
        
        # Initialize Twitter client
        if (TWITTER_AVAILABLE and 
            self.config.is_api_key_available('twitter') and 
            self.config.SOCIAL_SOURCES.get('twitter', False)):
            
            try:
                self.twitter_client = tweepy.Client(
                    bearer_token=self.config.TWITTER_BEARER_TOKEN,
                    wait_on_rate_limit=True
                )
                self.logger.info("Twitter client initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Twitter client: {e}")
        else:
            self.logger.info("Twitter client not initialized (API key missing or disabled)")
        
        # Initialize Reddit client
        if (REDDIT_AVAILABLE and 
            self.config.is_api_key_available('reddit') and 
            self.config.SOCIAL_SOURCES.get('reddit', False)):
            
            try:
                self.reddit_client = praw.Reddit(
                    client_id=self.config.REDDIT_CLIENT_ID,
                    client_secret=self.config.REDDIT_CLIENT_SECRET,
                    user_agent=self.config.REDDIT_USER_AGENT
                )
                self.logger.info("Reddit client initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Reddit client: {e}")
        else:
            self.logger.info("Reddit client not initialized (API key missing or disabled)")
    
    async def collect_current_sentiment(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Collect current social media sentiment for given symbols
        
        Args:
            symbols: List of stock symbols to analyze
            
        Returns:
            Dict mapping symbols to sentiment data
        """
        sentiment_data = {}
        
        for symbol in symbols:
            try:
                symbol_sentiment = await self._collect_symbol_sentiment(symbol)
                if symbol_sentiment:
                    sentiment_data[symbol] = symbol_sentiment
                    
            except Exception as e:
                self.logger.error(f"Error collecting sentiment for {symbol}: {e}")
                sentiment_data[symbol] = self._get_default_sentiment()
        
        return sentiment_data
    
    async def collect_historical_sentiment(self, symbols: List[str], days: int = 7) -> List[Dict]:
        """
        Collect historical social media sentiment
        
        Args:
            symbols: List of stock symbols
            days: Number of days to look back
            
        Returns:
            List of historical sentiment data
        """
        historical_data = []
        
        for symbol in symbols:
            try:
                symbol_history = await self._collect_symbol_historical_sentiment(symbol, days)
                historical_data.extend(symbol_history)
                
            except Exception as e:
                self.logger.error(f"Error collecting historical sentiment for {symbol}: {e}")
        
        return historical_data
    
    async def _collect_symbol_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Collect current sentiment for a specific symbol"""
        
        # Collect from all enabled sources
        twitter_sentiment = {}
        reddit_sentiment = {}
        
        if self.twitter_client:
            twitter_sentiment = await self._collect_twitter_sentiment(symbol)
        
        if self.reddit_client:
            reddit_sentiment = await self._collect_reddit_sentiment(symbol)
        
        # Combine sentiments
        combined_sentiment = self._combine_social_sentiments(
            twitter_sentiment, reddit_sentiment
        )
        
        return combined_sentiment
    
    async def _collect_twitter_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Collect sentiment from Twitter/X"""
        
        if not self.twitter_client:
            return {}
        
        try:
            # Rate limiting check
            await self._check_twitter_rate_limit()
            
            # Create search query
            queries = self._create_twitter_queries(symbol)
            
            tweets_data = []
            for query in queries:
                try:
                    # Search recent tweets
                    tweets = tweepy.Paginator(
                        self.twitter_client.search_recent_tweets,
                        query=query,
                        tweet_fields=['created_at', 'public_metrics', 'context_annotations'],
                        max_results=100
                    ).flatten(limit=200)  # Limit to 200 tweets per query
                    
                    for tweet in tweets:
                        if tweet.id not in self.processed_tweets:
                            tweet_data = self._process_tweet(tweet, symbol)
                            if tweet_data:
                                tweets_data.append(tweet_data)
                                self.processed_tweets.add(tweet.id)
                    
                    self.twitter_request_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error searching Twitter for query '{query}': {e}")
            
            # Analyze sentiment
            if tweets_data:
                sentiment = await self._analyze_twitter_sentiment(tweets_data)
                return sentiment
            
        except Exception as e:
            self.logger.error(f"Error collecting Twitter sentiment for {symbol}: {e}")
        
        return {}
    
    async def _collect_reddit_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Collect sentiment from Reddit"""
        
        if not self.reddit_client:
            return {}
        
        try:
            # Rate limiting check
            await self._check_reddit_rate_limit()
            
            posts_data = []
            
            # Search in relevant subreddits
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'SecurityAnalysis']
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)
                    
                    # Search for posts mentioning the symbol
                    search_query = f"{symbol} OR ${symbol}"
                    
                    # Get recent posts
                    for post in subreddit.search(search_query, time_filter='day', limit=50):
                        if post.id not in self.processed_posts:
                            post_data = self._process_reddit_post(post, symbol)
                            if post_data:
                                posts_data.append(post_data)
                                self.processed_posts.add(post.id)
                    
                    self.reddit_request_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error searching Reddit in r/{subreddit_name}: {e}")
            
            # Analyze sentiment
            if posts_data:
                sentiment = await self._analyze_reddit_sentiment(posts_data)
                return sentiment
            
        except Exception as e:
            self.logger.error(f"Error collecting Reddit sentiment for {symbol}: {e}")
        
        return {}
    
    def _create_twitter_queries(self, symbol: str) -> List[str]:
        """Create Twitter search queries for a symbol"""
        
        # Clean symbol (remove exchange suffix)
        clean_symbol = symbol.split('.')[0]
        
        queries = [
            f"${clean_symbol} lang:en -is:retweet",
            f"{clean_symbol} stock lang:en -is:retweet",
            f"{clean_symbol} shares lang:en -is:retweet"
        ]
        
        # Add company-specific terms for major stocks
        company_terms = {
            'AAPL': 'Apple',
            'GOOGL': 'Google Alphabet',
            'MSFT': 'Microsoft',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'META': 'Meta Facebook',
            'NVDA': 'Nvidia',
            'ASML': 'ASML'
        }
        
        if clean_symbol in company_terms:
            company_name = company_terms[clean_symbol]
            queries.append(f"{company_name} stock lang:en -is:retweet")
        
        return queries
    
    def _process_tweet(self, tweet, symbol: str) -> Optional[Dict]:
        """Process a single tweet"""
        
        try:
            # Basic filtering
            if not tweet.text or len(tweet.text) < 10:
                return None
            
            # Filter out obvious spam/bot tweets
            if self._is_spam_tweet(tweet.text):
                return None
            
            return {
                'id': tweet.id,
                'text': tweet.text,
                'created_at': tweet.created_at,
                'symbol': symbol,
                'source': 'twitter',
                'metrics': {
                    'retweet_count': getattr(tweet.public_metrics, 'retweet_count', 0),
                    'like_count': getattr(tweet.public_metrics, 'like_count', 0),
                    'reply_count': getattr(tweet.public_metrics, 'reply_count', 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing tweet: {e}")
            return None
    
    def _process_reddit_post(self, post, symbol: str) -> Optional[Dict]:
        """Process a single Reddit post"""
        
        try:
            # Combine title and selftext
            text = post.title
            if hasattr(post, 'selftext') and post.selftext:
                text += " " + post.selftext
            
            if len(text) < 10:
                return None
            
            # Filter out deleted/removed posts
            if '[deleted]' in text or '[removed]' in text:
                return None
            
            return {
                'id': post.id,
                'text': text,
                'created_at': datetime.fromtimestamp(post.created_utc),
                'symbol': symbol,
                'source': 'reddit',
                'subreddit': post.subreddit.display_name,
                'metrics': {
                    'score': post.score,
                    'upvote_ratio': getattr(post, 'upvote_ratio', 0.5),
                    'num_comments': post.num_comments
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing Reddit post: {e}")
            return None
    
    def _is_spam_tweet(self, text: str) -> bool:
        """Check if tweet appears to be spam"""
        
        spam_indicators = [
            len(re.findall(r'[!@#$%^&*]', text)) > 5,  # Too many special chars
            len(re.findall(r'http[s]?://', text)) > 2,  # Too many links
            len(re.findall(r'#\w+', text)) > 5,        # Too many hashtags
            'crypto' in text.lower() and 'pump' in text.lower(),  # Crypto pump
            'buy now' in text.lower(),
            'guaranteed' in text.lower(),
            'dm me' in text.lower()
        ]
        
        return sum(spam_indicators) >= 2
    
    async def _analyze_twitter_sentiment(self, tweets_data: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment from Twitter data"""
        
        if not tweets_data:
            return self._get_default_sentiment()
        
        # Simple sentiment analysis using keyword matching
        # In production, you might want to use a more sophisticated model
        sentiment_scores = []
        engagement_weights = []
        
        for tweet in tweets_data:
            score = self._calculate_text_sentiment(tweet['text'])
            
            # Weight by engagement (likes + retweets)
            engagement = (
                tweet['metrics'].get('like_count', 0) + 
                tweet['metrics'].get('retweet_count', 0) * 2  # Retweets weighted more
            )
            weight = max(1, engagement)  # Minimum weight of 1
            
            sentiment_scores.append(score)
            engagement_weights.append(weight)
        
        # Calculate weighted average
        weighted_sentiment = np.average(sentiment_scores, weights=engagement_weights)
        
        return {
            'sentiment_score': float(weighted_sentiment),
            'confidence': min(0.8, len(tweets_data) / 100),  # Higher confidence with more data
            'sample_size': len(tweets_data),
            'source': 'twitter',
            'engagement_total': sum(engagement_weights),
            'timestamp': datetime.now()
        }
    
    async def _analyze_reddit_sentiment(self, posts_data: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment from Reddit data"""
        
        if not posts_data:
            return self._get_default_sentiment()
        
        sentiment_scores = []
        score_weights = []
        
        for post in posts_data:
            sentiment = self._calculate_text_sentiment(post['text'])
            
            # Weight by Reddit score (upvotes - downvotes)
            reddit_score = post['metrics'].get('score', 1)
            weight = max(1, reddit_score)  # Minimum weight of 1
            
            sentiment_scores.append(sentiment)
            score_weights.append(weight)
        
        # Calculate weighted average
        weighted_sentiment = np.average(sentiment_scores, weights=score_weights)
        
        return {
            'sentiment_score': float(weighted_sentiment),
            'confidence': min(0.7, len(posts_data) / 50),  # Reddit generally has less volume
            'sample_size': len(posts_data),
            'source': 'reddit',
            'total_score': sum(score_weights),
            'timestamp': datetime.now()
        }
    
    def _calculate_text_sentiment(self, text: str) -> float:
        """
        Simple sentiment calculation using keyword matching
        Returns value between -1 (negative) and 1 (positive)
        """
        
        # Convert to lowercase for matching
        text_lower = text.lower()
        
        # Positive keywords
        positive_words = [
            'bullish', 'buy', 'bull', 'moon', 'rocket', 'gain', 'profit', 'up', 'rise',
            'strong', 'good', 'great', 'excellent', 'positive', 'growth', 'increase',
            'breakout', 'rally', 'surge', 'pump', 'hold', 'diamond hands'
        ]
        
        # Negative keywords
        negative_words = [
            'bearish', 'sell', 'bear', 'crash', 'drop', 'fall', 'loss', 'down', 'decline',
            'weak', 'bad', 'terrible', 'negative', 'dump', 'short', 'puts', 'puts',
            'recession', 'bubble', 'overvalued', 'paper hands'
        ]
        
        # Count occurrences
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate sentiment score
        if positive_count == 0 and negative_count == 0:
            return 0.0  # Neutral
        
        total_sentiment_words = positive_count + negative_count
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
        
        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, sentiment_score))
    
    def _combine_social_sentiments(self, twitter_sentiment: Dict, reddit_sentiment: Dict) -> Dict[str, Any]:
        """Combine sentiment from different social media sources"""
        
        if not twitter_sentiment and not reddit_sentiment:
            return self._get_default_sentiment()
        
        if not twitter_sentiment:
            return reddit_sentiment
        
        if not reddit_sentiment:
            return twitter_sentiment
        
        # Combine both sources with weights
        twitter_weight = 0.6  # Twitter often has more real-time sentiment
        reddit_weight = 0.4   # Reddit has more detailed discussions
        
        # Weight by confidence and sample size
        twitter_conf = twitter_sentiment.get('confidence', 0.5)
        reddit_conf = reddit_sentiment.get('confidence', 0.5)
        
        twitter_samples = twitter_sentiment.get('sample_size', 0)
        reddit_samples = reddit_sentiment.get('sample_size', 0)
        
        # Adjust weights based on data quality
        if twitter_samples == 0:
            twitter_weight = 0
            reddit_weight = 1
        elif reddit_samples == 0:
            twitter_weight = 1
            reddit_weight = 0
        else:
            # Normalize weights
            total_weight = twitter_weight + reddit_weight
            twitter_weight /= total_weight
            reddit_weight /= total_weight
        
        # Calculate combined sentiment
        combined_sentiment = (
            twitter_weight * twitter_sentiment.get('sentiment_score', 0) +
            reddit_weight * reddit_sentiment.get('sentiment_score', 0)
        )
        
        combined_confidence = (
            twitter_weight * twitter_conf +
            reddit_weight * reddit_conf
        )
        
        return {
            'sentiment_score': float(combined_sentiment),
            'confidence': float(combined_confidence),
            'sample_size': twitter_samples + reddit_samples,
            'source': 'combined_social',
            'twitter_component': twitter_sentiment,
            'reddit_component': reddit_sentiment,
            'weights': {
                'twitter': twitter_weight,
                'reddit': reddit_weight
            },
            'timestamp': datetime.now()
        }
    
    def _get_default_sentiment(self) -> Dict[str, Any]:
        """Return default neutral sentiment when no data is available"""
        return {
            'sentiment_score': 0.0,
            'confidence': 0.1,  # Low confidence for no data
            'sample_size': 0,
            'source': 'default',
            'timestamp': datetime.now()
        }
    
    async def _check_twitter_rate_limit(self):
        """Check and enforce Twitter rate limiting"""
        if not self.twitter_client:
            return
        
        now = datetime.now()
        
        # Twitter allows 300 requests per 15 minutes for search
        if self.last_twitter_request:
            time_diff = (now - self.last_twitter_request).total_seconds()
            
            # Reset counter every 15 minutes
            if time_diff > 900:  # 15 minutes
                self.twitter_request_count = 0
            
            # If we're near the limit, wait
            if self.twitter_request_count >= 290:  # Leave some buffer
                wait_time = 900 - time_diff
                if wait_time > 0:
                    self.logger.info(f"Twitter rate limit reached, waiting {wait_time:.1f} seconds")
                    await asyncio.sleep(wait_time)
                    self.twitter_request_count = 0
        
        self.last_twitter_request = now
    
    async def _check_reddit_rate_limit(self):
        """Check and enforce Reddit rate limiting"""
        if not self.reddit_client:
            return
        
        now = datetime.now()
        
        # Reddit allows 60 requests per minute
        if self.last_reddit_request:
            time_diff = (now - self.last_reddit_request).total_seconds()
            
            # Reset counter every minute
            if time_diff > 60:
                self.reddit_request_count = 0
            
            # If we're near the limit, wait
            if self.reddit_request_count >= 55:  # Leave some buffer
                wait_time = 60 - time_diff
                if wait_time > 0:
                    self.logger.info(f"Reddit rate limit reached, waiting {wait_time:.1f} seconds")
                    await asyncio.sleep(wait_time)
                    self.reddit_request_count = 0
        
        self.last_reddit_request = now
    
    async def _collect_symbol_historical_sentiment(self, symbol: str, days: int) -> List[Dict]:
        """Collect historical sentiment data for a symbol"""
        
        # For historical data, we would typically query a database
        # Since we're collecting real-time data, we simulate historical data
        # In a production system, you would store and retrieve actual historical data
        
        historical_data = []
        
        # Simulate some historical sentiment data points
        # This would be replaced with actual database queries
        for day_offset in range(days):
            date = datetime.now() - timedelta(days=day_offset)
            
            # Simulate sentiment data (in production, this would come from stored data)
            simulated_sentiment = {
                'symbol': symbol,
                'date': date,
                'sentiment_score': np.random.normal(0, 0.3),  # Random sentiment around neutral
                'confidence': np.random.uniform(0.3, 0.8),
                'sample_size': np.random.randint(10, 100),
                'source': 'historical_social',
                'timestamp': date
            }
            
            historical_data.append(simulated_sentiment)
        
        return historical_data
    
    def get_source_availability(self) -> Dict[str, bool]:
        """Get availability status of different social media sources"""
        return {
            'twitter': self.twitter_client is not None,
            'reddit': self.reddit_client is not None,
            'twitter_api_available': TWITTER_AVAILABLE,
            'reddit_api_available': REDDIT_AVAILABLE
        }
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        now = datetime.now()
        
        twitter_status = {}
        if self.last_twitter_request:
            time_since_last = (now - self.last_twitter_request).total_seconds()
            twitter_status = {
                'requests_made': self.twitter_request_count,
                'time_since_last_request': time_since_last,
                'requests_remaining': max(0, 290 - self.twitter_request_count)
            }
        
        reddit_status = {}
        if self.last_reddit_request:
            time_since_last = (now - self.last_reddit_request).total_seconds()
            reddit_status = {
                'requests_made': self.reddit_request_count,
                'time_since_last_request': time_since_last,
                'requests_remaining': max(0, 55 - self.reddit_request_count)
            }
        
        return {
            'twitter': twitter_status,
            'reddit': reddit_status,
            'timestamp': now
        }
    
    async def test_connections(self) -> Dict[str, bool]:
        """Test connections to social media APIs"""
        results = {}
        
        # Test Twitter connection
        if self.twitter_client:
            try:
                # Try a simple search to test the connection
                test_tweets = self.twitter_client.search_recent_tweets(
                    query="test", max_results=10
                )
                results['twitter'] = True
                self.logger.info("Twitter connection test successful")
            except Exception as e:
                results['twitter'] = False
                self.logger.error(f"Twitter connection test failed: {e}")
        else:
            results['twitter'] = False
        
        # Test Reddit connection
        if self.reddit_client:
            try:
                # Try to access a subreddit to test the connection
                test_subreddit = self.reddit_client.subreddit('test')
                test_subreddit.display_name  # This will trigger an API call
                results['reddit'] = True
                self.logger.info("Reddit connection test successful")
            except Exception as e:
                results['reddit'] = False
                self.logger.error(f"Reddit connection test failed: {e}")
        else:
            results['reddit'] = False
        
        return results

class SocialSentimentConfig:
    """Configuration specifically for social media sentiment analysis"""
    
    # Sentiment keywords - can be customized per use case
    POSITIVE_KEYWORDS = [
        'bullish', 'buy', 'bull', 'moon', 'rocket', 'gain', 'profit', 'up', 'rise',
        'strong', 'good', 'great', 'excellent', 'positive', 'growth', 'increase',
        'breakout', 'rally', 'surge', 'pump', 'hold', 'diamond hands', 'to the moon',
        'hodl', 'long', 'calls', 'yolo', 'stonks', 'green', 'tendies'
    ]
    
    NEGATIVE_KEYWORDS = [
        'bearish', 'sell', 'bear', 'crash', 'drop', 'fall', 'loss', 'down', 'decline',
        'weak', 'bad', 'terrible', 'negative', 'dump', 'short', 'puts', 'red',
        'recession', 'bubble', 'overvalued', 'paper hands', 'rug pull', 'drill'
    ]
    
    # Twitter-specific configuration
    TWITTER_CONFIG = {
        'max_tweets_per_query': 200,
        'max_queries_per_symbol': 4,
        'min_tweet_length': 10,
        'exclude_retweets': True,
        'language': 'en'
    }
    
    # Reddit-specific configuration
    REDDIT_CONFIG = {
        'subreddits': ['wallstreetbets', 'stocks', 'investing', 'SecurityAnalysis', 'StockMarket'],
        'max_posts_per_subreddit': 50,
        'min_post_score': 1,
        'time_filter': 'day'  # 'hour', 'day', 'week', 'month', 'year', 'all'
    }
    
    # Weighting configuration
    WEIGHTS = {
        'twitter': 0.6,
        'reddit': 0.4,
        'engagement_multiplier': 1.5,  # How much to weight high-engagement content
        'recency_decay': 0.1  # How much weight decreases per hour for older content
    }

# Example usage and testing functions
if __name__ == "__main__":
    """
    Example usage of the SocialMediaAnalyzer
    """
    import asyncio
    from core.config import Config
    
    async def test_social_analyzer():
        config = Config()
        config.ENABLE_SOCIAL_SENTIMENT = True
        
        analyzer = SocialMediaAnalyzer(config)
        
        # Test API connections
        print("Testing API connections...")
        connections = await analyzer.test_connections()
        print(f"Connection status: {connections}")
        
        # Test sentiment collection for a symbol
        if any(connections.values()):
            print("\nCollecting sentiment for AAPL...")
            sentiment = await analyzer.collect_current_sentiment(['AAPL'])
            print(f"Sentiment data: {sentiment}")
        
        # Check rate limit status
        rate_limits = analyzer.get_rate_limit_status()
        print(f"\nRate limit status: {rate_limits}")
    
    # Uncomment to run the test
    # asyncio.run(test_social_analyzer())