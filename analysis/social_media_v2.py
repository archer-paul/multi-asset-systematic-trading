"""
Version améliorée du module social media avec gestion intelligente des rate limits
et sources alternatives pour contourner les limitations Twitter
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import tweepy
import asyncpraw
import requests
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class RateLimitInfo:
    remaining: int
    reset_time: int
    limit: int
    
class SocialMediaAnalyzerV2:
    """Version améliorée avec gestion des rate limits et sources alternatives"""
    
    def __init__(self, config):
        self.config = config
        self.twitter_client = None
        self.reddit_client = None
        
        # Rate limiting tracking
        self.twitter_rate_limits = {}
        self.last_twitter_call = 0
        self.twitter_quota_exhausted = False
        
        # Sources alternatives 
        self.alternative_sources = True
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize clients with better error handling"""
        
        # Twitter avec gestion d'erreur améliorée
        enable_twitter = getattr(self.config, 'enable_twitter', True)
        if enable_twitter and self.config.twitter_bearer_token:
            try:
                self.twitter_client = tweepy.Client(
                    bearer_token=self.config.twitter_bearer_token,
                    wait_on_rate_limit=False  # Gestion manuelle
                )
                logger.info("Twitter client initialized with manual rate limiting")
            except Exception as e:
                logger.error(f"Failed to initialize Twitter client: {e}")
                self.twitter_client = None
        
        # Reddit (plus fiable que Twitter) - utiliser asyncpraw pour éviter les warnings
        if self.config.reddit_client_id and self.config.reddit_client_secret:
            try:
                self.reddit_client = asyncpraw.Reddit(
                    client_id=self.config.reddit_client_id,
                    client_secret=self.config.reddit_client_secret,
                    user_agent=self.config.reddit_user_agent
                )
                logger.info("Async Reddit client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Reddit client: {e}")
    
    async def collect_current_sentiment(self, symbols: List[str]) -> Dict[str, Dict]:
        """Version améliorée de la collecte de sentiment"""
        
        current_sentiment = {}
        limited_symbols = symbols[:5]  # Limiter encore plus pour éviter les quotas
        
        for symbol in limited_symbols:
            current_sentiment[symbol] = {
                'sentiment_score': 0.0,
                'confidence': 0.5,
                'mention_count': 0,
                'platforms': [],
                'timestamp': datetime.now(),
                'sources_used': []
            }
            
            # Essayer Twitter avec gestion des quotas
            twitter_success = await self._try_twitter_sentiment(symbol, current_sentiment[symbol])
            
            # Reddit comme source principale (plus fiable)
            reddit_success = await self._collect_reddit_sentiment(symbol, current_sentiment[symbol])
            
            # Sources alternatives si les APIs sont limitées
            if not (twitter_success or reddit_success):
                await self._collect_alternative_sentiment(symbol, current_sentiment[symbol])
        
        logger.info(f"Collected social sentiment for {len(current_sentiment)} symbols")
        return current_sentiment
    
    async def _try_twitter_sentiment(self, symbol: str, sentiment_data: Dict) -> bool:
        """Essaie Twitter avec gestion intelligente des rate limits"""
        
        if self.twitter_quota_exhausted:
            logger.debug(f"Twitter quota exhausted, skipping {symbol}")
            return False
            
        if not self.twitter_client:
            return False
        
        try:
            # Vérifier si on peut faire un appel
            if not self._can_make_twitter_call():
                logger.debug("Twitter rate limit reached, using alternative sources")
                return False
            
            # Query simple pour éviter les opérateurs problématiques
            query = f"{symbol} (stock OR shares OR trading) lang:en -is:retweet"
            
            # Appel avec timeout court
            tweets = await asyncio.wait_for(
                asyncio.to_thread(
                    self.twitter_client.search_recent_tweets,
                    query=query,
                    max_results=10,  # Très limité
                    tweet_fields=['created_at', 'public_metrics']
                ),
                timeout=10.0
            )
            
            if tweets and tweets.data:
                self._update_twitter_rate_limit()
                
                # Calcul simple du sentiment basé sur l'engagement
                total_engagement = sum(
                    tweet.public_metrics.get('like_count', 0) + 
                    tweet.public_metrics.get('retweet_count', 0) * 2
                    for tweet in tweets.data
                )
                
                sentiment_data['mention_count'] += len(tweets.data)
                sentiment_data['platforms'].append('twitter')
                sentiment_data['sources_used'].append('twitter_api')
                
                # Sentiment approximatif basé sur l'engagement
                if total_engagement > 200:
                    sentiment_data['sentiment_score'] = 0.4
                elif total_engagement > 50:
                    sentiment_data['sentiment_score'] = 0.2
                else:
                    sentiment_data['sentiment_score'] = 0.1
                    
                return True
                
        except asyncio.TimeoutError:
            logger.debug(f"Twitter API timeout for {symbol}")
        except tweepy.TooManyRequests:
            logger.warning("Twitter rate limit hit, marking as exhausted")
            self.twitter_quota_exhausted = True
        except Exception as e:
            logger.debug(f"Twitter collection failed for {symbol}: {e}")
        
        return False
    
    def _can_make_twitter_call(self) -> bool:
        """Vérifie si on peut faire un appel Twitter"""
        
        # Limite artificielle: max 1 appel par minute
        now = time.time()
        if now - self.last_twitter_call < 60:
            return False
            
        return True
    
    def _update_twitter_rate_limit(self):
        """Met à jour le tracking des rate limits"""
        self.last_twitter_call = time.time()
    
    async def _collect_reddit_sentiment(self, symbol: str, sentiment_data: Dict) -> bool:
        """Collecte sur Reddit (source plus fiable)"""
        
        if not self.reddit_client:
            return False
        
        try:
            subreddits = ['wallstreetbets', 'investing', 'stocks']
            mention_count = 0
            
            for subreddit_name in subreddits[:2]:  # Limiter à 2 subreddits
                try:
                    subreddit = await self.reddit_client.subreddit(subreddit_name)
                    
                    # Recherche simple avec asyncpraw
                    posts = []
                    async for submission in subreddit.search(symbol, limit=5, time_filter='day'):
                        posts.append(submission)
                    
                    mention_count += len(posts)
                    
                    # Analyse simple du sentiment basé sur les scores
                    total_score = sum(post.score for post in posts if hasattr(post, 'score'))
                    if posts and total_score > 0:
                        avg_score = total_score / len(posts)
                        if avg_score > 50:
                            sentiment_data['sentiment_score'] = max(sentiment_data['sentiment_score'], 0.3)
                        elif avg_score > 10:
                            sentiment_data['sentiment_score'] = max(sentiment_data['sentiment_score'], 0.1)
                    
                except Exception as e:
                    logger.debug(f"Reddit subreddit {subreddit_name} error: {e}")
                    continue
            
            if mention_count > 0:
                sentiment_data['mention_count'] += mention_count
                sentiment_data['platforms'].append('reddit')
                sentiment_data['sources_used'].append('reddit_api')
                return True
                
        except Exception as e:
            logger.debug(f"Reddit collection failed for {symbol}: {e}")
        
        return False
    
    async def _collect_alternative_sentiment(self, symbol: str, sentiment_data: Dict):
        """Sources alternatives quand les APIs sont limitées"""
        
        # 1. Finviz (gratuit, pas d'API key needed)
        finviz_success = await self._try_finviz_news(symbol, sentiment_data)
        
        # 2. Yahoo Finance discussions (scraping léger)
        yahoo_success = await self._try_yahoo_sentiment(symbol, sentiment_data)
        
        if finviz_success or yahoo_success:
            sentiment_data['sources_used'].append('alternative_sources')
    
    async def _try_finviz_news(self, symbol: str, sentiment_data: Dict) -> bool:
        """Essaie de récupérer le sentiment via Finviz"""
        
        try:
            # Finviz URL pour les news d'un symbole
            url = f"https://finviz.com/quote.ashx?t={symbol}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = await asyncio.to_thread(
                requests.get, url, headers=headers, timeout=5
            )
            
            if response.status_code == 200:
                # Analyse basique: si la page se charge, l'action existe et est tradée
                sentiment_data['mention_count'] += 1
                sentiment_data['platforms'].append('finviz')
                sentiment_data['confidence'] = max(sentiment_data['confidence'], 0.6)
                return True
                
        except Exception as e:
            logger.debug(f"Finviz lookup failed for {symbol}: {e}")
        
        return False
    
    async def _try_yahoo_sentiment(self, symbol: str, sentiment_data: Dict) -> bool:
        """Sentiment basique via Yahoo Finance"""
        
        try:
            # Yahoo Finance conversations (endpoint public)
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={symbol}"
            
            response = await asyncio.to_thread(
                requests.get, url, timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                quotes = data.get('quotes', [])
                
                if quotes:
                    # Si trouvé sur Yahoo Finance, c'est une action légitime
                    sentiment_data['mention_count'] += 1
                    sentiment_data['platforms'].append('yahoo_finance')
                    
                    # Sentiment neutre mais existence confirmée
                    sentiment_data['confidence'] = max(sentiment_data['confidence'], 0.7)
                    return True
                    
        except Exception as e:
            logger.debug(f"Yahoo Finance lookup failed for {symbol}: {e}")
        
        return False
    
    async def collect_historical_sentiment(self, symbols: List[str], days: int = 7) -> List[Dict]:
        """Version optimisée pour l'historique"""
        
        logger.info(f"Collecting historical sentiment with rate limit management")
        
        sentiment_data = []
        
        # Encore plus conservateur pour l'historique
        limited_symbols = symbols[:3]
        
        for symbol in limited_symbols:
            try:
                # Reddit uniquement pour l'historique (plus fiable)
                reddit_data = await self._collect_reddit_historical(symbol, days)
                sentiment_data.extend(reddit_data)
                
                # Pause entre chaque symbole
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Historical sentiment collection failed for {symbol}: {e}")
        
        logger.info(f"Collected {len(sentiment_data)} historical sentiment points")
        return sentiment_data
    
    async def _collect_reddit_historical(self, symbol: str, days: int) -> List[Dict]:
        """Collecte historique Reddit"""
        
        if not self.reddit_client:
            return []
        
        historical_data = []
        
        try:
            subreddit = await self.reddit_client.subreddit('wallstreetbets')
            
            # Posts récents mentionnant le symbole avec asyncpraw
            posts = []
            async for submission in subreddit.search(symbol, limit=10, time_filter='week'):
                posts.append(submission)
            
            for post in posts:
                historical_data.append({
                    'symbol': symbol,
                    'platform': 'reddit',
                    'content': f"{post.title} {post.selftext}"[:200],
                    'score': getattr(post, 'score', 0),
                    'created_at': datetime.fromtimestamp(post.created_utc),
                    'engagement': getattr(post, 'num_comments', 0)
                })
        
        except Exception as e:
            logger.debug(f"Reddit historical collection failed for {symbol}: {e}")
        
        return historical_data
    
    def get_quota_status(self) -> Dict:
        """Retourne le statut des quotas API"""
        
        return {
            'twitter_available': not self.twitter_quota_exhausted and self.twitter_client is not None,
            'twitter_exhausted': self.twitter_quota_exhausted,
            'reddit_available': self.reddit_client is not None,
            'alternative_sources': self.alternative_sources,
            'last_twitter_call': self.last_twitter_call,
            'recommendations': self._get_usage_recommendations()
        }
    
    def _get_usage_recommendations(self) -> List[str]:
        """Recommandations pour optimiser l'usage"""
        
        recommendations = []
        
        if self.twitter_quota_exhausted:
            recommendations.append("Twitter quota épuisé - utilisez Reddit et sources alternatives")
        
        if not self.twitter_client:
            recommendations.append("Configurez Twitter API v2 Bearer Token pour plus de données")
        
        if not self.reddit_client:
            recommendations.append("Configurez Reddit API pour des données sociales fiables")
        
        recommendations.append("Mode économique activé - collecte réduite mais stable")
        
        return recommendations