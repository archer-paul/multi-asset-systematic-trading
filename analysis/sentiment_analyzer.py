"""
Sentiment analysis module for Trading Bot
Analyzes sentiment from news and social media data
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import google.generativeai as genai
from core.config import config

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analyzes sentiment from various text sources"""
    
    def __init__(self, config=None):
        self.config = config
        self.gemini_client = None
        self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Gemini AI client"""
        try:
            if not self.config:
                logger.warning("No configuration provided for Gemini AI")
                return

            if not hasattr(self.config, 'gemini_api_key'):
                logger.warning("GEMINI_API_KEY not found in configuration")
                return

            if not self.config.gemini_api_key:
                logger.warning("GEMINI_API_KEY is empty - sentiment analysis will use fallback method")
                return

            # Log partial API key for verification (first 8 chars + ****)
            api_key_preview = f"{self.config.gemini_api_key[:8]}****" if len(self.config.gemini_api_key) > 8 else "****"
            logger.info(f"Initializing Gemini AI with API key: {api_key_preview}")

            genai.configure(api_key=self.config.gemini_api_key)
            self.gemini_client = genai.GenerativeModel('gemini-1.5-flash')

            logger.info("Gemini AI client initialized successfully")
            logger.info("Sentiment analysis will use Gemini AI for enhanced accuracy")

            # Test basic functionality during initialization
            test_response = self.gemini_client.generate_content("Test: respond with 'OK'")
            if test_response and test_response.text:
                logger.info(f"Gemini AI test successful: {test_response.text.strip()}")
            else:
                logger.warning("Gemini AI test failed - may have connectivity issues")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini AI: {e}")
            logger.warning("Sentiment analysis will use keyword-based fallback method")
            self.gemini_client = None
    
    async def initialize(self):
        """Initialize sentiment analyzer"""
        try:
            logger.info("SentimentAnalyzer initialized successfully")
            return True
        except Exception as e:
            logger.error(f"SentimentAnalyzer initialization failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            logger.info("SentimentAnalyzer cleanup completed")
        except Exception as e:
            logger.error(f"SentimentAnalyzer cleanup failed: {e}")
    
    async def analyze_financial_sentiment(self, text: str, symbols: List[str] = None,
                                         company: str = None, region: str = None) -> Dict[str, Any]:
        """Analyze financial sentiment from text with focus on specific symbols, company and region"""
        start_time = datetime.now()

        try:
            if not self.gemini_client:
                logger.warning(f"Gemini client not available for {company or 'general'} - using FALLBACK sentiment analysis")
                result = self._fallback_sentiment_analysis(text, symbols, company, region)
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"FALLBACK sentiment analysis completed in {elapsed:.2f}s for {company or 'general'}")
                return result

            logger.info(f"Starting GEMINI AI sentiment analysis for: {company or 'general'} - Text length: {len(text)} chars")
            
            # Create comprehensive financial sentiment prompt
            focus_text = ""
            if company:
                focus_text += f" focusing on company {company}"
            if symbols:
                focus_text += f" and symbols {', '.join(symbols)}"
            if region:
                focus_text += f" in {region} market"
                
            prompt = f"""
            Analyze the financial sentiment of the following text{focus_text}.
            Provide a comprehensive financial analysis with the following metrics:
            
            Text: {text}
            
            Format your response as:
            Sentiment Score: [number between -1 and 1]
            Confidence: [number between 0 and 1]
            Market Impact: [number between 0 and 1]
            Urgency: [number between 0 and 1]
            Key Themes: [comma-separated list]
            Risk Factors: [comma-separated list]
            Timeframe: [short-term, medium-term, or long-term]
            Sector Impact: [number between 0 and 1]
            Reasoning: [brief explanation]
            """
            
            logger.debug(f"Sending prompt to Gemini AI: {prompt[:200]}...")

            response = await asyncio.to_thread(
                self.gemini_client.generate_content,
                prompt
            )

            # Parse response
            response_text = response.text
            logger.info(f"Gemini AI response received for {company or 'general'}: {len(response_text)} chars")
            logger.debug(f"Full Gemini response: {response_text[:500]}...")

            sentiment_score = 0.0
            confidence = 0.5
            market_impact = 0.5
            urgency = 0.5
            key_themes = []
            risk_factors = []
            timeframe = 'short-term'
            sector_impact = 0.5
            reasoning = "Unable to parse response"
            
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('Sentiment Score:'):
                    try:
                        sentiment_score = float(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('Confidence:'):
                    try:
                        confidence = float(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('Market Impact:'):
                    try:
                        market_impact = float(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('Urgency:'):
                    try:
                        urgency = float(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('Key Themes:'):
                    try:
                        themes_text = line.split(':', 1)[1].strip()
                        key_themes = [t.strip() for t in themes_text.split(',') if t.strip()]
                    except:
                        pass
                elif line.startswith('Risk Factors:'):
                    try:
                        risks_text = line.split(':', 1)[1].strip()
                        risk_factors = [r.strip() for r in risks_text.split(',') if r.strip()]
                    except:
                        pass
                elif line.startswith('Timeframe:'):
                    try:
                        timeframe = line.split(':')[1].strip()
                    except:
                        pass
                elif line.startswith('Sector Impact:'):
                    try:
                        sector_impact = float(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('Reasoning:'):
                    reasoning = line.split(':', 1)[1].strip()
            
            return {
                'sentiment_score': max(-1, min(1, sentiment_score)),
                'confidence': max(0, min(1, confidence)),
                'market_impact': max(0, min(1, market_impact)),
                'urgency': max(0, min(1, urgency)),
                'key_themes': key_themes,
                'risk_factors': risk_factors,
                'timeframe': timeframe,
                'sector_impact': max(0, min(1, sector_impact)),
                'reasoning': reasoning
            }

            logger.info(f"Sentiment analysis for {company or 'general'}: Score={sentiment_score:.2f}, Confidence={confidence:.2f}")
            return result

        except Exception as e:
            logger.error(f"Gemini sentiment analysis failed for {company or 'general'}: {str(e)}")
            logger.warning("Falling back to basic sentiment analysis")
            return self._fallback_sentiment_analysis(text, symbols, company, region)

    def _fallback_sentiment_analysis(self, text: str, symbols: List[str] = None,
                                   company: str = None, region: str = None) -> Dict[str, Any]:
        """Fallback sentiment analysis when Gemini is not available"""
        try:
            # Simple keyword-based sentiment analysis
            positive_words = ['positive', 'good', 'strong', 'growth', 'increase', 'profit', 'gain', 'buy', 'bullish', 'up', 'rise', 'high', 'success', 'beat', 'exceed', 'outperform']
            negative_words = ['negative', 'bad', 'weak', 'decline', 'decrease', 'loss', 'sell', 'bearish', 'down', 'fall', 'low', 'failure', 'miss', 'underperform', 'concern', 'risk']

            text_lower = text.lower()

            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)

            total_words = len(text.split())
            sentiment_strength = (positive_count - negative_count) / max(total_words, 1)

            # Normalize to -1 to 1 range
            sentiment_score = max(-1, min(1, sentiment_strength * 10))

            # Calculate confidence based on sentiment word density
            confidence = min(0.8, (positive_count + negative_count) / max(total_words, 1) * 20)
            confidence = max(0.3, confidence)  # Minimum confidence

            # Determine themes based on keywords
            key_themes = []
            if 'earnings' in text_lower or 'profit' in text_lower:
                key_themes.append('Earnings')
            if 'revenue' in text_lower or 'sales' in text_lower:
                key_themes.append('Revenue')
            if 'growth' in text_lower:
                key_themes.append('Growth')
            if 'market' in text_lower:
                key_themes.append('Market')

            # Basic risk factors
            risk_factors = []
            if negative_count > positive_count:
                risk_factors.append('Negative sentiment')
            if 'risk' in text_lower or 'concern' in text_lower:
                risk_factors.append('Risk concerns')

            logger.debug(f"Fallback sentiment analysis: {sentiment_score:.2f} (pos:{positive_count}, neg:{negative_count})")

            return {
                'sentiment_score': sentiment_score,
                'confidence': confidence,
                'market_impact': abs(sentiment_score) * confidence,
                'urgency': 0.5,
                'key_themes': key_themes if key_themes else ['General Market'],
                'risk_factors': risk_factors if risk_factors else ['Standard Market Risk'],
                'timeframe': 'short-term',
                'sector_impact': abs(sentiment_score) * 0.7,
                'reasoning': f'Keyword-based analysis: {positive_count} positive, {negative_count} negative terms'
            }

        except Exception as e:
            logger.error(f"Fallback sentiment analysis failed: {e}")
            return {
                'sentiment_score': 0.0,
                'confidence': 0.3,
                'market_impact': 0.5,
                'urgency': 0.5,
                'key_themes': ['Unknown'],
                'risk_factors': ['Analysis Error'],
                'timeframe': 'short-term',
                'sector_impact': 0.5,
                'reasoning': f'Fallback analysis failed: {str(e)}'
            }

    async def analyze_text(self, text: str) -> float:
        """Simple text sentiment analysis returning just a score"""
        try:
            result = await self.analyze_financial_sentiment(text)
            return result.get('sentiment_score', 0.0)
        except Exception as e:
            logger.error(f"Text sentiment analysis failed: {e}")
            return 0.0
    
    async def analyze_news_sentiment(self, news_data: List[Dict]) -> List[Dict]:
        """Analyze sentiment of news articles"""
        analyzed_news = []
        
        for article in news_data:
            try:
                sentiment = await self._analyze_text_sentiment(
                    article.get('title', '') + ' ' + article.get('content', '')
                )
                
                article_with_sentiment = article.copy()
                article_with_sentiment.update(sentiment)
                analyzed_news.append(article_with_sentiment)
                
            except Exception as e:
                logger.error(f"Failed to analyze sentiment for article: {e}")
                # Add neutral sentiment as fallback
                article_with_sentiment = article.copy()
                article_with_sentiment.update({
                    'sentiment_score': 0.0,
                    'sentiment_label': 'neutral',
                    'confidence': 0.0
                })
                analyzed_news.append(article_with_sentiment)
        
        return analyzed_news
    
    async def _analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a given text"""
        if not text.strip():
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0
            }
        
        # Use Gemini AI for sentiment analysis
        if self.gemini_client:
            return await self._analyze_with_gemini(text)
        else:
            # Fallback to simple keyword-based analysis
            return self._analyze_with_keywords(text)
    
    async def _analyze_with_gemini(self, text: str) -> Dict[str, Any]:
        """Use Gemini AI for sentiment analysis"""
        try:
            prompt = f"""
            Analyze the sentiment of the following text for financial/trading context.
            Return ONLY a JSON response with exactly these fields:
            - sentiment_score: float between -1.0 (very negative) and 1.0 (very positive)
            - sentiment_label: one of "positive", "negative", or "neutral"
            - confidence: float between 0.0 and 1.0
            
            Text to analyze: {text[:500]}
            """
            
            response = await asyncio.to_thread(
                self.gemini_client.generate_content, prompt
            )
            
            # Parse JSON response
            import json
            result = json.loads(response.text)
            
            return {
                'sentiment_score': float(result.get('sentiment_score', 0.0)),
                'sentiment_label': result.get('sentiment_label', 'neutral'),
                'confidence': float(result.get('confidence', 0.0))
            }
            
        except Exception as e:
            logger.error(f"Gemini sentiment analysis failed: {e}")
            return self._analyze_with_keywords(text)
    
    def _analyze_with_keywords(self, text: str) -> Dict[str, Any]:
        """Fallback keyword-based sentiment analysis"""
        positive_words = [
            'good', 'great', 'excellent', 'positive', 'bullish', 'growth',
            'profit', 'gain', 'rise', 'increase', 'up', 'strong', 'buy',
            'outperform', 'beat', 'exceed', 'optimistic', 'confident'
        ]
        
        negative_words = [
            'bad', 'terrible', 'negative', 'bearish', 'loss', 'decline',
            'fall', 'decrease', 'down', 'weak', 'sell', 'underperform',
            'miss', 'below', 'pessimistic', 'worried', 'concern', 'risk'
        ]
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        
        if total_words == 0:
            return {'sentiment_score': 0.0, 'sentiment_label': 'neutral', 'confidence': 0.0}
        
        # Calculate sentiment score
        sentiment_score = (positive_count - negative_count) / max(total_words * 0.1, 1)
        sentiment_score = max(-1.0, min(1.0, sentiment_score))  # Clamp to [-1, 1]
        
        # Determine label
        if sentiment_score > 0.1:
            sentiment_label = 'positive'
        elif sentiment_score < -0.1:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        # Calculate confidence based on number of sentiment words found
        confidence = min(1.0, (positive_count + negative_count) / max(total_words * 0.05, 1))
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'confidence': confidence
        }
    
    async def analyze_social_media_sentiment(self, social_data: List[Dict]) -> List[Dict]:
        """Analyze sentiment of social media posts"""
        analyzed_posts = []
        
        for post in social_data:
            try:
                content = post.get('content', '') or post.get('text', '')
                sentiment = await self._analyze_text_sentiment(content)
                
                post_with_sentiment = post.copy()
                post_with_sentiment.update(sentiment)
                analyzed_posts.append(post_with_sentiment)
                
            except Exception as e:
                logger.error(f"Failed to analyze social media sentiment: {e}")
                post_with_sentiment = post.copy()
                post_with_sentiment.update({
                    'sentiment_score': 0.0,
                    'sentiment_label': 'neutral',
                    'confidence': 0.0
                })
                analyzed_posts.append(post_with_sentiment)
        
        return analyzed_posts
    
    def calculate_aggregate_sentiment(self, analyzed_data: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate sentiment from multiple sources"""
        if not analyzed_data:
            return {
                'average_sentiment': 0.0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'confidence': 0.0,
                'total_items': 0
            }
        
        scores = [item.get('sentiment_score', 0.0) for item in analyzed_data]
        confidences = [item.get('confidence', 0.0) for item in analyzed_data]
        labels = [item.get('sentiment_label', 'neutral') for item in analyzed_data]
        
        # Calculate weighted average (weight by confidence)
        if sum(confidences) > 0:
            weighted_sentiment = sum(s * c for s, c in zip(scores, confidences)) / sum(confidences)
        else:
            weighted_sentiment = sum(scores) / len(scores)
        
        # Count sentiment distribution
        sentiment_distribution = {
            'positive': labels.count('positive'),
            'negative': labels.count('negative'),
            'neutral': labels.count('neutral')
        }
        
        return {
            'average_sentiment': weighted_sentiment,
            'sentiment_distribution': sentiment_distribution,
            'confidence': sum(confidences) / len(confidences) if confidences else 0.0,
            'total_items': len(analyzed_data)
        }
    
    async def get_symbol_sentiment(self, symbol: str, news_data: List[Dict] = None, 
                                 social_data: List[Dict] = None) -> Dict[str, Any]:
        """Get comprehensive sentiment analysis for a specific symbol"""
        all_analyzed_data = []
        
        # Analyze news data
        if news_data:
            # Filter news relevant to symbol
            relevant_news = [
                article for article in news_data
                if symbol.lower() in (article.get('title', '') + article.get('content', '')).lower()
            ]
            analyzed_news = await self.analyze_news_sentiment(relevant_news)
            all_analyzed_data.extend(analyzed_news)
        
        # Analyze social media data
        if social_data:
            # Filter social posts relevant to symbol
            relevant_posts = [
                post for post in social_data
                if symbol.lower() in (post.get('content', '') or post.get('text', '')).lower()
            ]
            analyzed_posts = await self.analyze_social_media_sentiment(relevant_posts)
            all_analyzed_data.extend(analyzed_posts)
        
        # Calculate aggregate sentiment
        aggregate = self.calculate_aggregate_sentiment(all_analyzed_data)
        aggregate['symbol'] = symbol
        aggregate['timestamp'] = datetime.now()
        
        return aggregate