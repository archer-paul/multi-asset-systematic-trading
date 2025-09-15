'''
Macro-Economic and Geopolitical Analyzer for Trading Bot

This module is responsible for fetching and analyzing news and data from
institutional, macro-economic, and geopolitical sources to provide a broad
context for market movements.
'''

import logging
import asyncio
import feedparser
from typing import Dict, List, Any, Tuple

from core.config import Config
from analysis.sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)

# A curated list of reliable RSS feeds for macro-economic and geopolitical news
INSTITUTIONAL_SOURCES = {
    # Central Banks
    'fed_fomc': 'https://www.federalreserve.gov/feeds/press_all.xml',
    'ecb_press': 'https://www.ecb.europa.eu/rss/press.html',
    'boe_news': 'https://www.bankofengland.co.uk/rss/news',
    
    # International Organizations
    'imf_news': 'https://www.imf.org/external/rss/news.xml',
    'world_bank': 'https://www.worldbank.org/en/news/rss',
    'bis_press': 'https://www.bis.org/press/pressreleases.rss',

    # Economic Data & Government
    'us_treasury': 'https://home.treasury.gov/news/press-releases/feed',
    'fred_economic': 'https://fred.stlouisfed.org/rss/releases',
}

GEOPOLITICAL_SOURCES = {
    # Think Tanks & Analysis
    'council_foreign_relations': 'https://www.cfr.org/rss/all',
    'chatham_house': 'https://www.chathamhouse.org/rss/news',
    'foreign_affairs': 'https://www.foreignaffairs.com/rss.xml',
    'politico_economy': 'https://rss.politico.com/economy.xml',
    'financial_times_world': 'https://www.ft.com/world?format=rss',
}

class MacroEconomicAnalyzer:
    """Analyzes macro-economic and geopolitical news feeds."""

    def __init__(self, config: Config, sentiment_analyzer: SentimentAnalyzer):
        self.config = config
        self.sentiment_analyzer = sentiment_analyzer
        self.all_sources = {**INSTITUTIONAL_SOURCES, **GEOPOLITICAL_SOURCES}

    async def fetch_feed(self, session, name: str, url: str) -> List[Dict[str, Any]]:
        """Asynchronously fetches and parses a single RSS feed."""
        articles = []
        try:
            async with session.get(url, timeout=15) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    for entry in feed.entries[:5]: # Limit to 5 most recent entries per feed
                        articles.append({
                            'source': name,
                            'title': entry.title,
                            'link': entry.link,
                            'published': entry.get('published_parsed', entry.get('updated_parsed')),
                            'summary': entry.get('summary', '')
                        })
                else:
                    logger.warning(f"Failed to fetch {name} feed. Status: {response.status}")
        except Exception as e:
            logger.error(f"Error fetching or parsing feed {name} from {url}: {e}")
        return articles

    async def fetch_all_feeds(self) -> List[Dict[str, Any]]:
        """Fetches all configured macro and geopolitical RSS feeds concurrently."""
        import aiohttp
        all_articles = []
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_feed(session, name, url) for name, url in self.all_sources.items()]
            results = await asyncio.gather(*tasks)
            for article_list in results:
                all_articles.extend(article_list)
        
        logger.info(f"Fetched {len(all_articles)} articles from {len(self.all_sources)} macro/geopolitical sources.")
        return all_articles

    async def analyze_macro_sentiment(self, text: str) -> Dict[str, Any]:
        """ 
        Uses a specialized prompt with Gemini to analyze text for macro-economic
        and geopolitical sentiment and themes.
        """
        prompt = f'''
        Analyze the following text from a financial or geopolitical news source.
        Classify the sentiment and extract key themes based on these categories:

        1.  **Monetary Policy Stance**: Classify as 'Hawkish', 'Dovish', or 'Neutral'.
        2.  **Economic Outlook**: Classify as 'Expansionary', 'Contractionary', or 'Stable'.
        3.  **Geopolitical Risk**: Classify as 'Elevating', 'De-escalating', or 'Stable'.
        4.  **Market Impact**: Assess the potential market impact from 'High' to 'Low'.
        5.  **Key Themes**: Extract up to 3 key themes (e.g., 'inflation concerns', 'trade tensions', 'rate hike').
        6.  **Affected Regions/Sectors**: List regions (e.g., 'EU', 'Asia') or sectors (e.g., 'Energy', 'Tech') most affected.

        Format the output as a JSON object.

        Text to analyze:
        """{text}"""
        '''
        try:
            # This assumes the sentiment_analyzer has a generic method to call the AI model
            sentiment_data = await self.sentiment_analyzer.analyze_financial_sentiment(text=prompt)
            return sentiment_data
        except Exception as e:
            logger.error(f"Failed to get macro sentiment analysis from AI: {e}")
            return {'error': str(e)}

    async def get_full_macro_analysis(self) -> Dict[str, Any]:
        """
        Fetches all feeds, analyzes them, and returns a consolidated summary.
        """
        articles = await self.fetch_all_feeds()
        analyzed_articles = []

        for article in articles:
            full_text = f"{article['title']}. {article['summary']}"
            macro_sentiment = await self.analyze_macro_sentiment(full_text)
            article['analysis'] = macro_sentiment
            analyzed_articles.append(article)

        # Consolidate the analysis
        overall_sentiment = self._consolidate_analysis(analyzed_articles)
        return {
            'overall_sentiment': overall_sentiment,
            'articles': analyzed_articles
        }

    def _consolidate_analysis(self, analyzed_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidates analysis from multiple articles into a single summary view."""
        summary = {
            'monetary_policy_score': 0,
            'economic_outlook_score': 0,
            'geopolitical_risk_score': 0,
            'key_themes': {},
            'affected_sectors': {},
        }
        total_articles = len(analyzed_articles)
        if total_articles == 0: return summary

        scores = {'Hawkish': 1, 'Dovish': -1, 'Expansionary': 1, 'Contractionary': -1, 'Elevating': 1, 'De-escalating': -1}

        for article in analyzed_articles:
            analysis = article.get('analysis', {})
            if not isinstance(analysis, dict): continue

            summary['monetary_policy_score'] += scores.get(analysis.get('Monetary Policy Stance'), 0)
            summary['economic_outlook_score'] += scores.get(analysis.get('Economic Outlook'), 0)
            summary['geopolitical_risk_score'] += scores.get(analysis.get('Geopolitical Risk'), 0)

            for theme in analysis.get('Key Themes', []):
                summary['key_themes'][theme] = summary['key_themes'].get(theme, 0) + 1
            
            for sector in analysis.get('Affected Regions/Sectors', []):
                summary['affected_sectors'][sector] = summary['affected_sectors'].get(sector, 0) + 1

        # Normalize scores
        summary['monetary_policy_score'] /= total_articles
        summary['economic_outlook_score'] /= total_articles
        summary['geopolitical_risk_score'] /= total_articles

        # Sort themes and sectors by frequency
        summary['key_themes'] = sorted(summary['key_themes'].items(), key=lambda x: x[1], reverse=True)
        summary['affected_sectors'] = sorted(summary['affected_sectors'].items(), key=lambda x: x[1], reverse=True)

        return summary
