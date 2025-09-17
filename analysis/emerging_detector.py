"""
Détecteur d'actions émergentes utilisant l'IA pour identifier les opportunités de croissance
Analyse les news, financiers, et signaux techniques pour détecter les entreprises prometteuses
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EmergingSignal:
    """Signal d'action émergente"""
    symbol: str
    company_name: str
    score: float  # Score de 0 à 100
    growth_potential: str  # 'high', 'medium', 'low'
    timeframe: str  # 'short', 'medium', 'long'
    key_drivers: List[str]  # Facteurs de croissance identifiés
    risk_factors: List[str]  # Risques identifiés
    market_cap: float
    sector: str
    reasoning: str
    confidence: float
    timestamp: datetime

class EmergingStockDetector:
    """Détecteur d'actions émergentes utilisant l'analyse IA multi-facteurs"""
    
    def __init__(self, config, sentiment_analyzer):
        self.config = config
        self.sentiment_analyzer = sentiment_analyzer
        self.news_keywords = self._load_growth_keywords()
        
    def _load_growth_keywords(self):
        """Mots-clés indicateurs de croissance"""
        return {
            'innovation': ['AI', 'artificial intelligence', 'machine learning', 'breakthrough', 
                         'innovation', 'patent', 'technology', 'disruptive', 'revolutionary'],
            'growth': ['expansion', 'growing', 'scaling', 'partnership', 'acquisition', 
                      'merger', 'new market', 'launch', 'revenue growth', 'market share'],
            'financial': ['profitable', 'earnings beat', 'revenue surprise', 'guidance raised',
                         'strong quarter', 'record sales', 'cash flow', 'margin expansion'],
            'market_trends': ['ESG', 'sustainable', 'renewable', 'clean energy', 'biotech',
                            'fintech', 'cloud', 'SaaS', 'cybersecurity', 'blockchain'],
            'leadership': ['new CEO', 'leadership change', 'strategic hire', 'board member',
                          'management team', 'vision', 'strategy', 'transformation']
        }
    
    async def scan_emerging_opportunities(self, news_data: List[Dict],
                                        market_data: Dict) -> List[EmergingSignal]:
        """Analyse les données pour détecter les opportunités émergentes"""

        logger.info("Scanning for emerging stock opportunities...")
        emerging_signals = []

        # Si pas de données news, créer une analyse alternative
        if not news_data or len(news_data) == 0:
            logger.info("No news data available, using alternative screening method")
            return await self._screen_by_technical_fundamentals()

        # Analyser les mentions d'entreprises dans les news
        company_mentions = await self._extract_company_mentions(news_data)
        
        # Pour chaque entreprise mentionnée positivement
        for company_info in company_mentions:
            try:
                signal = await self._analyze_emerging_potential(
                    company_info, news_data, market_data
                )
                if signal and signal.score > 60:  # Seuil de qualité
                    emerging_signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error analyzing {company_info.get('symbol', 'unknown')}: {e}")
        
        # Trier par score décroissant
        emerging_signals.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Found {len(emerging_signals)} emerging opportunities")
        return emerging_signals
    
    async def _extract_company_mentions(self, news_data: List[Dict]) -> List[Dict]:
        """Extrait les mentions d'entreprises dans les news avec analyse IA"""
        
        company_mentions = []
        
        for article in news_data:
            try:
                text = f"{article.get('title', '')} {article.get('content', '')}"
                
                # Utiliser Gemini pour extraire les entreprises mentionnées
                analysis = await self._analyze_companies_in_text(text)
                
                for company in analysis.get('companies', []):
                    company_mentions.append({
                        'company_name': company.get('name', ''),
                        'symbol': company.get('symbol', ''),
                        'sentiment': company.get('sentiment', 0.0),
                        'growth_indicators': company.get('growth_indicators', []),
                        'context': company.get('context', ''),
                        'source_article': article,
                        'relevance_score': company.get('relevance', 0.0)
                    })
                    
            except Exception as e:
                logger.error(f"Error extracting companies from article: {e}")
        
        # Déduplication et filtrage
        return self._deduplicate_mentions(company_mentions)
    
    async def _analyze_companies_in_text(self, text: str) -> Dict:
        """Utilise Gemini pour analyser les mentions d'entreprises dans le texte"""
        
        if not self.sentiment_analyzer.gemini_client:
            return {'companies': []}
        
        prompt = f"""
        Analysez le texte suivant et identifiez toutes les entreprises mentionnées avec leur potentiel de croissance.
        
        Pour chaque entreprise trouvée, retournez:
        - Nom de l'entreprise
        - Symbole boursier si disponible  
        - Score de sentiment (-1 à 1)
        - Indicateurs de croissance mentionnés
        - Contexte de la mention
        - Score de pertinence pour l'investissement (0 à 1)
        
        Concentrez-vous sur les entreprises avec des signaux positifs de croissance, innovation, ou expansion.
        
        Texte: {text[:1000]}
        
        Retournez au format JSON:
        {{"companies": [
            {{
                "name": "Nom entreprise",
                "symbol": "SYMBOL", 
                "sentiment": 0.8,
                "growth_indicators": ["innovation", "partnership"],
                "context": "Description du contexte",
                "relevance": 0.9
            }}
        ]}}
        """
        
        try:
            response = await asyncio.to_thread(
                self.sentiment_analyzer.gemini_client.generate_content, prompt
            )
            
            # Parser la réponse JSON
            import json
            response_text = response.text
            
            # Nettoyer la réponse pour extraire le JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != 0:
                json_text = response_text[start:end]
                return json.loads(json_text)
            
            return {'companies': []}
            
        except Exception as e:
            logger.debug(f"Error in company analysis: {e}")
            return {'companies': []}
    
    def _deduplicate_mentions(self, mentions: List[Dict]) -> List[Dict]:
        """Supprime les doublons et combine les mentions multiples"""
        
        company_map = {}
        
        for mention in mentions:
            key = mention.get('symbol') or mention.get('company_name', '').upper()
            if not key:
                continue
                
            if key in company_map:
                # Combiner les scores et indicateurs
                existing = company_map[key]
                existing['sentiment'] = (existing['sentiment'] + mention['sentiment']) / 2
                existing['growth_indicators'].extend(mention['growth_indicators'])
                existing['relevance_score'] = max(existing['relevance_score'], mention['relevance_score'])
            else:
                company_map[key] = mention
        
        return list(company_map.values())
    
    async def _analyze_emerging_potential(self, company_info: Dict, 
                                        news_data: List[Dict], 
                                        market_data: Dict) -> Optional[EmergingSignal]:
        """Analyse le potentiel émergent d'une entreprise"""
        
        symbol = company_info.get('symbol', '')
        if not symbol:
            return None
        
        try:
            # Récupérer les données financières
            financial_data = await self._get_financial_metrics(symbol)
            if not financial_data:
                return None
            
            # Calculer le score composite
            score = await self._calculate_emerging_score(
                company_info, financial_data, news_data
            )
            
            # Générer l'analyse avec Gemini
            analysis = await self._generate_ai_analysis(
                symbol, company_info, financial_data
            )
            
            return EmergingSignal(
                symbol=symbol,
                company_name=company_info.get('company_name', symbol),
                score=score,
                growth_potential=analysis.get('growth_potential', 'medium'),
                timeframe=analysis.get('timeframe', 'medium'),
                key_drivers=analysis.get('key_drivers', []),
                risk_factors=analysis.get('risk_factors', []),
                market_cap=financial_data.get('market_cap', 0),
                sector=financial_data.get('sector', 'Unknown'),
                reasoning=analysis.get('reasoning', ''),
                confidence=analysis.get('confidence', 0.5),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing emerging potential for {symbol}: {e}")
            return None
    
    async def _get_financial_metrics(self, symbol: str) -> Optional[Dict]:
        """Récupère les métriques financières clés"""
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Informations de base
            info = await asyncio.to_thread(lambda: ticker.info)
            
            # Données historiques pour calculs
            hist = await asyncio.to_thread(ticker.history, period="1y")
            if hist.empty:
                return None
            
            # Calculer les métriques
            current_price = hist['Close'].iloc[-1]
            price_52w_high = hist['High'].max()
            price_52w_low = hist['Low'].min()
            
            # Performance récente
            returns_1m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-22] - 1) * 100 if len(hist) >= 22 else 0
            returns_3m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-66] - 1) * 100 if len(hist) >= 66 else 0
            returns_1y = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
            
            # Volume tendance
            avg_volume_3m = hist['Volume'].tail(66).mean() if len(hist) >= 66 else hist['Volume'].mean()
            recent_volume = hist['Volume'].tail(10).mean()
            volume_trend = (recent_volume / avg_volume_3m - 1) * 100 if avg_volume_3m > 0 else 0
            
            return {
                'market_cap': info.get('marketCap', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'current_price': current_price,
                'price_52w_high': price_52w_high,
                'price_52w_low': price_52w_low,
                'price_to_52w_high': (current_price / price_52w_high) if price_52w_high > 0 else 0,
                'returns_1m': returns_1m,
                'returns_3m': returns_3m,
                'returns_1y': returns_1y,
                'volume_trend': volume_trend,
                'pe_ratio': info.get('trailingPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'beta': info.get('beta', 1.0)
            }
            
        except Exception as e:
            logger.debug(f"Could not get financial data for {symbol}: {e}")
            return None
    
    async def _calculate_emerging_score(self, company_info: Dict, 
                                      financial_data: Dict, 
                                      news_data: List[Dict]) -> float:
        """Calcule un score composite pour le potentiel émergent"""
        
        score = 0.0
        
        # 1. Score de sentiment (0-25 points)
        sentiment_score = max(0, min(25, company_info.get('sentiment', 0) * 25 + 12.5))
        score += sentiment_score
        
        # 2. Score de croissance technique (0-25 points)
        returns_3m = financial_data.get('returns_3m', 0)
        volume_trend = financial_data.get('volume_trend', 0)
        
        tech_score = 0
        if returns_3m > 20:  # +20% en 3 mois
            tech_score += 15
        elif returns_3m > 10:
            tech_score += 10
        elif returns_3m > 0:
            tech_score += 5
            
        if volume_trend > 50:  # Volume en hausse
            tech_score += 10
        elif volume_trend > 20:
            tech_score += 5
            
        score += min(25, tech_score)
        
        # 3. Score fondamental (0-25 points)
        fundamental_score = 0
        revenue_growth = financial_data.get('revenue_growth', 0)
        earnings_growth = financial_data.get('earnings_growth', 0)
        
        if revenue_growth > 0.2:  # +20% croissance
            fundamental_score += 15
        elif revenue_growth > 0.1:
            fundamental_score += 10
            
        if earnings_growth > 0.2:
            fundamental_score += 10
        elif earnings_growth > 0:
            fundamental_score += 5
            
        score += min(25, fundamental_score)
        
        # 4. Score d'innovation/disruption (0-25 points)
        innovation_keywords = company_info.get('growth_indicators', [])
        innovation_score = min(25, len(innovation_keywords) * 5)
        score += innovation_score
        
        return min(100, score)
    
    async def _generate_ai_analysis(self, symbol: str, company_info: Dict, 
                                  financial_data: Dict) -> Dict:
        """Génère une analyse IA complète du potentiel"""
        
        if not self.sentiment_analyzer.gemini_client:
            return {
                'growth_potential': 'medium',
                'timeframe': 'medium',
                'key_drivers': ['growth indicators detected'],
                'risk_factors': ['market volatility'],
                'reasoning': 'AI analysis not available',
                'confidence': 0.5
            }
        
        prompt = f"""
        Analysez le potentiel d'investissement émergent pour l'action {symbol}.
        
        Données de l'entreprise:
        - Nom: {company_info.get('company_name', symbol)}
        - Secteur: {financial_data.get('sector', 'Unknown')}
        - Capitalisation: {financial_data.get('market_cap', 0):,.0f}
        - Croissance revenue: {financial_data.get('revenue_growth', 0)*100:.1f}%
        - Performance 3M: {financial_data.get('returns_3m', 0):.1f}%
        - Indicateurs de croissance: {company_info.get('growth_indicators', [])}
        - Contexte news: {company_info.get('context', '')}
        
        Évaluez:
        1. Potentiel de croissance: high/medium/low
        2. Horizon temporel optimal: short/medium/long (court/moyen/long terme)
        3. 3-5 facteurs clés de croissance
        4. 3-5 facteurs de risque principaux
        5. Raisonnement détaillé
        6. Niveau de confiance 0-1
        
        Format JSON:
        {{
            "growth_potential": "high/medium/low",
            "timeframe": "short/medium/long", 
            "key_drivers": ["driver1", "driver2", "driver3"],
            "risk_factors": ["risk1", "risk2", "risk3"],
            "reasoning": "Analyse détaillée...",
            "confidence": 0.8
        }}
        """
        
        try:
            response = await asyncio.to_thread(
                self.sentiment_analyzer.gemini_client.generate_content, prompt
            )
            
            # Parser JSON response
            import json
            response_text = response.text
            
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != 0:
                return json.loads(response_text[start:end])
            
        except Exception as e:
            logger.debug(f"Error in AI analysis: {e}")
        
        # Fallback
        return {
            'growth_potential': 'medium',
            'timeframe': 'medium', 
            'key_drivers': company_info.get('growth_indicators', [])[:3],
            'risk_factors': ['Market volatility', 'Competition', 'Execution risk'],
            'reasoning': 'Automated analysis based on available data',
            'confidence': 0.6
        }
    
    def filter_by_criteria(self, signals: List[EmergingSignal], 
                          min_score: float = 70,
                          max_market_cap: float = 10e9,  # 10B max
                          preferred_sectors: List[str] = None) -> List[EmergingSignal]:
        """Filtre les signaux selon des critères spécifiques"""
        
        filtered = []
        
        for signal in signals:
            if signal.score < min_score:
                continue
                
            if signal.market_cap > max_market_cap:
                continue
                
            if preferred_sectors and signal.sector not in preferred_sectors:
                continue
                
            filtered.append(signal)
        
        return filtered
    
    async def get_emerging_watchlist(self, top_n: int = 20) -> List[Dict]:
        """Retourne une liste de surveillance des actions émergentes"""

        try:
            # Utiliser la méthode de screening alternatif si pas de news
            signals = await self._screen_by_technical_fundamentals()
            
            # Convertir en format simple pour la watchlist
            watchlist = []
            for signal in signals[:top_n]:
                watchlist.append({
                    'symbol': signal.symbol,
                    'company_name': signal.company_name,
                    'score': signal.score,
                    'growth_potential': signal.growth_potential,
                    'sector': signal.sector,
                    'key_drivers': signal.key_drivers[:3],
                    'timeframe': signal.timeframe,
                    'confidence': signal.confidence
                })
            
            return watchlist
            
        except Exception as e:
            logger.error(f"Error generating emerging watchlist: {e}")
            return []

    async def _screen_by_technical_fundamentals(self) -> List[EmergingSignal]:
        """Screening alternatif basé sur l'analyse technique et fondamentale"""

        logger.info("Using technical and fundamental screening for emerging opportunities")
        emerging_signals = []

        # Utiliser la liste de symboles du config
        symbols_to_screen = getattr(self.config, 'ALL_SYMBOLS', [])[:50]  # Limiter pour la performance

        for symbol in symbols_to_screen:
            try:
                # Analyse rapide du potentiel
                financial_data = await self._get_financial_metrics(symbol)
                if not financial_data:
                    continue

                # Calculer un score basé sur les métriques
                score = await self._calculate_fundamental_score(financial_data)

                if score > 65:  # Seuil pour les opportunités émergentes
                    # Générer un signal
                    signal = EmergingSignal(
                        symbol=symbol,
                        company_name=financial_data.get('symbol', symbol),
                        score=score,
                        growth_potential=self._determine_growth_potential(financial_data),
                        timeframe='medium',
                        key_drivers=self._extract_key_drivers(financial_data),
                        risk_factors=['Market risk', 'Sector risk', 'Company specific risk'],
                        market_cap=financial_data.get('market_cap', 0),
                        sector=financial_data.get('sector', 'Unknown'),
                        reasoning=f"Strong technical and fundamental metrics with score {score:.1f}",
                        confidence=min(0.9, score / 100),
                        timestamp=datetime.now()
                    )
                    emerging_signals.append(signal)

            except Exception as e:
                logger.debug(f"Error screening {symbol}: {e}")
                continue

        # Trier par score
        emerging_signals.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Found {len(emerging_signals)} emerging opportunities via technical screening")
        return emerging_signals[:20]  # Top 20

    async def _calculate_fundamental_score(self, financial_data: Dict) -> float:
        """Score basé sur les fondamentaux et techniques"""

        score = 50.0  # Score de base

        # Performance récente (20 points max)
        returns_3m = financial_data.get('returns_3m', 0)
        if returns_3m > 30:
            score += 20
        elif returns_3m > 15:
            score += 15
        elif returns_3m > 5:
            score += 10
        elif returns_3m < -20:
            score -= 10

        # Croissance fundamentale (15 points max)
        revenue_growth = financial_data.get('revenue_growth', 0) or 0
        earnings_growth = financial_data.get('earnings_growth', 0) or 0

        if revenue_growth > 0.25:  # 25%+
            score += 8
        elif revenue_growth > 0.15:
            score += 5
        elif revenue_growth > 0.05:
            score += 2

        if earnings_growth > 0.25:
            score += 7
        elif earnings_growth > 0.15:
            score += 4
        elif earnings_growth > 0.05:
            score += 2

        # Valorisation (10 points max)
        pe_ratio = financial_data.get('pe_ratio', 0) or 0
        if 10 < pe_ratio < 25:  # PE raisonnable
            score += 5
        elif pe_ratio < 10 and pe_ratio > 0:
            score += 8  # Potentiellement sous-valorisé
        elif pe_ratio > 50:
            score -= 5  # Surévalué

        # Tendance volume (5 points max)
        volume_trend = financial_data.get('volume_trend', 0)
        if volume_trend > 50:
            score += 5
        elif volume_trend > 20:
            score += 3

        # Position dans la range 52 semaines (10 points max)
        price_to_52w_high = financial_data.get('price_to_52w_high', 0.5)
        if price_to_52w_high > 0.9:  # Près des hauts
            score += 10
        elif price_to_52w_high > 0.7:
            score += 7
        elif price_to_52w_high < 0.3:  # Trop bas, potentiel problème
            score -= 5

        return min(100, max(0, score))

    def _determine_growth_potential(self, financial_data: Dict) -> str:
        """Détermine le potentiel de croissance"""

        returns_3m = financial_data.get('returns_3m', 0)
        revenue_growth = financial_data.get('revenue_growth', 0) or 0
        earnings_growth = financial_data.get('earnings_growth', 0) or 0

        if (returns_3m > 20 and revenue_growth > 0.2) or earnings_growth > 0.3:
            return 'high'
        elif (returns_3m > 10 and revenue_growth > 0.1) or earnings_growth > 0.15:
            return 'medium'
        else:
            return 'low'

    def _extract_key_drivers(self, financial_data: Dict) -> List[str]:
        """Extrait les facteurs clés de croissance"""

        drivers = []

        if financial_data.get('returns_3m', 0) > 15:
            drivers.append('Strong recent performance')

        if financial_data.get('revenue_growth', 0) > 0.15:
            drivers.append('Revenue growth acceleration')

        if financial_data.get('earnings_growth', 0) > 0.2:
            drivers.append('Earnings expansion')

        if financial_data.get('volume_trend', 0) > 30:
            drivers.append('Increased institutional interest')

        sector = financial_data.get('sector', '')
        if sector in ['Technology', 'Healthcare', 'Consumer Discretionary']:
            drivers.append(f'Growth sector exposure ({sector})')

        if not drivers:
            drivers = ['Technical momentum', 'Market positioning']

        return drivers[:5]