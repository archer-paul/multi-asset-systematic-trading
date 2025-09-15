"""
Analyseur d'investissements à long terme (3-5 ans)
Utilise l'IA pour identifier les meilleures opportunités d'investissement à horizon long
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class LongTermRecommendation:
    """Recommandation d'investissement long terme"""
    symbol: str
    company_name: str
    sector: str
    
    # Scores (0-100)
    overall_score: float
    growth_score: float
    quality_score: float
    value_score: float
    moat_score: float  # Avantage concurrentiel
    
    # Prédictions
    target_price_3y: float
    expected_return_3y: float
    expected_return_5y: float
    
    # Analyse qualitative
    investment_thesis: str
    key_strengths: List[str]
    key_risks: List[str]
    catalysts: List[str]  # Catalyseurs de croissance
    
    # Métriques
    pe_ratio: float
    peg_ratio: float
    roe: float
    debt_to_equity: float
    
    confidence: float
    recommendation: str  # 'Strong Buy', 'Buy', 'Hold', 'Sell'
    timestamp: datetime

class LongTermAnalyzer:
    """Analyse les investissements à horizon 3-5 ans"""
    
    def __init__(self, config, sentiment_analyzer):
        self.config = config
        self.sentiment_analyzer = sentiment_analyzer
        
        # Pondérations pour le score composite
        self.weights = {
            'growth': 0.3,
            'quality': 0.25,
            'value': 0.2,
            'moat': 0.25
        }
        
        # Seuils de qualité
        self.thresholds = {
            'min_market_cap': 1e9,  # 1B minimum
            'min_revenue': 100e6,   # 100M minimum
            'max_pe': 50,           # PE trop élevé = risque
            'min_roe': 0.1,         # ROE minimum 10%
            'max_debt_equity': 2.0  # Max dette/equity
        }
    
    async def analyze_long_term_opportunities(self, symbols: List[str]) -> List[LongTermRecommendation]:
        """Analyse les opportunités d'investissement long terme"""
        
        logger.info(f"Analyzing long-term opportunities for {len(symbols)} symbols")
        recommendations = []
        
        for symbol in symbols:
            try:
                recommendation = await self._analyze_single_stock(symbol)
                if recommendation and recommendation.overall_score > 60:
                    recommendations.append(recommendation)
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol} for long-term: {e}")
                continue
        
        # Trier par score décroissant
        recommendations.sort(key=lambda x: x.overall_score, reverse=True)
        
        logger.info(f"Generated {len(recommendations)} long-term recommendations")
        return recommendations
    
    async def _analyze_single_stock(self, symbol: str) -> Optional[LongTermRecommendation]:
        """Analyse approfondie d'une action pour investissement long terme"""
        
        try:
            # Récupérer les données complètes
            stock_data = await self._get_comprehensive_data(symbol)
            if not stock_data:
                return None
            
            # Vérifier les critères de base
            if not self._meets_basic_criteria(stock_data):
                logger.debug(f"{symbol} doesn't meet basic criteria")
                return None
            
            # Calculer les scores
            scores = await self._calculate_all_scores(symbol, stock_data)
            
            # Générer l'analyse IA
            ai_analysis = await self._generate_long_term_analysis(symbol, stock_data, scores)
            
            # Calculer les prédictions de prix
            price_targets = self._calculate_price_targets(stock_data, scores)
            
            return LongTermRecommendation(
                symbol=symbol,
                company_name=stock_data['info'].get('shortName', symbol),
                sector=stock_data['info'].get('sector', 'Unknown'),
                
                overall_score=scores['overall'],
                growth_score=scores['growth'],
                quality_score=scores['quality'], 
                value_score=scores['value'],
                moat_score=scores['moat'],
                
                target_price_3y=price_targets['3y'],
                expected_return_3y=price_targets['return_3y'],
                expected_return_5y=price_targets['return_5y'],
                
                investment_thesis=ai_analysis.get('thesis', ''),
                key_strengths=ai_analysis.get('strengths', []),
                key_risks=ai_analysis.get('risks', []),
                catalysts=ai_analysis.get('catalysts', []),
                
                pe_ratio=stock_data['ratios']['pe'],
                peg_ratio=stock_data['ratios']['peg'],
                roe=stock_data['ratios']['roe'],
                debt_to_equity=stock_data['ratios']['debt_equity'],
                
                confidence=ai_analysis.get('confidence', 0.5),
                recommendation=self._get_recommendation(scores['overall']),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in single stock analysis for {symbol}: {e}")
            return None
    
    async def _get_comprehensive_data(self, symbol: str) -> Optional[Dict]:
        """Récupère toutes les données nécessaires pour l'analyse"""
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Informations de base (non-blocking)
            info = await asyncio.to_thread(lambda: ticker.info)
            if not info or info.get('marketCap', 0) < self.thresholds['min_market_cap']:
                return None
            
            # Données historiques (non-blocking)
            hist_5y_task = asyncio.to_thread(ticker.history, period="5y")
            hist_1y_task = asyncio.to_thread(ticker.history, period="1y")
            
            # Données financières (non-blocking)
            financials_task = asyncio.to_thread(lambda: ticker.financials)
            balance_sheet_task = asyncio.to_thread(lambda: ticker.balance_sheet)
            cash_flow_task = asyncio.to_thread(lambda: ticker.cashflow)
            
            # Exécuter toutes les collectes en parallèle
            hist_5y, hist_1y, financials, balance_sheet, cash_flow = await asyncio.gather(
                hist_5y_task, hist_1y_task, financials_task, balance_sheet_task, cash_flow_task
            )
            
            if hist_1y.empty:
                return None
            
            # Calculer des métriques importantes
            current_price = hist_1y['Close'].iloc[-1]
            
            return {
                'info': info,
                'current_price': current_price,
                'hist_5y': hist_5y,
                'hist_1y': hist_1y,
                'financials': financials,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'ratios': self._calculate_financial_ratios(info, financials, balance_sheet)
            }
            
        except Exception as e:
            logger.debug(f"Could not get comprehensive data for {symbol}: {e}")
            return None
    
    def _calculate_financial_ratios(self, info: Dict, financials: pd.DataFrame, 
                                  balance_sheet: pd.DataFrame) -> Dict:
        """Calcule les ratios financiers clés"""
        
        ratios = {
            'pe': info.get('trailingPE', 0) or info.get('forwardPE', 0),
            'peg': info.get('pegRatio', 0),
            'pb': info.get('priceToBook', 0),
            'ps': info.get('priceToSalesTrailing12Months', 0),
            'roe': info.get('returnOnEquity', 0) or 0,
            'roa': info.get('returnOnAssets', 0) or 0,
            'debt_equity': info.get('debtToEquity', 0) or 0,
            'current_ratio': info.get('currentRatio', 0),
            'revenue_growth': info.get('revenueGrowth', 0) or 0,
            'earnings_growth': info.get('earningsGrowth', 0) or 0,
            'gross_margin': info.get('grossMargins', 0) or 0,
            'operating_margin': info.get('operatingMargins', 0) or 0,
            'profit_margin': info.get('profitMargins', 0) or 0
        }
        
        # Calculer des ratios supplémentaires si les données sont disponibles
        try:
            if not financials.empty and not balance_sheet.empty:
                # ROE calculation si pas dans info
                if ratios['roe'] == 0:
                    net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else 0
                    shareholders_equity = balance_sheet.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in balance_sheet.index else 0
                    if shareholders_equity != 0:
                        ratios['roe'] = net_income / shareholders_equity
        except:
            pass  # Utiliser les valeurs par défaut
        
        return ratios
    
    def _meets_basic_criteria(self, stock_data: Dict) -> bool:
        """Vérifie les critères de base pour l'investissement long terme"""
        
        info = stock_data['info']
        ratios = stock_data['ratios']
        
        # Market cap minimum
        if info.get('marketCap', 0) < self.thresholds['min_market_cap']:
            return False
        
        # Revenus minimum
        if info.get('totalRevenue', 0) < self.thresholds['min_revenue']:
            return False
        
        # PE ratio raisonnable (pas de bulle)
        if ratios['pe'] > self.thresholds['max_pe'] and ratios['pe'] > 0:
            return False
        
        # Dette pas excessive
        if ratios['debt_equity'] > self.thresholds['max_debt_equity'] and ratios['debt_equity'] > 0:
            return False
        
        return True
    
    async def _calculate_all_scores(self, symbol: str, stock_data: Dict) -> Dict[str, float]:
        """Calcule tous les scores composants"""
        
        ratios = stock_data['ratios']
        info = stock_data['info']
        hist_5y = stock_data['hist_5y']
        
        # Score de croissance (0-100)
        growth_score = self._calculate_growth_score(ratios, hist_5y)
        
        # Score de qualité (0-100)
        quality_score = self._calculate_quality_score(ratios, info)
        
        # Score de valorisation (0-100)
        value_score = self._calculate_value_score(ratios)
        
        # Score de "moat" / avantage concurrentiel (0-100)
        moat_score = await self._calculate_moat_score(symbol, info, ratios)
        
        # Score global pondéré
        overall_score = (
            growth_score * self.weights['growth'] +
            quality_score * self.weights['quality'] +
            value_score * self.weights['value'] +
            moat_score * self.weights['moat']
        )
        
        return {
            'growth': growth_score,
            'quality': quality_score,
            'value': value_score,
            'moat': moat_score,
            'overall': overall_score
        }
    
    def _calculate_growth_score(self, ratios: Dict, hist_5y: pd.DataFrame) -> float:
        """Score de croissance basé sur les métriques historiques et futures"""
        
        score = 0.0
        
        # Croissance du chiffre d'affaires (0-30 points)
        revenue_growth = ratios['revenue_growth']
        if revenue_growth > 0.25:  # 25%+
            score += 30
        elif revenue_growth > 0.15:  # 15%+
            score += 25
        elif revenue_growth > 0.1:   # 10%+
            score += 20
        elif revenue_growth > 0.05:  # 5%+
            score += 10
        
        # Croissance des bénéfices (0-30 points)
        earnings_growth = ratios['earnings_growth']
        if earnings_growth > 0.2:    # 20%+
            score += 30
        elif earnings_growth > 0.15: # 15%+
            score += 25
        elif earnings_growth > 0.1:  # 10%+
            score += 20
        elif earnings_growth > 0:    # Positif
            score += 10
        
        # Performance prix historique (0-25 points)
        if not hist_5y.empty and len(hist_5y) > 252:
            returns_5y = (hist_5y['Close'].iloc[-1] / hist_5y['Close'].iloc[0] - 1)
            annualized_return = (1 + returns_5y) ** (1/5) - 1
            
            if annualized_return > 0.15:   # 15%+ par an
                score += 25
            elif annualized_return > 0.1:  # 10%+ par an
                score += 20
            elif annualized_return > 0.05: # 5%+ par an
                score += 15
            elif annualized_return > 0:    # Positif
                score += 10
        
        # Marges en expansion (0-15 points)
        if ratios['operating_margin'] > 0.15:  # 15%+ marge opérationnelle
            score += 15
        elif ratios['operating_margin'] > 0.1:  # 10%+
            score += 10
        elif ratios['operating_margin'] > 0.05: # 5%+
            score += 5
        
        return min(100, score)
    
    def _calculate_quality_score(self, ratios: Dict, info: Dict) -> float:
        """Score de qualité de l'entreprise"""
        
        score = 0.0
        
        # Return on Equity (0-25 points)
        roe = ratios['roe']
        if roe > 0.2:      # 20%+
            score += 25
        elif roe > 0.15:   # 15%+
            score += 20
        elif roe > 0.1:    # 10%+
            score += 15
        elif roe > 0.05:   # 5%+
            score += 10
        
        # Marges de profit (0-20 points)
        profit_margin = ratios['profit_margin']
        if profit_margin > 0.15:   # 15%+
            score += 20
        elif profit_margin > 0.1:  # 10%+
            score += 15
        elif profit_margin > 0.05: # 5%+
            score += 10
        elif profit_margin > 0:    # Positif
            score += 5
        
        # Ratio d'endettement (0-20 points)
        debt_equity = ratios['debt_equity']
        if debt_equity < 0.3:      # Très peu de dette
            score += 20
        elif debt_equity < 0.5:    # Peu de dette
            score += 15
        elif debt_equity < 1.0:    # Modéré
            score += 10
        elif debt_equity < 1.5:    # Acceptable
            score += 5
        
        # Liquidité (0-15 points)
        current_ratio = ratios['current_ratio']
        if current_ratio > 2.0:    # Très liquide
            score += 15
        elif current_ratio > 1.5:  # Bonne liquidité
            score += 12
        elif current_ratio > 1.2:  # Acceptable
            score += 8
        elif current_ratio > 1.0:  # Limite
            score += 5
        
        # Stabilité du secteur (0-20 points)
        sector = info.get('sector', '')
        stable_sectors = ['Consumer Staples', 'Utilities', 'Healthcare', 'Technology']
        if sector in stable_sectors:
            score += 15
        else:
            score += 10  # Autres secteurs
        
        return min(100, score)
    
    def _calculate_value_score(self, ratios: Dict) -> float:
        """Score de valorisation (plus faible = mieux)"""
        
        score = 0.0
        
        # P/E Ratio (0-30 points)
        pe = ratios['pe']
        if pe > 0:
            if pe < 12:        # Très bon marché
                score += 30
            elif pe < 16:      # Bon marché
                score += 25
            elif pe < 20:      # Raisonnable
                score += 20
            elif pe < 25:      # Acceptable
                score += 15
            elif pe < 30:      # Élevé
                score += 10
            else:              # Très élevé
                score += 5
        else:
            score += 15  # Pas de PE disponible
        
        # PEG Ratio (0-25 points)
        peg = ratios['peg']
        if peg > 0:
            if peg < 1.0:      # Excellent
                score += 25
            elif peg < 1.5:    # Bon
                score += 20
            elif peg < 2.0:    # Acceptable
                score += 15
            elif peg < 2.5:    # Élevé
                score += 10
            else:              # Très élevé
                score += 5
        else:
            score += 15
        
        # Price to Book (0-20 points)
        pb = ratios['pb']
        if pb > 0:
            if pb < 1.5:       # Très bon marché
                score += 20
            elif pb < 2.5:     # Bon marché
                score += 15
            elif pb < 4.0:     # Raisonnable
                score += 10
            else:              # Élevé
                score += 5
        else:
            score += 10
        
        # Price to Sales (0-25 points)
        ps = ratios['ps']
        if ps > 0:
            if ps < 1.0:       # Excellent
                score += 25
            elif ps < 2.0:     # Bon
                score += 20
            elif ps < 4.0:     # Acceptable
                score += 15
            elif ps < 6.0:     # Élevé
                score += 10
            else:              # Très élevé
                score += 5
        else:
            score += 12
        
        return min(100, score)
    
    async def _calculate_moat_score(self, symbol: str, info: Dict, ratios: Dict) -> float:
        """Score d'avantage concurrentiel / "moat économique" """
        
        score = 0.0
        
        # Taille et position de marché (0-25 points)
        market_cap = info.get('marketCap', 0)
        if market_cap > 100e9:    # 100B+ = mega cap
            score += 25
        elif market_cap > 50e9:   # 50B+ = large cap
            score += 20
        elif market_cap > 10e9:   # 10B+ = mid-large cap
            score += 15
        elif market_cap > 5e9:    # 5B+ = mid cap
            score += 10
        else:                     # Small cap
            score += 5
        
        # Marges élevées = pouvoir de fixation des prix (0-25 points)
        gross_margin = ratios['gross_margin']
        operating_margin = ratios['operating_margin']
        
        if gross_margin > 0.6:    # 60%+ = très fort pouvoir de prix
            score += 15
        elif gross_margin > 0.4:  # 40%+ = bon pouvoir
            score += 10
        elif gross_margin > 0.3:  # 30%+ = correct
            score += 5
        
        if operating_margin > 0.25:  # 25%+ = excellente efficacité
            score += 10
        elif operating_margin > 0.15: # 15%+ = bonne efficacité
            score += 7
        elif operating_margin > 0.1:  # 10%+ = correct
            score += 5
        
        # ROE élevé de manière durable = avantage concurrentiel (0-20 points)
        roe = ratios['roe']
        if roe > 0.25:      # 25%+ = exceptionnel
            score += 20
        elif roe > 0.2:     # 20%+ = excellent
            score += 17
        elif roe > 0.15:    # 15%+ = très bon
            score += 15
        elif roe > 0.1:     # 10%+ = bon
            score += 10
        
        # Secteur défensif/avec barrières (0-15 points)
        sector = info.get('sector', '')
        moat_sectors = {
            'Technology': 12,           # Effet de réseau, R&D
            'Healthcare': 10,           # Brevets, régulation
            'Consumer Staples': 8,      # Marques, habitudes
            'Utilities': 12,            # Monopoles naturels
            'Financials': 6,            # Régulation, taille
            'Communication Services': 10, # Effet de réseau
            'Industrial': 5,            # Cyclique
            'Energy': 4,                # Commodité
            'Materials': 3,             # Commodité
            'Real Estate': 7,           # Localisation
            'Consumer Discretionary': 4 # Cyclique, concurrence
        }
        score += moat_sectors.get(sector, 5)
        
        # Analyse IA pour détecter d'autres avantages (0-15 points)
        ai_moat_score = await self._analyze_competitive_moat(symbol, info)
        score += ai_moat_score
        
        return min(100, score)
    
    async def _analyze_competitive_moat(self, symbol: str, info: Dict) -> float:
        """Utilise l'IA pour analyser l'avantage concurrentiel"""
        
        if not self.sentiment_analyzer.gemini_client:
            return 7.5  # Score neutre
        
        business_summary = info.get('longBusinessSummary', '')
        sector = info.get('sector', '')
        industry = info.get('industry', '')
        
        if not business_summary:
            return 5.0
        
        prompt = f"""
        Analysez l'avantage concurrentiel (moat économique) de cette entreprise:
        
        Secteur: {sector}
        Industrie: {industry} 
        Description: {business_summary[:500]}
        
        Évaluez les avantages concurrentiels potentiels:
        - Effet de réseau
        - Économies d'échelle
        - Coûts de changement élevés
        - Actifs intangibles (marques, brevets)
        - Avantages réglementaires
        - Avantages géographiques
        - Technologie propriétaire
        
        Donnez un score de 0 à 15 basé sur la force des avantages concurrentiels.
        
        Répondez uniquement avec le score numérique (exemple: 12.5)
        """
        
        try:
            response = await asyncio.to_thread(
                self.sentiment_analyzer.gemini_client.generate_content, prompt
            )
            
            # Extraire le score numérique
            score_text = response.text.strip()
            score = float(score_text.split()[0])  # Premier nombre trouvé
            return max(0, min(15, score))
            
        except:
            return 7.5  # Score neutre par défaut
    
    def _calculate_price_targets(self, stock_data: Dict, scores: Dict) -> Dict:
        """Calcule les objectifs de prix basés sur l'analyse"""
        
        current_price = stock_data['current_price']
        
        # Estimation basée sur le score global et les multiples sectoriels
        overall_score = scores['overall']
        
        # Multiple de croissance basé sur le score
        if overall_score > 85:
            growth_multiple = 2.5  # Actions exceptionnelles
        elif overall_score > 75:
            growth_multiple = 2.0  # Excellentes actions
        elif overall_score > 65:
            growth_multiple = 1.5  # Bonnes actions
        else:
            growth_multiple = 1.2  # Actions correctes
        
        # Ajustement par les ratios actuels
        pe_current = stock_data['ratios']['pe']
        if pe_current > 25:  # Déjà cher
            growth_multiple *= 0.8
        elif pe_current < 15:  # Bon marché
            growth_multiple *= 1.2
        
        # Objectifs de prix
        target_3y = current_price * growth_multiple
        target_5y = current_price * (growth_multiple * 1.3)  # Croissance supplémentaire
        
        # Calcul des rendements
        return_3y = (target_3y / current_price) ** (1/3) - 1  # Rendement annualisé
        return_5y = (target_5y / current_price) ** (1/5) - 1  # Rendement annualisé
        
        return {
            '3y': target_3y,
            '5y': target_5y,
            'return_3y': return_3y * 100,  # En pourcentage
            'return_5y': return_5y * 100   # En pourcentage
        }
    
    async def _generate_long_term_analysis(self, symbol: str, stock_data: Dict, scores: Dict) -> Dict:
        """Génère une analyse IA complète pour l'investissement long terme"""
        
        if not self.sentiment_analyzer.gemini_client:
            return self._generate_fallback_analysis(stock_data, scores)
        
        info = stock_data['info']
        ratios = stock_data['ratios']
        
        prompt = f"""
        Réalisez une analyse d'investissement long terme (3-5 ans) pour {symbol}:
        
        DONNÉES FINANCIÈRES:
        - Secteur: {info.get('sector', 'Unknown')}
        - Industrie: {info.get('industry', 'Unknown')}
        - Capitalisation: {info.get('marketCap', 0):,.0f}
        - P/E: {ratios['pe']:.1f}
        - PEG: {ratios['peg']:.2f}
        - ROE: {ratios['roe']*100:.1f}%
        - Marge opérationnelle: {ratios['operating_margin']*100:.1f}%
        - Dette/Capitaux: {ratios['debt_equity']:.2f}
        - Croissance revenus: {ratios['revenue_growth']*100:.1f}%
        
        SCORES CALCULÉS:
        - Croissance: {scores['growth']:.0f}/100
        - Qualité: {scores['quality']:.0f}/100
        - Valorisation: {scores['value']:.0f}/100
        - Avantage concurrentiel: {scores['moat']:.0f}/100
        - Score global: {scores['overall']:.0f}/100
        
        Fournissez:
        1. Thèse d'investissement (2-3 phrases)
        2. 3-4 forces clés
        3. 3-4 risques principaux
        4. 2-3 catalyseurs de croissance futurs
        5. Niveau de confiance 0-1
        
        Format JSON strict:
        {{
            "thesis": "Thèse d'investissement...",
            "strengths": ["Force 1", "Force 2", "Force 3"],
            "risks": ["Risque 1", "Risque 2", "Risque 3"],
            "catalysts": ["Catalyseur 1", "Catalyseur 2"],
            "confidence": 0.8
        }}
        """
        
        try:
            response = await asyncio.to_thread(
                self.sentiment_analyzer.gemini_client.generate_content, prompt
            )
            
            # Parser la réponse JSON
            response_text = response.text
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            
            if start != -1 and end != 0:
                return json.loads(response_text[start:end])
                
        except Exception as e:
            logger.debug(f"AI analysis failed for {symbol}: {e}")
        
        return self._generate_fallback_analysis(stock_data, scores)
    
    def _generate_fallback_analysis(self, stock_data: Dict, scores: Dict) -> Dict:
        """Analyse de fallback quand l'IA n'est pas disponible"""
        
        info = stock_data['info']
        ratios = stock_data['ratios']
        sector = info.get('sector', 'Unknown')
        
        # Thèse basique selon le secteur et les scores
        if scores['overall'] > 80:
            thesis = f"Action de qualité supérieure dans le secteur {sector} avec d'excellents fondamentaux."
        elif scores['overall'] > 70:
            thesis = f"Solide opportunité d'investissement avec de bons indicateurs de croissance et qualité."
        else:
            thesis = f"Opportunité d'investissement modérée nécessitant une surveillance attentive."
        
        # Forces génériques basées sur les scores
        strengths = []
        if scores['growth'] > 70:
            strengths.append("Forte croissance historique et perspective")
        if scores['quality'] > 70:
            strengths.append("Excellente qualité financière")
        if scores['value'] > 70:
            strengths.append("Valorisation attractive")
        if scores['moat'] > 70:
            strengths.append("Avantages concurrentiels solides")
        
        if not strengths:
            strengths = ["Fondamentaux stables", "Position sectorielle"]
        
        # Risques génériques
        risks = ["Volatilité des marchés", "Concurrence sectorielle", "Risques macroéconomiques"]
        
        # Catalyseurs génériques
        catalysts = ["Expansion géographique", "Innovation produits"]
        
        return {
            'thesis': thesis,
            'strengths': strengths[:4],
            'risks': risks[:3],
            'catalysts': catalysts[:2],
            'confidence': 0.6
        }
    
    def _get_recommendation(self, overall_score: float) -> str:
        """Convertit le score en recommandation"""
        
        if overall_score > 85:
            return "Strong Buy"
        elif overall_score > 75:
            return "Buy"
        elif overall_score > 65:
            return "Moderate Buy"
        elif overall_score > 50:
            return "Hold"
        else:
            return "Avoid"
    
    def get_sector_recommendations(self, recommendations: List[LongTermRecommendation]) -> Dict[str, List[LongTermRecommendation]]:
        """Groupe les recommandations par secteur"""
        
        sector_recs = {}
        
        for rec in recommendations:
            if rec.sector not in sector_recs:
                sector_recs[rec.sector] = []
            sector_recs[rec.sector].append(rec)
        
        # Trier chaque secteur par score
        for sector in sector_recs:
            sector_recs[sector].sort(key=lambda x: x.overall_score, reverse=True)
        
        return sector_recs
    
    def get_diversified_portfolio(self, recommendations: List[LongTermRecommendation], 
                                 max_stocks: int = 15, max_per_sector: int = 3) -> List[LongTermRecommendation]:
        """Crée un portefeuille diversifié à partir des recommandations"""
        
        sector_recs = self.get_sector_recommendations(recommendations)
        diversified = []
        
        # Prendre le meilleur de chaque secteur d'abord
        for sector, recs in sector_recs.items():
            diversified.extend(recs[:min(max_per_sector, len(recs))])
        
        # Trier par score et prendre les meilleurs
        diversified.sort(key=lambda x: x.overall_score, reverse=True)
        
        return diversified[:max_stocks]