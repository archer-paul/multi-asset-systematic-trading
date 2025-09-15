'''
Geopolitical Risk Analyzer for Trading Bot

This module analyzes macro-economic data to identify, score, and categorize
geopolitical risks that could impact financial markets.
'''

import logging
from typing import Dict, List, Any
from datetime import datetime

from core.config import Config

logger = logging.getLogger(__name__)

class GeopoliticalRiskAnalyzer:
    """Analyzes geopolitical events and their potential market impact."""

    def __init__(self, config: Config):
        self.config = config
        # Define risk factors with keywords and impact scores
        self.risk_factors = {
            'conflict_escalation': {
                'keywords': ['war', 'conflict', 'invasion', 'military', 'attack', 'escalation', 'hostilities'],
                'base_risk_score': 0.8,
                'impact_sectors': ['Energy', 'Defense', 'Commodities']
            },
            'trade_dispute': {
                'keywords': ['tariff', 'trade war', 'sanctions', 'embargo', 'trade dispute'],
                'base_risk_score': 0.6,
                'impact_sectors': ['Technology', 'Manufacturing', 'Agriculture']
            },
            'elections_instability': {
                'keywords': ['election', 'protest', 'unrest', 'coup', 'political instability'],
                'base_risk_score': 0.5,
                'impact_sectors': ['Financials', 'Consumer Discretionary']
            },
            'central_bank_policy': {
                'keywords': ['rate hike', 'interest rate', 'monetary policy', 'inflation', 'quantitative easing'],
                'base_risk_score': 0.7,
                'impact_sectors': ['Financials', 'Real Estate', 'Technology']
            }
        }

    def analyze_geopolitical_risks(self, macro_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes a collection of macro-economic articles to identify and score geopolitical risks.

        Args:
            macro_analysis: The output from MacroEconomicAnalyzer's get_full_macro_analysis.

        Returns:
            A dictionary containing a list of identified risks and a summary.
        """
        identified_risks = []
        if 'articles' not in macro_analysis:
            return {'risks': [], 'summary': {}}

        for article in macro_analysis['articles']:
            text_to_analyze = f"{article.get('title', '')} {article.get('summary', '')}".lower()
            
            for risk_name, risk_data in self.risk_factors.items():
                if any(keyword in text_to_analyze for keyword in risk_data['keywords']):
                    risk_score = self._calculate_risk_score(article, risk_data)
                    
                    risk_event = {
                        'risk_type': risk_name,
                        'source': article.get('source'),
                        'title': article.get('title'),
                        'link': article.get('link'),
                        'risk_score': risk_score,
                        'impact_sectors': risk_data['impact_sectors'],
                        'timestamp': datetime.now().isoformat()
                    }
                    identified_risks.append(risk_event)
        
        risk_summary = self._summarize_risks(identified_risks)

        logger.info(f"Identified {len(identified_risks)} geopolitical risk events. Overall risk score: {risk_summary.get('overall_risk_score', 0):.2f}")

        return {
            'risks': identified_risks,
            'summary': risk_summary
        }

    def _calculate_risk_score(self, article: Dict[str, Any], risk_data: Dict[str, Any]) -> float:
        """Calculates a score for a given risk based on article analysis."""
        base_score = risk_data['base_risk_score']
        analysis = article.get('analysis', {})

        if not isinstance(analysis, dict): return base_score

        # Adjust score based on AI analysis
        if analysis.get('Geopolitical Risk') == 'Elevating':
            base_score *= 1.2
        elif analysis.get('Geopolitical Risk') == 'De-escalating':
            base_score *= 0.5

        if analysis.get('Market Impact') == 'High':
            base_score *= 1.2
        elif analysis.get('Market Impact') == 'Low':
            base_score *= 0.8

        return min(1.0, base_score) # Cap score at 1.0

    def _summarize_risks(self, identified_risks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Creates a summary of all identified risks."""
        if not identified_risks:
            return {
                'overall_risk_score': 0.0,
                'risk_count': 0,
                'top_risk_type': 'None',
                'top_impacted_sectors': []
            }

        overall_risk_score = sum(r['risk_score'] for r in identified_risks) / len(identified_risks)

        risk_type_counts = {}
        sector_impact_counts = {}
        for risk in identified_risks:
            risk_type_counts[risk['risk_type']] = risk_type_counts.get(risk['risk_type'], 0) + 1
            for sector in risk['impact_sectors']:
                sector_impact_counts[sector] = sector_impact_counts.get(sector, 0) + 1

        top_risk_type = max(risk_type_counts, key=risk_type_counts.get) if risk_type_counts else 'None'
        top_sectors = sorted(sector_impact_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            'overall_risk_score': overall_risk_score,
            'risk_count': len(identified_risks),
            'top_risk_type': top_risk_type,
            'top_impacted_sectors': [sector for sector, count in top_sectors[:3]]
        }
