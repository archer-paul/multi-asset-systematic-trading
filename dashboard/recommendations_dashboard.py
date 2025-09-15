"""
Dashboard de recommandations d'investissement
Combine trading court terme et investissement long terme
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class RecommendationsDashboard:
    """Dashboard centralisé pour toutes les recommandations d'investissement"""
    
    def __init__(self, config):
        self.config = config
        self.dashboard_data = {}
        self.last_update = None
        
        # Chemins pour sauvegarder les recommandations
        self.dashboard_dir = Path("dashboard_data")
        self.dashboard_dir.mkdir(exist_ok=True)
    
    async def generate_complete_dashboard(self, short_term_signals: List[Dict] = None,
                                        emerging_opportunities: List = None,
                                        long_term_recommendations: List = None,
                                        congress_signals: List[Dict] = None) -> Dict[str, Any]:
        """Génère un dashboard complet avec toutes les recommandations"""
        
        logger.info("Generating complete investment dashboard...")
        
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'summary': await self._generate_summary(short_term_signals, emerging_opportunities, long_term_recommendations, congress_signals),
            'short_term_trading': self._format_short_term_signals(short_term_signals or []),
            'emerging_opportunities': self._format_emerging_opportunities(emerging_opportunities or []),
            'long_term_investments': self._format_long_term_recommendations(long_term_recommendations or []),
            'congress_trading': self._format_congress_signals(congress_signals or []),
            'portfolio_allocation': self._suggest_portfolio_allocation(short_term_signals, emerging_opportunities, long_term_recommendations, congress_signals),
            'risk_analysis': self._analyze_overall_risk(short_term_signals, emerging_opportunities, long_term_recommendations, congress_signals),
            'market_outlook': await self._generate_market_outlook()
        }
        
        # Sauvegarder le dashboard
        await self._save_dashboard(dashboard)
        
        self.dashboard_data = dashboard
        self.last_update = datetime.now()
        
        logger.info("Dashboard generated successfully")
        return dashboard
    
    async def _generate_summary(self, short_term: List = None, emerging: List = None, long_term: List = None, congress: List = None) -> Dict:
        """Génère un résumé exécutif"""
        
        summary = {
            'total_opportunities': 0,
            'top_recommendation': None,
            'risk_level': 'Medium',
            'expected_returns': {
                'short_term': '5-15%',
                'long_term': '8-12% annualisé'
            },
            'key_insights': []
        }
        
        # Compter les opportunités
        if short_term:
            summary['total_opportunities'] += len(short_term)
        if emerging:
            summary['total_opportunities'] += len(emerging)
        if long_term:
            summary['total_opportunities'] += len(long_term)
        if congress:
            summary['total_opportunities'] += len(congress)
        
        # Identifier la meilleure recommandation
        all_recs = []
        
        if long_term:
            for rec in long_term[:3]:  # Top 3 long terme
                all_recs.append({
                    'symbol': rec.symbol,
                    'type': 'Long Term',
                    'score': rec.overall_score,
                    'expected_return': f"{rec.expected_return_3y:.1f}% (3 ans)",
                    'recommendation': rec.recommendation
                })
        
        if emerging:
            for opp in emerging[:3]:  # Top 3 émergents
                all_recs.append({
                    'symbol': opp.symbol,
                    'type': 'Emerging',
                    'score': opp.score,
                    'expected_return': 'High Growth Potential',
                    'recommendation': 'Speculative Buy'
                })
        
        if short_term:
            for signal in short_term[:3]:  # Top 3 court terme
                all_recs.append({
                    'symbol': signal.get('symbol', ''),
                    'type': 'Short Term',
                    'score': signal.get('confidence', 0) * 100,
                    'expected_return': 'Quick Gain',
                    'recommendation': signal.get('signal_type', '')
                })
        
        # Ajouter les signaux du Congrès
        if congress:
            for signal in congress[:3]:  # Top 3 Congress
                sentiment_desc = "Congressional Buy" if signal.get('sentiment') == 'bullish' else "Congressional Sell" if signal.get('sentiment') == 'bearish' else "Congressional Hold"
                all_recs.append({
                    'symbol': signal.get('symbol', ''),
                    'type': 'Congress',
                    'score': signal.get('trend_score', 0),
                    'expected_return': f"${signal.get('net_activity', 0):,.0f} net activity",
                    'recommendation': sentiment_desc
                })
        
        if all_recs:
            summary['top_recommendation'] = max(all_recs, key=lambda x: x['score'])
        
        # Insights clés
        if len(all_recs) > 10:
            summary['key_insights'].append(f"Marché riche en opportunités: {len(all_recs)} signaux détectés")
        
        if long_term:
            avg_score = sum(rec.overall_score for rec in long_term[:10]) / min(10, len(long_term))
            if avg_score > 75:
                summary['key_insights'].append("Environnement favorable aux investissements long terme")
        
        if emerging:
            high_potential = len([opp for opp in emerging if opp.score > 80])
            if high_potential > 0:
                summary['key_insights'].append(f"{high_potential} actions émergentes à fort potentiel identifiées")
        
        return summary
    
    def _format_short_term_signals(self, signals: List[Dict]) -> Dict:
        """Formate les signaux de trading court terme"""
        
        formatted = {
            'total_signals': len(signals),
            'buy_signals': len([s for s in signals if s.get('signal_type') == 'buy']),
            'sell_signals': len([s for s in signals if s.get('signal_type') == 'sell']),
            'top_signals': []
        }
        
        # Trier par confiance et prendre les meilleurs
        sorted_signals = sorted(signals, key=lambda x: x.get('confidence', 0), reverse=True)
        
        for signal in sorted_signals[:10]:
            formatted['top_signals'].append({
                'symbol': signal.get('symbol', ''),
                'action': signal.get('signal_type', '').upper(),
                'confidence': f"{signal.get('confidence', 0)*100:.0f}%",
                'price': signal.get('price', 0),
                'reasoning': signal.get('reasoning', 'Analyse technique'),
                'timeframe': 'Court terme (1-7 jours)'
            })
        
        return formatted
    
    def _format_emerging_opportunities(self, opportunities: List) -> Dict:
        """Formate les opportunités émergentes"""
        
        formatted = {
            'total_opportunities': len(opportunities),
            'high_potential': len([opp for opp in opportunities if opp.score > 75]),
            'sectors_covered': len(set(opp.sector for opp in opportunities)),
            'top_opportunities': []
        }
        
        for opp in opportunities[:10]:
            formatted['top_opportunities'].append({
                'symbol': opp.symbol,
                'company': opp.company_name,
                'sector': opp.sector,
                'score': f"{opp.score:.0f}/100",
                'growth_potential': opp.growth_potential.title(),
                'timeframe': opp.timeframe.title(),
                'key_drivers': opp.key_drivers[:2],  # Top 2 drivers
                'risk_level': 'High' if opp.score < 70 else 'Medium',
                'confidence': f"{opp.confidence*100:.0f}%"
            })
        
        return formatted
    
    def _format_long_term_recommendations(self, recommendations: List) -> Dict:
        """Formate les recommandations d'investissement long terme"""
        
        if not recommendations:
            return {'total_recommendations': 0, 'sectors': {}, 'top_picks': []}
        
        # Grouper par secteur
        sectors = {}
        for rec in recommendations:
            if rec.sector not in sectors:
                sectors[rec.sector] = []
            sectors[rec.sector].append(rec)
        
        formatted = {
            'total_recommendations': len(recommendations),
            'sectors': {},
            'top_picks': [],
            'diversified_portfolio': []
        }
        
        # Formater par secteur
        for sector, recs in sectors.items():
            formatted['sectors'][sector] = {
                'count': len(recs),
                'avg_score': sum(r.overall_score for r in recs) / len(recs),
                'best_pick': {
                    'symbol': recs[0].symbol,
                    'score': recs[0].overall_score,
                    'expected_return_3y': f"{recs[0].expected_return_3y:.1f}%"
                }
            }
        
        # Top picks globaux
        for rec in recommendations[:15]:
            formatted['top_picks'].append({
                'symbol': rec.symbol,
                'company': rec.company_name,
                'sector': rec.sector,
                'recommendation': rec.recommendation,
                'overall_score': f"{rec.overall_score:.0f}/100",
                'expected_return_3y': f"{rec.expected_return_3y:.1f}%",
                'expected_return_5y': f"{rec.expected_return_5y:.1f}%",
                'target_price_3y': f"${rec.target_price_3y:.2f}",
                'investment_thesis': rec.investment_thesis,
                'key_strengths': rec.key_strengths[:3],
                'main_risks': rec.key_risks[:2],
                'catalysts': rec.catalysts[:2],
                'pe_ratio': rec.pe_ratio,
                'confidence': f"{rec.confidence*100:.0f}%"
            })
        
        # Portfolio diversifié suggéré
        from analysis.long_term_analyzer import LongTermAnalyzer
        analyzer = LongTermAnalyzer(self.config, None)
        diversified = analyzer.get_diversified_portfolio(recommendations, max_stocks=12, max_per_sector=2)
        
        for rec in diversified:
            formatted['diversified_portfolio'].append({
                'symbol': rec.symbol,
                'sector': rec.sector,
                'allocation_suggested': f"{100/len(diversified):.1f}%",
                'score': f"{rec.overall_score:.0f}/100",
                'expected_return': f"{rec.expected_return_3y:.1f}%"
            })
        
        return formatted
    
    def _suggest_portfolio_allocation(self, short_term: List = None, emerging: List = None, long_term: List = None, congress: List = None) -> Dict:
        """Suggère une allocation de portfolio basée sur les opportunités"""
        
        allocation = {
            'conservative': {
                'long_term_quality': 70,  # 70% actions de qualité long terme
                'emerging_growth': 15,    # 15% actions émergentes
                'short_term_trading': 10, # 10% trading court terme
                'cash_reserve': 5         # 5% liquidités
            },
            'balanced': {
                'long_term_quality': 60,
                'emerging_growth': 25,
                'short_term_trading': 15,
                'cash_reserve': 0
            },
            'aggressive': {
                'long_term_quality': 40,
                'emerging_growth': 35,
                'short_term_trading': 25,
                'cash_reserve': 0
            }
        }
        
        # Recommandation basée sur les opportunités disponibles
        recommendation = 'balanced'
        
        if long_term and len([r for r in long_term if r.overall_score > 80]) > 5:
            recommendation = 'conservative'  # Beaucoup de qualité disponible
        elif emerging and len([e for e in emerging if e.score > 75]) > 10:
            recommendation = 'aggressive'   # Beaucoup d'opportunités émergentes
        
        return {
            'recommended_profile': recommendation,
            'allocations': allocation,
            'reasoning': self._get_allocation_reasoning(recommendation, short_term, emerging, long_term)
        }
    
    def _get_allocation_reasoning(self, profile: str, short_term: List, emerging: List, long_term: List) -> str:
        """Explique le raisonnement derrière l'allocation recommandée"""
        
        if profile == 'conservative':
            return "Environnement riche en actions de qualité. Privilégier la sécurité avec du potentiel de croissance."
        elif profile == 'aggressive':
            return "Nombreuses opportunités émergentes détectées. Moment favorable pour prendre plus de risques."
        else:
            return "Approche équilibrée recommandée compte tenu du mix d'opportunités disponibles."
    
    def _analyze_overall_risk(self, short_term: List = None, emerging: List = None, long_term: List = None, congress: List = None) -> Dict:
        """Analyse le risque global des recommandations"""
        
        risk_analysis = {
            'overall_risk': 'Medium',
            'risk_factors': [],
            'risk_mitigation': [],
            'volatility_expectation': 'Moderate',
            'time_horizon_risk': {}
        }
        
        # Analyser les risques par horizon
        if short_term:
            risk_analysis['time_horizon_risk']['short_term'] = {
                'level': 'High',
                'description': 'Trading actif avec risque de pertes rapides mais potentiel de gains immédiats'
            }
        
        if emerging:
            high_risk_emerging = len([e for e in emerging if e.score < 70])
            risk_level = 'High' if high_risk_emerging > len(emerging) * 0.3 else 'Medium'
            risk_analysis['time_horizon_risk']['emerging'] = {
                'level': risk_level,
                'description': 'Actions émergentes volatiles avec fort potentiel mais risque d\'échec élevé'
            }
        
        if long_term:
            quality_stocks = len([l for l in long_term if l.overall_score > 75])
            risk_level = 'Low' if quality_stocks > len(long_term) * 0.6 else 'Medium'
            risk_analysis['time_horizon_risk']['long_term'] = {
                'level': risk_level,
                'description': 'Investissements de qualité avec risque réduit sur le long terme'
            }
        
        # Facteurs de risque généraux
        risk_analysis['risk_factors'] = [
            "Volatilité des marchés financiers",
            "Risques géopolitiques et macroéconomiques",
            "Risque de concentration sectorielle",
            "Risque de liquidité sur certaines positions"
        ]
        
        # Recommandations de mitigation
        risk_analysis['risk_mitigation'] = [
            "Diversification géographique et sectorielle",
            "Étalement des entrées dans le temps (DCA)",
            "Définition de stops loss sur les positions courtes",
            "Revue régulière et rééquilibrage du portfolio"
        ]
        
        return risk_analysis
    
    async def _generate_market_outlook(self) -> Dict:
        """Génère une perspective de marché"""
        
        # Analyse basique - pourrait être enrichie avec des données macro
        outlook = {
            'general_sentiment': 'Neutral to Positive',
            'key_themes': [
                'Transformation digitale continue',
                'Transition énergétique et ESG',
                'Innovation en santé et biotechnologie',
                'Croissance des marchés émergents'
            ],
            'sectors_to_watch': [
                'Technology - IA et cloud computing',
                'Healthcare - Biotechnologies et dispositifs médicaux',
                'Clean Energy - Énergies renouvelables',
                'Financial Services - Fintech et digitalisation'
            ],
            'potential_headwinds': [
                'Inflation et politiques monétaires',
                'Tensions géopolitiques',
                'Régulation accrue du secteur tech',
                'Risques de récession'
            ],
            'investment_strategy': 'Focus sur la qualité avec exposition sélective à la croissance'
        }
        
        return outlook
    
    async def _save_dashboard(self, dashboard: Dict):
        """Sauvegarde le dashboard sur disque"""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Sauvegarder la version complète
            full_path = self.dashboard_dir / f"dashboard_complete_{timestamp}.json"
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(dashboard, f, indent=2, default=str, ensure_ascii=False)
            
            # Sauvegarder une version résumé
            summary_path = self.dashboard_dir / "dashboard_latest.json"
            summary = {
                'timestamp': dashboard['timestamp'],
                'summary': dashboard['summary'],
                'top_short_term': dashboard['short_term_trading']['top_signals'][:5],
                'top_emerging': dashboard['emerging_opportunities']['top_opportunities'][:5],
                'top_long_term': dashboard['long_term_investments']['top_picks'][:10],
                'portfolio_allocation': dashboard['portfolio_allocation']
            }
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Dashboard saved to {full_path}")
            
        except Exception as e:
            logger.error(f"Error saving dashboard: {e}")
    
    def get_dashboard_summary_text(self) -> str:
        """Retourne un résumé textuel du dashboard"""
        
        if not self.dashboard_data:
            return "Dashboard not yet generated"
        
        summary = self.dashboard_data['summary']
        short_term = self.dashboard_data['short_term_trading']
        long_term = self.dashboard_data['long_term_investments']
        emerging = self.dashboard_data['emerging_opportunities']
        
        text = f"""
=== DASHBOARD D'INVESTISSEMENT ===
Dernière mise à jour: {self.last_update.strftime('%Y-%m-%d %H:%M')}

RÉSUMÉ EXÉCUTIF:
• Total opportunités détectées: {summary['total_opportunities']}
• Recommandation principale: {summary['top_recommendation']['symbol'] if summary['top_recommendation'] else 'N/A'}
• Niveau de risque: {summary['risk_level']}

TRADING COURT TERME:
• Signaux actifs: {short_term['total_signals']}
• Signaux d'achat: {short_term['buy_signals']}
• Signaux de vente: {short_term['sell_signals']}

OPPORTUNITÉS ÉMERGENTES:
• Actions émergentes identifiées: {emerging['total_opportunities']}
• À fort potentiel: {emerging['high_potential']}
• Secteurs couverts: {emerging['sectors_covered']}

INVESTISSEMENT LONG TERME:
• Recommandations qualité: {long_term['total_recommendations']}
• Secteurs analysés: {len(long_term['sectors'])}
• Portfolio diversifié: {len(long_term.get('diversified_portfolio', []))} positions

TRANSACTIONS DU CONGRÈS:
• Signaux détectés: {self.dashboard_data['congress_trading']['total_signals']}
• Signaux bullish: {self.dashboard_data['congress_trading']['bullish_signals']}
• Signaux bearish: {self.dashboard_data['congress_trading']['bearish_signals']}

ALLOCATION RECOMMANDÉE: {self.dashboard_data['portfolio_allocation']['recommended_profile'].upper()}
        """
        
        return text.strip()
    
    def export_to_csv(self, output_dir: str = "exports"):
        """Exporte les recommandations en CSV"""
        
        import pandas as pd
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not self.dashboard_data:
            logger.warning("No dashboard data to export")
            return
        
        try:
            # Export long term recommendations
            if self.dashboard_data['long_term_investments']['top_picks']:
                df_long = pd.DataFrame(self.dashboard_data['long_term_investments']['top_picks'])
                df_long.to_csv(output_path / f"long_term_recommendations_{timestamp}.csv", index=False)
            
            # Export emerging opportunities
            if self.dashboard_data['emerging_opportunities']['top_opportunities']:
                df_emerging = pd.DataFrame(self.dashboard_data['emerging_opportunities']['top_opportunities'])
                df_emerging.to_csv(output_path / f"emerging_opportunities_{timestamp}.csv", index=False)
            
            # Export short term signals
            if self.dashboard_data['short_term_trading']['top_signals']:
                df_short = pd.DataFrame(self.dashboard_data['short_term_trading']['top_signals'])
                df_short.to_csv(output_path / f"short_term_signals_{timestamp}.csv", index=False)
            
            logger.info(f"Dashboard data exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting dashboard data: {e}")
    
    def _format_congress_signals(self, congress_signals: List[Dict]) -> Dict[str, Any]:
        """Formate les signaux du Congrès pour le dashboard"""
        
        if not congress_signals:
            return {
                'total_signals': 0,
                'bullish_signals': 0,
                'bearish_signals': 0,
                'top_signals': [],
                'summary': 'Aucune activité significative du Congrès détectée'
            }
        
        bullish_count = len([s for s in congress_signals if s.get('sentiment') == 'bullish'])
        bearish_count = len([s for s in congress_signals if s.get('sentiment') == 'bearish'])
        
        # Top signaux avec plus de détails
        top_signals = []
        for signal in congress_signals[:10]:
            top_signals.append({
                'symbol': signal['symbol'],
                'sentiment': signal['sentiment'],
                'net_activity': signal['net_activity'],
                'representatives_count': signal['representatives_count'],
                'total_value': signal['total_value'],
                'trend_score': signal.get('trend_score', 0)
            })
        
        return {
            'total_signals': len(congress_signals),
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count,
            'top_signals': top_signals,
            'summary': f"{len(congress_signals)} signaux du Congrès détectés ({bullish_count} bullish, {bearish_count} bearish)"
        }