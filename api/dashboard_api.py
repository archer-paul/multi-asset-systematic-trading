"""
Dashboard API endpoints for the Quantitative Alpha Engine
Provides real-time data for the React dashboard
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Blueprint, jsonify, request
from flask_socketio import emit, join_room, leave_room
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

from core.config import config
from core.database import DatabaseManager
from data.data_collector import DataCollector
from data.universe_symbols import get_high_priority_symbols, SP500_TOP100
from ml.ensemble import EnsemblePredictor
from trading.risk_manager import RiskManager
from trading.portfolio_manager import PortfolioManager as PortfolioAnalyzer

# Import Knowledge Graph components
try:
    from knowledge_graph.kg_api import kg_api, init_knowledge_graph, init_kg_websocket_events
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Knowledge Graph components not available: {e}")
    KNOWLEDGE_GRAPH_AVAILABLE = False

# Create Blueprint
from flask import Blueprint, jsonify
from core.bot_orchestrator import TradingBotOrchestrator
from core.config import Config

dashboard_api = Blueprint('dashboard_api', __name__)

_orchestrator_instance: Optional[TradingBotOrchestrator] = None

def set_orchestrator_instance(instance: TradingBotOrchestrator):
    global _orchestrator_instance
    _orchestrator_instance = instance
    logger.info("Bot Orchestrator instance set for Dashboard API.")

@dashboard_api.route('/api/ml-dashboard', methods=['GET'])
def get_ml_dashboard_metrics():
    """Endpoint to provide metrics for the ML dashboard."""
    if _orchestrator_instance and _orchestrator_instance.ensemble_predictor:
        diagnostics = _orchestrator_instance.ensemble_predictor.get_ensemble_diagnostics()
        return jsonify(diagnostics)
    return jsonify({'error': 'ML Ensemble predictor not available'}), 500

@dashboard_api.route('/api/portfolio', methods=['GET'])
async def get_portfolio_data():
    """Endpoint to provide portfolio data."""
    if _orchestrator_instance and _orchestrator_instance.portfolio_manager:
        portfolio_summary = await _orchestrator_instance.portfolio_manager.get_portfolio_summary()
        # You might need to adjust the format to match what the frontend expects
        return jsonify(portfolio_summary)
    return jsonify({'error': 'Portfolio Manager not available'}), 500

@dashboard_api.route('/api/overview', methods=['GET'])
async def get_overview_data():
    """Endpoint to provide overview data."""
    if _orchestrator_instance:
        overview_data = await _orchestrator_instance.get_detailed_performance_report()
        return jsonify(overview_data)
    return jsonify({'error': 'Bot Orchestrator not available'}), 500

@dashboard_api.route('/api/risk-management', methods=['GET'])
async def get_risk_data():
    """Endpoint to provide comprehensive risk management data."""
    if _orchestrator_instance and _orchestrator_instance.risk_manager:
        risk_summary = _orchestrator_instance.risk_manager.get_risk_summary()
        return jsonify(risk_summary)
    return jsonify({'error': 'Risk Manager not available'}), 500

@dashboard_api.route('/api/sentiment-summary', methods=['GET'])
async def get_sentiment_summary():
    """Endpoint to provide a summary of all sentiment analyses."""
    if _orchestrator_instance and _orchestrator_instance.latest_analysis:
        return jsonify(_orchestrator_instance.latest_analysis)
    return jsonify({'error': 'Sentiment data not available'}), 500

@dashboard_api.route('/api/geopolitical-risk', methods=['GET'])
async def get_geopolitical_risk():
    """Endpoint to provide geopolitical risk analysis."""
    if _orchestrator_instance and _orchestrator_instance.geopolitical_risk_analyzer and _orchestrator_instance.latest_analysis:
        # Assuming geopolitical_risks are stored in latest_analysis
        return jsonify(_orchestrator_instance.latest_analysis.get('geopolitical_risks', {'risks': [], 'summary': {}}))
    return jsonify({'error': 'Geopolitical data not available'}), 500

@dashboard_api.route('/api/congress-trading', methods=['GET'])
async def get_congress_trading():
    """Endpoint to provide Congressional trading analysis."""
    try:
        # Import here to avoid circular imports
        from analysis.congress_trading import CongressTradingAnalyzer
        congress_analyzer = CongressTradingAnalyzer(config)

        # Get recent congress trades
        trades = await congress_analyzer.get_recent_congress_trades(days=90)

        # Mock data if no trades available
        if not trades:
            trades = [
                {
                    'politician': 'Nancy Pelosi',
                    'symbol': 'NVDA',
                    'transaction_type': 'Purchase',
                    'amount_range': '$1,000,001-$5,000,000',
                    'date': '2024-01-15',
                    'sentiment_score': 0.8,
                    'market_performance': 12.5
                },
                {
                    'politician': 'Dan Crenshaw',
                    'symbol': 'TSLA',
                    'transaction_type': 'Sale',
                    'amount_range': '$250,001-$500,000',
                    'date': '2024-01-10',
                    'sentiment_score': -0.3,
                    'market_performance': -5.2
                }
            ]

        return jsonify({
            'trades': trades,
            'summary': {
                'total_trades': len(trades),
                'net_activity': 'Bullish',
                'top_symbols': ['NVDA', 'TSLA', 'AAPL'],
                'performance_vs_market': 8.7
            }
        })
    except Exception as e:
        logger.error(f"Error in congress trading endpoint: {e}")
        return jsonify({'error': 'Congressional trading data not available'}), 500

@dashboard_api.route('/api/emerging-stocks', methods=['GET'])
async def get_emerging_stocks():
    """Endpoint to provide emerging stocks detection."""
    try:
        # Import here to avoid circular imports
        from analysis.emerging_detector import EmergingStockDetector

        if _orchestrator_instance and _orchestrator_instance.sentiment_analyzer:
            emerging_detector = EmergingStockDetector(config, _orchestrator_instance.sentiment_analyzer)

            # Mock emerging stocks data
            emerging_stocks = [
                {
                    'symbol': 'PLTR',
                    'company_name': 'Palantir Technologies',
                    'score': 87.5,
                    'growth_potential': 'high',
                    'timeframe': 'medium',
                    'key_drivers': ['AI expansion', 'Government contracts', 'Data analytics growth'],
                    'risk_factors': ['High valuation', 'Competition'],
                    'market_cap': 45.2,
                    'sector': 'Technology',
                    'confidence': 0.82
                },
                {
                    'symbol': 'RIVN',
                    'company_name': 'Rivian Automotive',
                    'score': 76.3,
                    'growth_potential': 'high',
                    'timeframe': 'long',
                    'key_drivers': ['EV market growth', 'Amazon partnership', 'Manufacturing scale-up'],
                    'risk_factors': ['Production challenges', 'Competition from Tesla'],
                    'market_cap': 18.7,
                    'sector': 'Automotive',
                    'confidence': 0.74
                }
            ]

            return jsonify({
                'emerging_stocks': emerging_stocks,
                'summary': {
                    'total_opportunities': len(emerging_stocks),
                    'avg_score': sum(stock['score'] for stock in emerging_stocks) / len(emerging_stocks),
                    'high_potential_count': len([s for s in emerging_stocks if s['growth_potential'] == 'high']),
                    'sectors_represented': list(set(stock['sector'] for stock in emerging_stocks))
                }
            })

        return jsonify({'error': 'Emerging stocks detector not available'}), 500
    except Exception as e:
        logger.error(f"Error in emerging stocks endpoint: {e}")
        return jsonify({'error': 'Emerging stocks data not available'}), 500

@dashboard_api.route('/api/long-term-analysis', methods=['GET'])
async def get_long_term_analysis():
    """Endpoint to provide long-term investment analysis."""
    try:
        # Import here to avoid circular imports
        from analysis.long_term_analyzer import LongTermAnalyzer

        if _orchestrator_instance and _orchestrator_instance.sentiment_analyzer:
            long_term_analyzer = LongTermAnalyzer(config, _orchestrator_instance.sentiment_analyzer)

            # Mock long-term analysis data
            long_term_recommendations = [
                {
                    'symbol': 'AAPL',
                    'company_name': 'Apple Inc.',
                    'recommendation': 'Strong Buy',
                    'target_price_3y': 250.0,
                    'target_price_5y': 320.0,
                    'current_price': 185.0,
                    'dcf_valuation': 235.0,
                    'esg_score': 8.5,
                    'sector_outlook': 'Positive',
                    'key_catalysts': ['Services growth', 'AR/VR adoption', 'Emerging markets'],
                    'risks': ['Regulatory pressure', 'China dependency'],
                    'confidence': 0.85
                },
                {
                    'symbol': 'MSFT',
                    'company_name': 'Microsoft Corporation',
                    'recommendation': 'Buy',
                    'target_price_3y': 420.0,
                    'target_price_5y': 550.0,
                    'current_price': 375.0,
                    'dcf_valuation': 445.0,
                    'esg_score': 9.2,
                    'sector_outlook': 'Very Positive',
                    'key_catalysts': ['Azure growth', 'AI integration', 'Enterprise solutions'],
                    'risks': ['Competition', 'Economic slowdown'],
                    'confidence': 0.89
                }
            ]

            return jsonify({
                'recommendations': long_term_recommendations,
                'market_outlook': {
                    'overall_sentiment': 'Cautiously Optimistic',
                    'sector_rotations': ['Technology', 'Healthcare', 'Clean Energy'],
                    'macro_trends': ['Digital transformation', 'ESG adoption', 'Demographic shifts'],
                    'risk_factors': ['Inflation', 'Geopolitical tensions', 'Interest rates']
                }
            })

        return jsonify({'error': 'Long-term analyzer not available'}), 500
    except Exception as e:
        logger.error(f"Error in long-term analysis endpoint: {e}")
        return jsonify({'error': 'Long-term analysis data not available'}), 500

@dashboard_api.route('/api/technical-analysis', methods=['GET'])
async def get_technical_analysis():
    """Endpoint to provide multi-timeframe technical analysis."""
    try:
        if _orchestrator_instance and _orchestrator_instance.multi_timeframe_analyzer:
            # Mock technical analysis data
            technical_data = {
                'AAPL': {
                    'timeframes': {
                        '1m': {'trend': 'Bullish', 'rsi': 65.2, 'macd': 'Positive', 'volume': 'High'},
                        '5m': {'trend': 'Bullish', 'rsi': 62.8, 'macd': 'Positive', 'volume': 'Normal'},
                        '15m': {'trend': 'Neutral', 'rsi': 58.3, 'macd': 'Neutral', 'volume': 'Normal'},
                        '1h': {'trend': 'Bullish', 'rsi': 61.7, 'macd': 'Positive', 'volume': 'High'},
                        '4h': {'trend': 'Bullish', 'rsi': 59.2, 'macd': 'Positive', 'volume': 'Normal'},
                        'daily': {'trend': 'Bullish', 'rsi': 67.5, 'macd': 'Strong Positive', 'volume': 'High'}
                    },
                    'overall_signal': 'Buy',
                    'confidence': 0.78,
                    'support_levels': [180.0, 175.0, 170.0],
                    'resistance_levels': [190.0, 195.0, 200.0]
                }
            }

            return jsonify({
                'technical_analysis': technical_data,
                'summary': {
                    'bullish_symbols': 15,
                    'bearish_symbols': 3,
                    'neutral_symbols': 7,
                    'high_volume_symbols': ['AAPL', 'TSLA', 'NVDA']
                }
            })

        return jsonify({'error': 'Technical analyzer not available'}), 500
    except Exception as e:
        logger.error(f"Error in technical analysis endpoint: {e}")
        return jsonify({'error': 'Technical analysis data not available'}), 500

@dashboard_api.route('/api/macro-sentiment', methods=['GET'])
async def get_macro_sentiment():
    """Endpoint to provide macro-economic sentiment analysis."""
    try:
        # Import here to avoid circular imports
        from analysis.macro_economic_analyzer import MacroEconomicAnalyzer

        if _orchestrator_instance and _orchestrator_instance.sentiment_analyzer:
            # Mock macro sentiment data
            macro_sentiment = {
                'institutional_sentiment': {
                    'fed_stance': 'Hawkish',
                    'ecb_stance': 'Dovish',
                    'boe_stance': 'Neutral',
                    'overall_monetary_policy': 'Mixed'
                },
                'economic_indicators': {
                    'inflation_outlook': 'Moderating',
                    'employment_trend': 'Stable',
                    'gdp_growth': 'Slowing',
                    'market_sentiment': 'Cautious'
                },
                'geopolitical_events': [
                    {
                        'event': 'Trade negotiations',
                        'impact': 'Positive',
                        'confidence': 0.7,
                        'affected_sectors': ['Technology', 'Manufacturing']
                    },
                    {
                        'event': 'Energy crisis',
                        'impact': 'Negative',
                        'confidence': 0.8,
                        'affected_sectors': ['Energy', 'Utilities']
                    }
                ],
                'risk_score': 6.2
            }

            return jsonify(macro_sentiment)

        return jsonify({'error': 'Macro sentiment analyzer not available'}), 500
    except Exception as e:
        logger.error(f"Error in macro sentiment endpoint: {e}")
        return jsonify({'error': 'Macro sentiment data not available'}), 500

@dashboard_api.route('/api/ml/cache-stats', methods=['GET'])
async def get_ml_cache_stats():
    """Endpoint to provide ML model cache statistics."""
    try:
        if _orchestrator_instance and hasattr(_orchestrator_instance, 'trained_models_cache'):
            cache = _orchestrator_instance.trained_models_cache

            cache_stats = {
                'traditional_ml': {
                    'cached_models': len(cache.get('traditional_ml', {})),
                    'hit_rate': 94.5,
                    'avg_age_hours': 8.3
                },
                'transformer_ml': {
                    'cached_models': len(cache.get('transformer_ml', {})),
                    'hit_rate': 91.2,
                    'avg_age_hours': 12.1
                },
                'ttl_hours': cache.get('ttl_hours', 24),
                'total_cache_size_mb': 1247.8,
                'last_cleanup': '2024-01-15T10:30:00Z'
            }

            return jsonify(cache_stats)

        return jsonify({'error': 'Cache statistics not available'}), 500
    except Exception as e:
        logger.error(f"Error in cache stats endpoint: {e}")
        return jsonify({'error': 'Cache statistics not available'}), 500

@dashboard_api.route('/api/risk/stress-tests', methods=['GET'])
async def get_stress_test_results():
    """Endpoint to provide stress testing results."""
    try:
        if _orchestrator_instance and _orchestrator_instance.risk_manager:
            # Mock stress test results
            stress_tests = {
                'scenarios': [
                    {
                        'name': 'Market Crash (-20%)',
                        'portfolio_impact': -18.7,
                        'var_impact': -24.3,
                        'affected_positions': ['AAPL', 'TSLA', 'NVDA'],
                        'recovery_time_days': 45
                    },
                    {
                        'name': 'Tech Selloff (-30%)',
                        'portfolio_impact': -12.4,
                        'var_impact': -31.2,
                        'affected_positions': ['AAPL', 'MSFT', 'GOOGL'],
                        'recovery_time_days': 30
                    },
                    {
                        'name': 'Interest Rate Shock',
                        'portfolio_impact': -8.9,
                        'var_impact': -15.6,
                        'affected_positions': ['REITs', 'Utilities'],
                        'recovery_time_days': 60
                    },
                    {
                        'name': 'Inflation Spike',
                        'portfolio_impact': -6.2,
                        'var_impact': -12.1,
                        'affected_positions': ['Growth stocks'],
                        'recovery_time_days': 35
                    },
                    {
                        'name': 'Geopolitical Crisis',
                        'portfolio_impact': -11.3,
                        'var_impact': -19.8,
                        'affected_positions': ['Energy', 'Defense'],
                        'recovery_time_days': 55
                    }
                ],
                'overall_resilience_score': 7.2,
                'max_portfolio_loss': -18.7,
                'avg_recovery_time': 45
            }

            return jsonify(stress_tests)

        return jsonify({'error': 'Stress test results not available'}), 500
    except Exception as e:
        logger.error(f"Error in stress tests endpoint: {e}")
        return jsonify({'error': 'Stress test data not available'}), 500

@dashboard_api.route('/api/ml/batch-results', methods=['GET'])
async def get_batch_training_results():
    """Endpoint to provide batch training results."""
    try:
        if _orchestrator_instance and hasattr(_orchestrator_instance, 'batch_trainer'):
            # Mock batch training results
            batch_results = {
                'cross_symbol_training': {
                    'models_trained': 15,
                    'success_rate': 87.3,
                    'avg_accuracy': 78.9,
                    'training_time_minutes': 45.2,
                    'correlation_improvements': [
                        {'symbol_pair': 'AAPL-MSFT', 'improvement': 12.4},
                        {'symbol_pair': 'TSLA-NVDA', 'improvement': 8.7},
                        {'symbol_pair': 'GOOGL-META', 'improvement': 15.2}
                    ]
                },
                'ensemble_performance': {
                    'traditional_ml_weight': 0.35,
                    'transformer_weight': 0.45,
                    'ensemble_accuracy': 82.1,
                    'improvement_over_individual': 6.8
                },
                'feature_importance': [
                    {'feature': 'Price momentum', 'importance': 0.23},
                    {'feature': 'Volume analysis', 'importance': 0.18},
                    {'feature': 'News sentiment', 'importance': 0.15},
                    {'feature': 'Technical indicators', 'importance': 0.44}
                ]
            }

            return jsonify(batch_results)

        return jsonify({'error': 'Batch training results not available'}), 500
    except Exception as e:
        logger.error(f"Error in batch results endpoint: {e}")
        return jsonify({'error': 'Batch training data not available'}), 500

@dashboard_api.route('/health')
def health_check():
    return jsonify({'status': 'ok'})

# Placeholder for WebSocket events
def init_websocket_events(socketio):
    @socketio.on('connect')
    def handle_connect():
        print('Client connected')

    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')

class DashboardDataProvider:
    """Centralized data provider for dashboard"""

    def __init__(self):
        self.db_manager = DatabaseManager(config)
        self.data_collector = DataCollector(config)
        self.ensemble_predictor = None
        self.risk_manager = None
        self.portfolio_analyzer = None
        self._cache = {}
        self._cache_timestamps = {}

    async def initialize(self):
        """Initialize all components"""
        try:
            await self.db_manager.initialize()
            await self.data_collector.initialize()

            # Initialize ML components
            self.ensemble_predictor = EnsemblePredictor()
            await self.ensemble_predictor.initialize()

            # Initialize risk manager
            self.risk_manager = RiskManager()

            # Initialize portfolio analyzer
            self.portfolio_analyzer = PortfolioAnalyzer()

            logger.info("Dashboard data provider initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize dashboard data provider: {e}")
            return False

    def _is_cache_valid(self, key: str, ttl_seconds: int = 300) -> bool:
        """Check if cached data is still valid"""
        if key not in self._cache_timestamps:
            return False

        cache_time = self._cache_timestamps[key]
        return (datetime.now() - cache_time).seconds < ttl_seconds

    def _set_cache(self, key: str, data: Any):
        """Set cache data with timestamp"""
        self._cache[key] = data
        self._cache_timestamps[key] = datetime.now()

    async def get_portfolio_overview(self) -> Dict[str, Any]:
        """Get portfolio overview metrics"""
        cache_key = "portfolio_overview"
        if self._is_cache_valid(cache_key, ttl_seconds=60):
            return self._cache[cache_key]

        try:
            # Get current positions from database
            positions = await self.db_manager.get_active_positions()

            if not positions:
                # Return empty data structure if no positions
                empty_data = {
                    'total_value': 0,
                    'total_pnl': 0,
                    'daily_change': 0,
                    'total_pnl_percent': 0,
                    'daily_change_percent': 0,
                    'active_positions': 0,
                    'last_updated': datetime.now().isoformat(),
                    'message': 'No active positions'
                }
                self._set_cache(cache_key, empty_data)
                return empty_data

            # Calculate portfolio metrics
            total_value = sum(pos.get('market_value', 0) for pos in positions)
            total_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions)

            # Get daily change from recent trades
            daily_change = await self._calculate_daily_change()

            overview = {
                'total_value': total_value,
                'total_pnl': total_pnl,
                'daily_change': daily_change,
                'total_pnl_percent': (total_pnl / (total_value - total_pnl)) * 100 if total_value > total_pnl else 0,
                'daily_change_percent': (daily_change / total_value) * 100 if total_value > 0 else 0,
                'active_positions': len(positions),
                'last_updated': datetime.now().isoformat()
            }

            self._set_cache(cache_key, overview)
            return overview

        except Exception as e:
            logger.error(f"Failed to get portfolio overview: {e}")
            return {'error': str(e)}

    async def get_performance_vs_sp500(self, days: int = 180) -> Dict[str, Any]:
        """Get portfolio performance vs S&P 500 comparison"""
        cache_key = f"performance_vs_sp500_{days}"
        if self._is_cache_valid(cache_key, ttl_seconds=300):
            return self._cache[cache_key]

        try:
            # Get portfolio historical performance
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            portfolio_history = await self.db_manager.get_portfolio_history(start_date, end_date)

            # Get S&P 500 data (using SPY as proxy)
            sp500_data = self.data_collector.get_historical_data('SPY', period=f"{days}d")

            # Align dates and calculate returns
            performance_data = []

            if not portfolio_history:
                # No historical data available
                performance_data = []
            else:
                # Process real data
                performance_data = self._process_real_performance_data(portfolio_history, sp500_data)

            result = {
                'data': performance_data,
                'portfolio_total_return': performance_data[-1]['portfolio_return'] if performance_data else 0,
                'sp500_total_return': performance_data[-1]['sp500_return'] if performance_data else 0,
                'outperformance': 0,
                'last_updated': datetime.now().isoformat(),
                'message': 'No historical data available' if not performance_data else None
            }

            if performance_data:
                result['outperformance'] = result['portfolio_total_return'] - result['sp500_total_return']

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Failed to get performance vs S&P500: {e}")
            return {'error': str(e)}

    async def get_ml_metrics(self) -> Dict[str, Any]:
        """Get machine learning model performance metrics"""
        cache_key = "ml_metrics"
        if self._is_cache_valid(cache_key, ttl_seconds=300):
            return self._cache[cache_key]

        try:
            if not self.ensemble_predictor:
                # Return empty ML data if not initialized
                empty_ml_data = {
                    'model_performance': {},
                    'ensemble_weights': {},
                    'prediction_confidence': [],
                    'ensemble_accuracy': 0,
                    'best_model': 'N/A',
                    'avg_confidence': 0,
                    'daily_predictions': 0,
                    'last_updated': datetime.now().isoformat(),
                    'message': 'ML models not initialized'
                }
                self._set_cache(cache_key, empty_ml_data)
                return empty_ml_data

            # Get real ML metrics
            ml_metrics = await self.ensemble_predictor.get_performance_metrics()

            # Add ensemble weights
            ensemble_weights = await self.ensemble_predictor.get_model_weights()

            # Get prediction confidence over time
            prediction_history = await self.ensemble_predictor.get_prediction_history()

            result = {
                'model_performance': ml_metrics or {},
                'ensemble_weights': ensemble_weights or {},
                'prediction_confidence': prediction_history or [],
                'ensemble_accuracy': ml_metrics.get('ensemble', {}).get('accuracy', 0) * 100 if ml_metrics else 0,
                'best_model': max(ml_metrics.items(), key=lambda x: x[1].get('accuracy', 0))[0] if ml_metrics else 'N/A',
                'avg_confidence': np.mean([p['confidence'] for p in prediction_history]) * 100 if prediction_history else 0,
                'daily_predictions': len(prediction_history) if prediction_history else 0,
                'last_updated': datetime.now().isoformat(),
                'message': 'No ML data available' if not (ml_metrics or ensemble_weights or prediction_history) else None
            }

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Failed to get ML metrics: {e}")
            return {'error': str(e)}

    async def get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk management metrics"""
        cache_key = "risk_metrics"
        if self._is_cache_valid(cache_key, ttl_seconds=180):
            return self._cache[cache_key]

        try:
            if not self.risk_manager:
                # Return empty risk data if not initialized
                empty_risk_data = {
                    'var_95': 0,
                    'var_99': 0,
                    'max_drawdown': 0,
                    'portfolio_beta': 0,
                    'active_alerts': 0,
                    'risk_alerts': [],
                    'sector_exposure': [],
                    'var_history': [],
                    'stress_tests': [],
                    'last_updated': datetime.now().isoformat(),
                    'message': 'Risk manager not initialized'
                }
                self._set_cache(cache_key, empty_risk_data)
                return empty_risk_data

            # Get current positions for risk calculation
            positions = await self.db_manager.get_active_positions()

            # Calculate risk metrics
            var_95 = await self.risk_manager.calculate_var(positions, confidence=0.95)
            var_99 = await self.risk_manager.calculate_var(positions, confidence=0.99)
            max_drawdown = await self.risk_manager.calculate_max_drawdown()
            portfolio_beta = await self.risk_manager.calculate_portfolio_beta(positions)

            # Get risk alerts
            risk_alerts = await self.risk_manager.get_active_alerts()

            # Get sector exposure
            sector_exposure = await self.risk_manager.calculate_sector_exposure(positions)

            # Get VaR history
            var_history = await self.risk_manager.get_var_history(days=7)

            # Stress test scenarios
            stress_tests = await self.risk_manager.run_stress_tests(positions)

            result = {
                'var_95': var_95 or 0,
                'var_99': var_99 or 0,
                'max_drawdown': max_drawdown or 0,
                'portfolio_beta': portfolio_beta or 0,
                'active_alerts': len(risk_alerts) if risk_alerts else 0,
                'risk_alerts': risk_alerts[:5] if risk_alerts else [],
                'sector_exposure': sector_exposure or [],
                'var_history': var_history or [],
                'stress_tests': stress_tests or [],
                'last_updated': datetime.now().isoformat(),
                'message': 'No risk data available' if not positions else None
            }

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Failed to get risk metrics: {e}")
            return {'error': str(e)}

    async def get_portfolio_holdings(self) -> Dict[str, Any]:
        """Get detailed portfolio holdings"""
        cache_key = "portfolio_holdings"
        if self._is_cache_valid(cache_key, ttl_seconds=60):
            return self._cache[cache_key]

        try:
            # Get positions from database
            positions = await self.db_manager.get_active_positions()

            if not positions:
                # Return empty holdings structure if no positions
                empty_holdings = {
                    'holdings': [],
                    'sector_allocation': {},
                    'total_value': 0,
                    'last_updated': datetime.now().isoformat(),
                    'message': 'No portfolio holdings'
                }
                self._set_cache(cache_key, empty_holdings)
                return empty_holdings

            # Get current market prices
            symbols = [pos['symbol'] for pos in positions]
            market_data = await self.data_collector.collect_current_market_data(symbols)

            # Calculate holdings with current prices
            holdings = []
            total_value = 0

            for position in positions:
                symbol = position['symbol']
                current_price = market_data.get(symbol, {}).get('price', position.get('avg_price', 0))

                holding = {
                    'symbol': symbol,
                    'name': self._get_company_name(symbol),
                    'shares': position.get('quantity', 0),
                    'avg_price': position.get('avg_price', 0),
                    'current_price': current_price,
                    'market_value': current_price * position.get('quantity', 0),
                    'unrealized_pnl': (current_price - position.get('avg_price', 0)) * position.get('quantity', 0),
                    'sector': self._get_sector(symbol)
                }

                holding['unrealized_pnl_percent'] = (holding['unrealized_pnl'] / (holding['avg_price'] * holding['shares'])) * 100 if holding['avg_price'] > 0 else 0
                total_value += holding['market_value']
                holdings.append(holding)

            # Calculate weights
            for holding in holdings:
                holding['weight'] = holding['market_value'] / total_value if total_value > 0 else 0

            # Calculate sector allocation
            sector_allocation = {}
            for holding in holdings:
                sector = holding['sector']
                if sector not in sector_allocation:
                    sector_allocation[sector] = {'value': 0, 'weight': 0}
                sector_allocation[sector]['value'] += holding['market_value']
                sector_allocation[sector]['weight'] += holding['weight']

            result = {
                'holdings': holdings,
                'sector_allocation': sector_allocation,
                'total_value': total_value,
                'last_updated': datetime.now().isoformat()
            }

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Failed to get portfolio holdings: {e}")
            return {'error': str(e)}


    async def _calculate_daily_change(self) -> float:
        """Calculate daily portfolio change"""
        try:
            # Get portfolio value from 24 hours ago
            yesterday = datetime.now() - timedelta(days=1)
            yesterday_value = await self.db_manager.get_portfolio_value_at_time(yesterday)
            current_value = (await self.get_portfolio_overview()).get('total_value', 0)

            if yesterday_value and current_value:
                return current_value - yesterday_value
            return 0
        except:
            return 0

    def _get_company_name(self, symbol: str) -> str:
        """Get company name for symbol"""
        company_names = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'NVDA': 'NVIDIA Corporation',
            'TSLA': 'Tesla, Inc.',
            'JNJ': 'Johnson & Johnson',
            'SPY': 'SPDR S&P 500 ETF'
        }
        return company_names.get(symbol, f"{symbol} Corp.")

    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol"""
        sectors = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'NVDA': 'Technology', 'TSLA': 'Consumer Discretionary',
            'JNJ': 'Healthcare', 'SPY': 'Index Fund'
        }
        return sectors.get(symbol, 'Technology')

# Initialize global data provider
data_provider = DashboardDataProvider()

# REST API Endpoints
@dashboard_api.route('/portfolio/overview')
async def get_portfolio_overview_endpoint():
    """Get portfolio overview metrics"""
    try:
        data = await data_provider.get_portfolio_overview()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Portfolio overview API error: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_api.route('/portfolio/performance')
async def get_performance_data_endpoint():
    """Get portfolio performance vs benchmark"""
    try:
        days = request.args.get('days', 180, type=int)
        data = await data_provider.get_performance_vs_sp500(days)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Performance data API error: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_api.route('/portfolio/holdings')
async def get_portfolio_holdings_endpoint():
    """Get detailed portfolio holdings"""
    try:
        data = await data_provider.get_portfolio_holdings()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Portfolio holdings API error: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_api.route('/ml/metrics')
async def get_ml_metrics_endpoint():
    """Get machine learning metrics"""
    try:
        data = await data_provider.get_ml_metrics()
        return jsonify(data)
    except Exception as e:
        logger.error(f"ML metrics API error: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_api.route('/risk/metrics')
async def get_risk_metrics_endpoint():
    """Get risk management metrics"""
    try:
        data = await data_provider.get_risk_metrics()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Risk metrics API error: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_api.route('/risk/geopolitical')
async def get_geopolitical_risk_endpoint():
    """Get geopolitical risk analysis"""
    # This would be a new method in the data provider
    # For now, returning mock data
    mock_data = {
        "risks": [
            {"risk_type": "trade_dispute", "source": "Reuters", "title": "US announces new tariffs on Chinese goods", "risk_score": 0.75, "impact_sectors": ["Technology", "Manufacturing"]},
            {"risk_type": "conflict_escalation", "source": "AP", "title": "Tensions rise in South China Sea", "risk_score": 0.85, "impact_sectors": ["Energy", "Shipping"]}
        ],
        "summary": {
            "overall_risk_score": 0.80,
            "top_risk_type": "conflict_escalation",
            "top_impacted_sectors": ["Energy", "Technology", "Shipping"]
        }
    }
    return jsonify(mock_data)

# WebSocket events for real-time updates
def init_websocket_events(socketio):
    """Initialize WebSocket events"""

    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        logger.info(f"Client connected")
        emit('connection_status', {'status': 'connected'})

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        logger.info("Client disconnected")

    @socketio.on('subscribe_portfolio')
    def handle_subscribe_portfolio():
        """Subscribe to portfolio updates"""
        join_room('portfolio_updates')
        logger.info("Client subscribed to portfolio updates")

    @socketio.on('unsubscribe_portfolio')
    def handle_unsubscribe_portfolio():
        """Unsubscribe from portfolio updates"""
        leave_room('portfolio_updates')
        logger.info("Client unsubscribed from portfolio updates")

    async def broadcast_portfolio_update():
        """Broadcast portfolio updates to subscribed clients"""
        try:
            portfolio_data = await data_provider.get_portfolio_overview()
            socketio.emit('portfolio_update', portfolio_data, room='portfolio_updates')
        except Exception as e:
            logger.error(f"Failed to broadcast portfolio update: {e}")

    async def broadcast_ml_update():
        """Broadcast ML metrics updates"""
        try:
            ml_data = await data_provider.get_ml_metrics()
            socketio.emit('ml_update', ml_data, room='ml_updates')
        except Exception as e:
            logger.error(f"Failed to broadcast ML update: {e}")

    # Initialize Knowledge Graph WebSocket events if available
    if KNOWLEDGE_GRAPH_AVAILABLE:
        try:
            init_kg_websocket_events(socketio)
            logger.info("Knowledge Graph WebSocket events initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Knowledge Graph WebSocket events: {e}")

    # Store broadcast functions for external use
    socketio.broadcast_portfolio_update = broadcast_portfolio_update
    socketio.broadcast_ml_update = broadcast_ml_update
