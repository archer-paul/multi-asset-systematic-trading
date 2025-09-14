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

from core.config import config
from core.database import DatabaseManager
from data.data_collector import DataCollector
from data.universe_symbols import get_high_priority_symbols, SP500_TOP100
from ml.ensemble import EnsemblePredictor
from analysis.risk_manager import RiskManager
from analysis.portfolio_analyzer import PortfolioAnalyzer

logger = logging.getLogger(__name__)

# Create Blueprint
from flask import Blueprint, jsonify
from core.bot_orchestrator import TradingBotOrchestrator
from core.config import Config

dashboard_api = Blueprint('dashboard_api', __name__)

# This is a placeholder. In a real app, you'd have a shared instance of the bot.
# For now, we create one to access its components.
config = Config()
try:
    bot_orchestrator = TradingBotOrchestrator(config)
    # You might need to run parts of the initialization if it's not running
except Exception as e:
    print(f"Could not initialize bot orchestrator for API: {e}")
    bot_orchestrator = None

@dashboard_api.route('/api/ml-dashboard', methods=['GET'])
def get_ml_dashboard_metrics():
    """Endpoint to provide metrics for the ML dashboard."""
    if bot_orchestrator and bot_orchestrator.ensemble_predictor:
        # In a real scenario, you'd ensure the models are trained.
        # For now, we call the diagnostics method.
        # You might need to load a pre-trained model for this to work without training every time.
        diagnostics = bot_orchestrator.ensemble_predictor.get_ensemble_diagnostics()
        return jsonify(diagnostics)
    return jsonify({'error': 'ML Ensemble predictor not available'}), 500

@dashboard_api.route('/api/portfolio', methods=['GET'])
def get_portfolio_data():
    """Endpoint to provide portfolio data."""
    # This is mock data. In a real implementation, you would fetch this
    # from the bot_orchestrator.portfolio_manager
    # from bot_orchestrator.portfolio_manager
    mock_data = {
        "holdings": [
            {
                "symbol": "GOOGL", "name": "Alphabet Inc.", "shares": 50, "avgPrice": 2800.50, 
                "currentPrice": 2850.75, "value": 142537.50, "pnl": 2512.50, "pnlPercent": 1.79, 
                "weight": 0.4, "sector": "Technology"
            },
            {
                "symbol": "TSLA", "name": "Tesla, Inc.", "shares": 100, "avgPrice": 700.00, 
                "currentPrice": 680.50, "value": 68050.00, "pnl": -1950.00, "pnlPercent": -2.78, 
                "weight": 0.2, "sector": "Automotive"
            },
            {
                "symbol": "JPM", "name": "JPMorgan Chase & Co.", "shares": 200, "avgPrice": 150.25, 
                "currentPrice": 155.00, "value": 31000.00, "pnl": 950.00, "pnlPercent": 3.16, 
                "weight": 0.1, "sector": "Financial"
            }
        ],
        "sectorAllocation": [
            { "sector": "Technology", "value": 0.6, "amount": 210587.5, "color": "#3b82f6" },
            { "sector": "Automotive", "value": 0.3, "amount": 68050.00, "color": "#ef4444" },
            { "sector": "Financial", "value": 0.1, "amount": 31000.00, "color": "#f59e0b" }
        ],
        "performanceHistory": [
            { "date": "2025-09-01", "value": 300000, "benchmark": 300000 },
            { "date": "2025-09-02", "value": 301500, "benchmark": 300500 },
            { "date": "2025-09-03", "value": 300800, "benchmark": 301000 },
            { "date": "2025-09-04", "value": 302500, "benchmark": 301500 }
        ]
    }
    return jsonify(mock_data)

@dashboard_api.route('/api/overview', methods=['GET'])
def get_overview_data():
    # Mock data for the main dashboard overview
    mock_data = {
        "metrics": {
            "totalReturn": 15.7, "dailyReturn": -0.25, "sharpeRatio": 1.92,
            "maxDrawdown": -6.8, "winRate": 68.4, "activePositions": 12
        },
        "performanceHistory": [
            # Generate some sample data
        ],
        "recentActivity": [
            { "action": "BUY", "symbol": "NVDA", "quantity": 20, "price": 475.50, "time": "14:30" },
            { "action": "SELL", "symbol": "JPM", "quantity": 100, "price": 155.20, "time": "12:10" },
        ],
        "systemStatus": [
            { "component": "Trading Engine", "status": "online", "uptime": "99.9%" },
            { "component": "ML Models", "status": "online", "uptime": "99.2%" },
        ]
    }
    return jsonify(mock_data)

@dashboard_api.route('/api/risk-management', methods=['GET'])
def get_risk_data():
    """Endpoint to provide comprehensive risk management data."""
    if bot_orchestrator and bot_orchestrator.risk_manager:
        # In a real app, you would pass real portfolio & market data
        # For now, we call the summary method which might use internal state
        risk_summary = bot_orchestrator.risk_manager.get_risk_summary()
        
        # This part would require real data, so we add mock data for now
        # to ensure the frontend has what it needs.
        if 'error' in risk_summary: # If no history, get_risk_summary returns an error
            risk_summary = {
                "metrics": { "portfolioVaR95": -2.3, "maxDrawdown": -7.1, "beta": 1.05, "activeAlerts": 2 },
                "portfolioRisk": [
                    { "date": "2025-09-01", "var95": -2.1, "var99": -3.8, "expectedShortfall": -4.2, "realized": -1.8 },
                ],
                "sectorExposure": [
                    { "sector": "Technology", "exposure": 0.45, "risk": 0.20, "color": "#3b82f6" },
                ],
                "riskAlerts": [
                    { "id": 1, "type": "high", "title": "High Correlation Warning", "description": "NVDA and AMD correlation exceeded 0.90", "time": "5 minutes ago", "action": "Review" },
                ],
                "stressTests": [
                    { "scenario": "Market Crash (-20%)", "portfolioImpact": -15.2, "probability": 0.05 },
                ]
            }

        return jsonify(risk_summary)
    return jsonify({'error': 'Risk Manager not available'}), 500

# Placeholder for WebSocket events
def init_websocket_events(socketio):
    @socketio.on('connect')
    def handle_connect():
        print('Client connected')

    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')

    # Example of how you might push updates
    # def push_portfolio_update(data):
    #     socketio.emit('portfolio_update', data)


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
async def get_portfolio_overview():
    """Get portfolio overview metrics"""
    try:
        data = await data_provider.get_portfolio_overview()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Portfolio overview API error: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_api.route('/portfolio/performance')
async def get_performance_data():
    """Get portfolio performance vs benchmark"""
    try:
        days = request.args.get('days', 180, type=int)
        data = await data_provider.get_performance_vs_sp500(days)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Performance data API error: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_api.route('/portfolio/holdings')
async def get_portfolio_holdings():
    """Get detailed portfolio holdings"""
    try:
        data = await data_provider.get_portfolio_holdings()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Portfolio holdings API error: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_api.route('/ml/metrics')
async def get_ml_metrics():
    """Get machine learning metrics"""
    try:
        data = await data_provider.get_ml_metrics()
        return jsonify(data)
    except Exception as e:
        logger.error(f"ML metrics API error: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_api.route('/risk/metrics')
async def get_risk_metrics():
    """Get risk management metrics"""
    try:
        data = await data_provider.get_risk_metrics()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Risk metrics API error: {e}")
        return jsonify({'error': str(e)}), 500

@dashboard_api.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

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

    # Store broadcast functions for external use
    socketio.broadcast_portfolio_update = broadcast_portfolio_update
    socketio.broadcast_ml_update = broadcast_ml_update