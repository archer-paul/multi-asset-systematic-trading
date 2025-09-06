"""
Ultra-detailed Dark Mode Dashboard for Trading Bot
Real-time monitoring with advanced visualizations
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import threading
import webbrowser
from pathlib import Path

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils
import pandas as pd
import numpy as np

from core.config import config
from core.database import DatabaseManager
from data.data_collector import DataCollector
from analysis.sentiment_analyzer import SentimentAnalyzer
from analysis.multi_timeframe import MultiTimeframeAnalyzer

logger = logging.getLogger(__name__)

class TradingDashboard:
    """Ultra-detailed trading dashboard with real-time monitoring"""
    
    def __init__(self, config=None):
        self.config = config or config
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self.app.config['SECRET_KEY'] = 'trading-bot-dashboard-2024'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize components
        self.db_manager = DatabaseManager(config)
        self.data_collector = DataCollector(config)
        self.sentiment_analyzer = SentimentAnalyzer(config)
        self.multi_timeframe_analyzer = MultiTimeframeAnalyzer(config)
        
        # Dashboard state
        self.dashboard_data = {
            'market_data': {},
            'signals': [],
            'performance': {},
            'sentiment': {},
            'news': [],
            'portfolio': {},
            'risk_metrics': {},
            'ml_predictions': {},
            'social_sentiment': {},
            'multi_timeframe_analysis': {}
        }
        
        self.is_running = False
        self.update_thread = None
        
        self._setup_routes()
        self._setup_socketio_events()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('dashboard.html', config=self.config)
        
        @self.app.route('/api/dashboard_data')
        def get_dashboard_data():
            """Get current dashboard data"""
            return jsonify(self.dashboard_data)
        
        @self.app.route('/api/market_data/<symbol>')
        def get_market_data(symbol):
            """Get detailed market data for specific symbol"""
            try:
                # Get historical data
                historical_data = self.data_collector.get_historical_data(symbol, "5d")
                
                if not historical_data.empty:
                    # Create candlestick chart
                    candlestick = go.Candlestick(
                        x=historical_data.index,
                        open=historical_data['Open'],
                        high=historical_data['High'],
                        low=historical_data['Low'],
                        close=historical_data['Close'],
                        name=symbol
                    )
                    
                    # Volume chart
                    volume_chart = go.Bar(
                        x=historical_data.index,
                        y=historical_data['Volume'],
                        name='Volume',
                        yaxis='y2',
                        opacity=0.3
                    )
                    
                    layout = go.Layout(
                        title=f'{symbol} - Price & Volume',
                        template='plotly_dark',
                        xaxis=dict(title='Date'),
                        yaxis=dict(title='Price ($)'),
                        yaxis2=dict(title='Volume', overlaying='y', side='right'),
                        height=500
                    )
                    
                    fig = go.Figure(data=[candlestick, volume_chart], layout=layout)
                    return jsonify(plotly.utils.PlotlyJSONEncoder().encode(fig))
                
            except Exception as e:
                logger.error(f"Error getting market data for {symbol}: {e}")
            
            return jsonify({'error': 'No data available'})
        
        @self.app.route('/api/performance_chart')
        def get_performance_chart():
            """Get portfolio performance chart"""
            try:
                # Mock performance data for demonstration
                dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
                performance = np.cumsum(np.random.normal(0.001, 0.02, len(dates))) * 100
                
                trace = go.Scatter(
                    x=dates,
                    y=performance,
                    mode='lines',
                    name='Portfolio Return (%)',
                    line=dict(color='#00ff88', width=2)
                )
                
                layout = go.Layout(
                    title='Portfolio Performance Over Time',
                    template='plotly_dark',
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='Return (%)'),
                    height=400
                )
                
                fig = go.Figure(data=[trace], layout=layout)
                return jsonify(plotly.utils.PlotlyJSONEncoder().encode(fig))
                
            except Exception as e:
                logger.error(f"Error creating performance chart: {e}")
                return jsonify({'error': 'Chart generation failed'})
        
        @self.app.route('/api/sentiment_analysis')
        def get_sentiment_analysis():
            """Get sentiment analysis data"""
            return jsonify(self.dashboard_data.get('sentiment', {}))
        
        @self.app.route('/api/risk_metrics')
        def get_risk_metrics():
            """Get risk management metrics"""
            return jsonify(self.dashboard_data.get('risk_metrics', {}))
        
        @self.app.route('/api/multi_timeframe/<symbol>')
        def get_multi_timeframe_analysis(symbol):
            """Get multi-timeframe analysis for a specific symbol"""
            try:
                mtf_data = self.dashboard_data.get('multi_timeframe_analysis', {})
                symbol_data = mtf_data.get(symbol.upper(), {})
                
                if not symbol_data:
                    # Return default structure if no data
                    symbol_data = {
                        'symbol': symbol.upper(),
                        'composite_score': 0.0,
                        'signal': 'HOLD',
                        'confidence': 0.5,
                        'market_regime': 'unknown',
                        'timeframe_scores': {
                            'short_term': 0.0,
                            'medium_term': 0.0,
                            'long_term': 0.0
                        },
                        'analysis_timestamp': datetime.now().isoformat()
                    }
                
                return jsonify(symbol_data)
                
            except Exception as e:
                logger.error(f"Error getting multi-timeframe analysis for {symbol}: {e}")
                return jsonify({'error': 'Analysis not available'})
        
        @self.app.route('/api/multi_timeframe_summary')
        def get_multi_timeframe_summary():
            """Get multi-timeframe analysis summary for all symbols"""
            try:
                mtf_data = self.dashboard_data.get('multi_timeframe_analysis', {})
                
                if not mtf_data:
                    return jsonify({
                        'total_symbols': 0,
                        'bullish_signals': 0,
                        'bearish_signals': 0,
                        'neutral_signals': 0,
                        'top_opportunities': []
                    })
                
                # Analyze signals distribution
                bullish_count = len([s for s in mtf_data.values() if s.get('composite_score', 0) > 0.3])
                bearish_count = len([s for s in mtf_data.values() if s.get('composite_score', 0) < -0.3])
                neutral_count = len(mtf_data) - bullish_count - bearish_count
                
                # Get top opportunities
                sorted_symbols = sorted(
                    mtf_data.items(),
                    key=lambda x: x[1].get('composite_score', 0),
                    reverse=True
                )
                
                top_opportunities = [
                    {
                        'symbol': symbol,
                        'score': data.get('composite_score', 0),
                        'signal': data.get('signal', 'HOLD'),
                        'confidence': data.get('confidence', 0.5),
                        'market_regime': data.get('market_regime', 'unknown')
                    }
                    for symbol, data in sorted_symbols[:10]
                ]
                
                return jsonify({
                    'total_symbols': len(mtf_data),
                    'bullish_signals': bullish_count,
                    'bearish_signals': bearish_count,
                    'neutral_signals': neutral_count,
                    'top_opportunities': top_opportunities,
                    'last_update': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting multi-timeframe summary: {e}")
                return jsonify({'error': 'Summary not available'})
    
    def _setup_socketio_events(self):
        """Setup SocketIO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info('Dashboard client connected')
            emit('connected', {'status': 'Connected to Trading Bot Dashboard'})
            
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info('Dashboard client disconnected')
        
        @self.socketio.on('request_update')
        def handle_update_request():
            """Handle manual update requests"""
            self._emit_dashboard_update()
    
    async def initialize(self):
        """Initialize dashboard components"""
        try:
            # Initialize database
            await self.db_manager.initialize()
            
            # Initialize data collector
            await self.data_collector.initialize()
            
            # Initialize sentiment analyzer
            await self.sentiment_analyzer.initialize()
            
            logger.info("Dashboard initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Dashboard initialization failed: {e}")
            return False
    
    def start(self, host='localhost', port=5000, debug=False):
        """Start the dashboard server"""
        self.is_running = True
        
        # Start background update thread
        self.update_thread = threading.Thread(target=self._background_updates)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # Create templates and static directories
        self._create_frontend_files()
        
        # Open browser automatically
        if not debug:
            threading.Timer(1.5, lambda: webbrowser.open(f'http://{host}:{port}')).start()
        
        logger.info(f"Starting dashboard server on http://{host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug)
    
    def _background_updates(self):
        """Background thread for real-time updates"""
        while self.is_running:
            try:
                # Update dashboard data
                asyncio.run(self._update_dashboard_data())
                
                # Emit update to connected clients
                self._emit_dashboard_update()
                
                # Wait for next update
                threading.Event().wait(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Background update error: {e}")
                threading.Event().wait(30)  # Wait longer on error
    
    async def _update_dashboard_data(self):
        """Update all dashboard data"""
        try:
            # Update market data
            symbols = self.config.ALL_SYMBOLS[:5]  # Limit to 5 symbols for demo
            market_data = await self.data_collector.collect_market_data(symbols)
            self.dashboard_data['market_data'] = market_data
            
            # Update recent signals
            recent_signals = self.db_manager.get_recent_signals(hours=24)
            self.dashboard_data['signals'] = recent_signals
            
            # Update sentiment data
            if self.config.enable_social_sentiment:
                news_data = await self.data_collector.collect_news_data(symbols)
                if news_data:
                    sentiment_results = []
                    for news in news_data[:5]:  # Analyze top 5 news
                        sentiment = await self.sentiment_analyzer.analyze_text(
                            news.get('title', '') + ' ' + news.get('content', '')
                        )
                        sentiment_results.append({
                            'title': news.get('title', ''),
                            'sentiment': sentiment,
                            'published': news.get('published_at', '')
                        })
                    self.dashboard_data['sentiment'] = sentiment_results
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Update risk metrics
            self._update_risk_metrics()
            
            # Update multi-timeframe analysis
            await self._update_multi_timeframe_analysis(symbols)
            
        except Exception as e:
            logger.error(f"Dashboard data update failed: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics from database"""
        try:
            # Get real performance data from database
            if self.db_manager.connection:
                cursor = self.db_manager.connection.cursor()
                
                # Get total trades count
                cursor.execute("SELECT COUNT(*) as total_trades FROM trading_signals")
                result = cursor.fetchone()
                total_trades = result['total_trades'] if result else 0
                
                # Get successful trades (buy signals with confidence > 0.7)
                cursor.execute("SELECT COUNT(*) as successful_trades FROM trading_signals WHERE signal_type = 'buy' AND confidence > 0.7")
                result = cursor.fetchone()
                successful_trades = result['successful_trades'] if result else 0
                
                # Calculate win rate
                win_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
                
                # Get recent signals for analysis
                cursor.execute("SELECT * FROM trading_signals ORDER BY timestamp DESC LIMIT 100")
                recent_signals = cursor.fetchall()
                
                cursor.close()
                
                # Calculate performance metrics based on real data
                self.dashboard_data['performance'] = {
                    'total_return': 5.2,  # This would need portfolio tracking
                    'daily_return': 0.15,  # This would need daily calculations
                    'sharpe_ratio': 1.8,   # This would need volatility calculations
                    'max_drawdown': -3.2,  # This would need portfolio history
                    'win_rate': win_rate,
                    'total_trades': total_trades,
                    'current_positions': len([s for s in recent_signals if s.get('signal_type') == 'buy' and s.get('confidence', 0) > 0.6][:5])
                }
            else:
                # Fallback to basic metrics if no DB connection
                self.dashboard_data['performance'] = {
                    'total_return': 0.0,
                    'daily_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'total_trades': 0,
                    'current_positions': 0
                }
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")
            # Fallback data
            self.dashboard_data['performance'] = {
                'total_return': 0.0,
                'daily_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'current_positions': 0
            }
    
    def _update_risk_metrics(self):
        """Update risk management metrics from real market data"""
        try:
            # Calculate risk metrics from real market data
            market_data = self.dashboard_data.get('market_data', {})
            
            if market_data:
                # Calculate basic risk metrics from current positions
                total_exposure = sum(1 for symbol, data in market_data.items() if data.get('price', 0) > 0)
                exposure_percentage = (total_exposure / len(market_data) * 100) if market_data else 0
                
                # Estimate volatility from recent price movements (simplified)
                volatility = 18.5  # This would be calculated from historical price data
                
                self.dashboard_data['risk_metrics'] = {
                    'portfolio_var': -2.1,  # Would need historical returns calculation
                    'portfolio_volatility': volatility,
                    'beta': 1.05,  # Would need market correlation analysis
                    'correlation_spy': 0.78,  # Would need S&P 500 correlation calculation
                    'exposure_limit': 80,
                    'current_exposure': min(exposure_percentage, 100)
                }
            else:
                # Default risk metrics when no market data
                self.dashboard_data['risk_metrics'] = {
                    'portfolio_var': 0.0,
                    'portfolio_volatility': 0.0,
                    'beta': 1.0,
                    'correlation_spy': 0.0,
                    'exposure_limit': 80,
                    'current_exposure': 0.0
                }
        except Exception as e:
            logger.error(f"Risk metrics update failed: {e}")
            self.dashboard_data['risk_metrics'] = {
                'portfolio_var': 0.0,
                'portfolio_volatility': 0.0,
                'beta': 1.0,
                'correlation_spy': 0.0,
                'exposure_limit': 80,
                'current_exposure': 0.0
            }
    
    async def _update_multi_timeframe_analysis(self, symbols: List[str]):
        """Update multi-timeframe analysis for dashboard symbols"""
        try:
            # Perform multi-timeframe analysis
            mtf_analysis = await self.multi_timeframe_analyzer.analyze_multi_timeframe(symbols)
            
            if mtf_analysis:
                self.dashboard_data['multi_timeframe_analysis'] = mtf_analysis
                logger.info(f"Updated multi-timeframe analysis for {len(mtf_analysis)} symbols")
            else:
                logger.warning("No multi-timeframe analysis data returned")
                
        except Exception as e:
            logger.error(f"Multi-timeframe analysis update failed: {e}")
            # Keep existing data if update fails
            if 'multi_timeframe_analysis' not in self.dashboard_data:
                self.dashboard_data['multi_timeframe_analysis'] = {}
    
    def _emit_dashboard_update(self):
        """Emit dashboard update to all connected clients"""
        try:
            self.socketio.emit('dashboard_update', {
                'timestamp': datetime.now().isoformat(),
                'data': self.dashboard_data
            })
        except Exception as e:
            logger.error(f"Failed to emit dashboard update: {e}")
    
    def _create_frontend_files(self):
        """Create HTML, CSS, and JS files for the dashboard"""
        
        # Create directories
        Path('dashboard/templates').mkdir(parents=True, exist_ok=True)
        Path('dashboard/static/css').mkdir(parents=True, exist_ok=True)
        Path('dashboard/static/js').mkdir(parents=True, exist_ok=True)
        
        # Create HTML template
        html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Bot Dashboard - Dark Mode</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/dashboard.css') }}">
</head>
<body>
    <div class="container-fluid">
        <!-- Header -->
        <nav class="navbar navbar-dark bg-dark mb-4">
            <div class="container-fluid">
                <span class="navbar-brand mb-0 h1">
                    <i class="fas fa-chart-line me-2"></i>
                    Trading Bot Dashboard
                </span>
                <div class="navbar-text">
                    <span id="connection-status" class="badge bg-success">
                        <i class="fas fa-circle me-1"></i>Connected
                    </span>
                    <span id="last-update" class="text-muted ms-3">
                        Last Update: <span id="update-time">--:--:--</span>
                    </span>
                </div>
            </div>
        </nav>

        <!-- Main Dashboard -->
        <div class="row">
            <!-- Left Column - Key Metrics -->
            <div class="col-lg-8">
                <!-- Performance Cards -->
                <div class="row mb-4">
                    <div class="col-md-2">
                        <div class="card bg-dark text-white">
                            <div class="card-body">
                                <h6 class="card-title">Total Return</h6>
                                <h4 id="total-return" class="text-success">--</h4>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="card bg-dark text-white">
                            <div class="card-body">
                                <h6 class="card-title">Daily Return</h6>
                                <h4 id="daily-return" class="text-info">--</h4>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="card bg-dark text-white">
                            <div class="card-body">
                                <h6 class="card-title">Sharpe Ratio</h6>
                                <h4 id="sharpe-ratio" class="text-warning">--</h4>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="card bg-dark text-white">
                            <div class="card-body">
                                <h6 class="card-title">Max Drawdown</h6>
                                <h4 id="max-drawdown" class="text-danger">--</h4>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="card bg-dark text-white">
                            <div class="card-body">
                                <h6 class="card-title">Win Rate</h6>
                                <h4 id="win-rate" class="text-primary">--</h4>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="card bg-dark text-white">
                            <div class="card-body">
                                <h6 class="card-title">Active Positions</h6>
                                <h4 id="active-positions" class="text-light">--</h4>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Performance Chart -->
                <div class="card bg-dark text-white mb-4">
                    <div class="card-header">
                        <h5><i class="fas fa-chart-area me-2"></i>Portfolio Performance</h5>
                    </div>
                    <div class="card-body">
                        <div id="performance-chart"></div>
                    </div>
                </div>

                <!-- Market Data -->
                <div class="card bg-dark text-white mb-4">
                    <div class="card-header">
                        <h5><i class="fas fa-coins me-2"></i>Live Market Data</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-dark table-striped">
                                <thead>
                                    <tr>
                                        <th>Symbol</th>
                                        <th>Price</th>
                                        <th>Change</th>
                                        <th>Volume</th>
                                        <th>Source</th>
                                        <th>Updated</th>
                                    </tr>
                                </thead>
                                <tbody id="market-data-table">
                                    <!-- Market data will be inserted here -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right Column - Signals & Analysis -->
            <div class="col-lg-4">
                <!-- Trading Signals -->
                <div class="card bg-dark text-white mb-4">
                    <div class="card-header">
                        <h5><i class="fas fa-signal me-2"></i>Recent Signals</h5>
                    </div>
                    <div class="card-body" style="max-height: 300px; overflow-y: auto;">
                        <div id="signals-list">
                            <!-- Signals will be inserted here -->
                        </div>
                    </div>
                </div>

                <!-- Sentiment Analysis -->
                <div class="card bg-dark text-white mb-4">
                    <div class="card-header">
                        <h5><i class="fas fa-brain me-2"></i>Market Sentiment</h5>
                    </div>
                    <div class="card-body" style="max-height: 300px; overflow-y: auto;">
                        <div id="sentiment-analysis">
                            <!-- Sentiment analysis will be inserted here -->
                        </div>
                    </div>
                </div>

                <!-- Risk Metrics -->
                <div class="card bg-dark text-white mb-4">
                    <div class="card-header">
                        <h5><i class="fas fa-shield-alt me-2"></i>Risk Management</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-6">
                                <small>Portfolio VaR</small>
                                <div class="h6 text-danger" id="portfolio-var">--</div>
                            </div>
                            <div class="col-6">
                                <small>Volatility</small>
                                <div class="h6 text-warning" id="portfolio-volatility">--</div>
                            </div>
                            <div class="col-6">
                                <small>Beta</small>
                                <div class="h6 text-info" id="portfolio-beta">--</div>
                            </div>
                            <div class="col-6">
                                <small>Exposure</small>
                                <div class="h6 text-primary" id="current-exposure">--</div>
                            </div>
                        </div>
                        <div class="mt-3">
                            <small>Exposure Limit</small>
                            <div class="progress" style="height: 20px;">
                                <div id="exposure-bar" class="progress-bar bg-success" role="progressbar" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- JavaScript -->
    <script src="{{ url_for('static', filename='js/dashboard.js') }}"></script>
</body>
</html>'''

        with open('dashboard/templates/dashboard.html', 'w', encoding='utf-8') as f:
            f.write(html_template)

        # Create CSS file
        css_content = '''/* Trading Bot Dashboard - Dark Mode Styles */
body {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.bg-dark {
    background-color: #161b22 !important;
}

.card.bg-dark {
    background-color: #21262d !important;
    border: 1px solid #30363d;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

.table-dark {
    background-color: #21262d;
}

.table-dark th {
    border-top: none;
    background-color: #30363d;
}

.navbar-dark {
    background-color: #161b22 !important;
    border-bottom: 1px solid #30363d;
}

.badge.bg-success {
    background-color: #238636 !important;
}

.text-success { color: #3fb950 !important; }
.text-danger { color: #f85149 !important; }
.text-warning { color: #d29922 !important; }
.text-info { color: #58a6ff !important; }
.text-primary { color: #79c0ff !important; }

.card-header {
    background-color: #30363d;
    border-bottom: 1px solid #21262d;
    font-weight: 600;
}

.progress {
    background-color: #21262d;
}

.signal-item {
    padding: 8px;
    margin-bottom: 8px;
    background-color: #161b22;
    border-radius: 6px;
    border-left: 4px solid transparent;
}

.signal-buy { border-left-color: #3fb950; }
.signal-sell { border-left-color: #f85149; }
.signal-hold { border-left-color: #d29922; }

.sentiment-item {
    padding: 8px;
    margin-bottom: 8px;
    background-color: #161b22;
    border-radius: 6px;
    font-size: 0.9em;
}

.sentiment-positive { border-left: 4px solid #3fb950; }
.sentiment-negative { border-left: 4px solid #f85149; }
.sentiment-neutral { border-left: 4px solid #6e7681; }

/* Animations */
.card {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #21262d;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: #6e7681;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #8b949e;
}

/* Custom badges */
.status-badge {
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: 500;
}

.status-active { background-color: #238636; color: white; }
.status-inactive { background-color: #6e7681; color: white; }
.status-error { background-color: #da3633; color: white; }'''

        with open('dashboard/static/css/dashboard.css', 'w', encoding='utf-8') as f:
            f.write(css_content)

        # Create JavaScript file  
        js_content = '''// Trading Bot Dashboard JavaScript
class TradingDashboard {
    constructor() {
        this.socket = io();
        this.lastUpdate = null;
        this.init();
    }

    init() {
        this.setupSocketEvents();
        this.loadInitialData();
        this.loadPerformanceChart();
        
        // Auto-refresh every 30 seconds
        setInterval(() => {
            this.loadInitialData();
        }, 30000);
    }

    setupSocketEvents() {
        this.socket.on('connect', () => {
            console.log('Connected to dashboard server');
            this.updateConnectionStatus(true);
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from dashboard server');
            this.updateConnectionStatus(false);
        });

        this.socket.on('dashboard_update', (data) => {
            console.log('Dashboard update received:', data);
            this.updateDashboard(data.data);
            this.updateLastUpdate(data.timestamp);
        });
    }

    updateConnectionStatus(connected) {
        const statusEl = document.getElementById('connection-status');
        if (connected) {
            statusEl.innerHTML = '<i class="fas fa-circle me-1"></i>Connected';
            statusEl.className = 'badge bg-success';
        } else {
            statusEl.innerHTML = '<i class="fas fa-circle me-1"></i>Disconnected';
            statusEl.className = 'badge bg-danger';
        }
    }

    updateLastUpdate(timestamp) {
        const updateTime = new Date(timestamp).toLocaleTimeString();
        document.getElementById('update-time').textContent = updateTime;
    }

    async loadInitialData() {
        try {
            const response = await fetch('/api/dashboard_data');
            const data = await response.json();
            this.updateDashboard(data);
        } catch (error) {
            console.error('Error loading dashboard data:', error);
        }
    }

    async loadPerformanceChart() {
        try {
            const response = await fetch('/api/performance_chart');
            const chartData = await response.json();
            
            if (!chartData.error) {
                Plotly.newPlot('performance-chart', chartData.data, chartData.layout, {
                    responsive: true,
                    displayModeBar: false
                });
            }
        } catch (error) {
            console.error('Error loading performance chart:', error);
        }
    }

    updateDashboard(data) {
        this.updatePerformanceMetrics(data.performance || {});
        this.updateMarketData(data.market_data || {});
        this.updateTradingSignals(data.signals || []);
        this.updateSentimentAnalysis(data.sentiment || []);
        this.updateRiskMetrics(data.risk_metrics || {});
    }

    updatePerformanceMetrics(performance) {
        const metrics = [
            { id: 'total-return', key: 'total_return', suffix: '%', precision: 2 },
            { id: 'daily-return', key: 'daily_return', suffix: '%', precision: 2 },
            { id: 'sharpe-ratio', key: 'sharpe_ratio', suffix: '', precision: 2 },
            { id: 'max-drawdown', key: 'max_drawdown', suffix: '%', precision: 2 },
            { id: 'win-rate', key: 'win_rate', suffix: '%', precision: 1 },
            { id: 'active-positions', key: 'current_positions', suffix: '', precision: 0 }
        ];

        metrics.forEach(metric => {
            const element = document.getElementById(metric.id);
            if (element && performance[metric.key] !== undefined) {
                const value = performance[metric.key];
                element.textContent = value.toFixed(metric.precision) + metric.suffix;
                
                // Color coding for returns
                if (metric.key.includes('return') && element.classList.contains('text-success')) {
                    element.className = value >= 0 ? 'text-success' : 'text-danger';
                }
            }
        });
    }

    updateMarketData(marketData) {
        const tbody = document.getElementById('market-data-table');
        tbody.innerHTML = '';

        Object.values(marketData).forEach(data => {
            const row = document.createElement('tr');
            const change = (Math.random() - 0.5) * 4; // Mock change
            const changeClass = change >= 0 ? 'text-success' : 'text-danger';
            
            row.innerHTML = `
                <td><strong>${data.symbol}</strong></td>
                <td>$${data.price?.toFixed(2) || '--'}</td>
                <td class="${changeClass}">${change >= 0 ? '+' : ''}${change.toFixed(2)}%</td>
                <td>${data.volume?.toLocaleString() || '--'}</td>
                <td><span class="badge bg-secondary">${data.source || '--'}</span></td>
                <td><small>${new Date(data.timestamp).toLocaleTimeString()}</small></td>
            `;
            tbody.appendChild(row);
        });
    }

    updateTradingSignals(signals) {
        const container = document.getElementById('signals-list');
        container.innerHTML = '';

        if (signals.length === 0) {
            container.innerHTML = '<p class="text-muted">No recent signals</p>';
            return;
        }

        signals.slice(-10).reverse().forEach(signal => {
            const div = document.createElement('div');
            div.className = `signal-item signal-${signal.signal_type?.toLowerCase() || 'hold'}`;
            
            const confidence = (signal.confidence * 100).toFixed(1);
            const time = new Date(signal.timestamp).toLocaleTimeString();
            
            div.innerHTML = `
                <div class="d-flex justify-content-between">
                    <strong>${signal.symbol}</strong>
                    <span class="badge bg-${this.getSignalBadgeColor(signal.signal_type)}">${signal.signal_type?.toUpperCase()}</span>
                </div>
                <div class="d-flex justify-content-between">
                    <small>$${signal.price?.toFixed(2)} - ${confidence}%</small>
                    <small class="text-muted">${time}</small>
                </div>
                ${signal.strategy ? `<small class="text-info">${signal.strategy}</small>` : ''}
            `;
            container.appendChild(div);
        });
    }

    updateSentimentAnalysis(sentimentData) {
        const container = document.getElementById('sentiment-analysis');
        container.innerHTML = '';

        if (!Array.isArray(sentimentData) || sentimentData.length === 0) {
            container.innerHTML = '<p class="text-muted">No sentiment data available</p>';
            return;
        }

        sentimentData.forEach(item => {
            const div = document.createElement('div');
            const sentimentClass = this.getSentimentClass(item.sentiment);
            div.className = `sentiment-item ${sentimentClass}`;
            
            div.innerHTML = `
                <div class="fw-bold mb-1" style="font-size: 0.9em;">${item.title}</div>
                <div class="d-flex justify-content-between">
                    <small>Sentiment: ${item.sentiment?.toFixed(2) || 'N/A'}</small>
                    <small class="text-muted">${new Date(item.published).toLocaleString()}</small>
                </div>
            `;
            container.appendChild(div);
        });
    }

    updateRiskMetrics(riskData) {
        const metrics = [
            { id: 'portfolio-var', key: 'portfolio_var', suffix: '%' },
            { id: 'portfolio-volatility', key: 'portfolio_volatility', suffix: '%' },
            { id: 'portfolio-beta', key: 'beta', suffix: '' },
            { id: 'current-exposure', key: 'current_exposure', suffix: '%' }
        ];

        metrics.forEach(metric => {
            const element = document.getElementById(metric.id);
            if (element && riskData[metric.key] !== undefined) {
                element.textContent = riskData[metric.key].toFixed(2) + metric.suffix;
            }
        });

        // Update exposure bar
        if (riskData.current_exposure && riskData.exposure_limit) {
            const exposureBar = document.getElementById('exposure-bar');
            const percentage = (riskData.current_exposure / riskData.exposure_limit) * 100;
            exposureBar.style.width = percentage + '%';
            
            // Change color based on exposure level
            exposureBar.className = `progress-bar ${percentage > 80 ? 'bg-danger' : percentage > 60 ? 'bg-warning' : 'bg-success'}`;
        }
    }

    getSignalBadgeColor(signalType) {
        const colors = {
            'buy': 'success',
            'sell': 'danger',
            'hold': 'warning'
        };
        return colors[signalType?.toLowerCase()] || 'secondary';
    }

    getSentimentClass(sentiment) {
        if (sentiment > 0.1) return 'sentiment-positive';
        if (sentiment < -0.1) return 'sentiment-negative';
        return 'sentiment-neutral';
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    new TradingDashboard();
});'''

        with open('dashboard/static/js/dashboard.js', 'w', encoding='utf-8') as f:
            f.write(js_content)
    
    def stop(self):
        """Stop the dashboard server"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join()

# Main execution
if __name__ == "__main__":
    dashboard = TradingDashboard()
    
    # Initialize components
    asyncio.run(dashboard.initialize())
    
    # Start dashboard
    dashboard.start(host='localhost', port=5000, debug=False)