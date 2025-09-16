import logging
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Set, Any, Optional
from flask import Flask
from flask_socketio import SocketIO, emit, join_room, leave_room
import eventlet

# Import API blueprints and initializers
from api.dashboard_api import dashboard_api, init_websocket_events, set_orchestrator_instance
from api.advanced_portfolio_api import create_advanced_portfolio_bp, set_bot_orchestrator

try:
    from knowledge_graph.kg_api import kg_api, init_kg_websocket_events
    KNOWLEDGE_GRAPH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    KNOWLEDGE_GRAPH_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for managing real-time data
connected_clients = set()
client_subscriptions = {}
data_cache = {
    'performance': {},
    'trades': [],
    'ml_metrics': {},
    'risk_metrics': {},
    'sentiment': {},
    'system_health': {}
}

def create_flask_app(bot_orchestrator):
    """
    Creates the Flask application and registers blueprints.
    """
    flask_app = Flask(__name__)
    flask_app.register_blueprint(dashboard_api, url_prefix='/api')
    logger.info("Registered dashboard_api blueprint under /api.")

    # Register advanced portfolio API
    advanced_bp = create_advanced_portfolio_bp()
    flask_app.register_blueprint(advanced_bp)
    set_bot_orchestrator(bot_orchestrator)  # Inject bot orchestrator
    logger.info("Registered advanced_portfolio_api blueprint under /api/advanced.")

    if KNOWLEDGE_GRAPH_AVAILABLE:
        flask_app.register_blueprint(kg_api)
        logger.info("Registered kg_api blueprint.")
    else:
        logger.warning("Knowledge Graph API not available, skipping blueprint registration.")

    # Pass the orchestrator instance to the dashboard API
    set_orchestrator_instance(bot_orchestrator)

    return flask_app

def init_realtime_events(socketio):
    """Initialize real-time WebSocket events for dashboard updates"""

    @socketio.on('connect')
    def handle_connect():
        client_id = f"client_{len(connected_clients) + 1}_{int(time.time())}"
        connected_clients.add(client_id)
        client_subscriptions[client_id] = set()
        logger.info(f"Client {client_id} connected. Total clients: {len(connected_clients)}")

        # Send initial data to the new client
        emit('initial_data', {
            'type': 'initial_data',
            'timestamp': datetime.now().isoformat(),
            'data': data_cache,
            'client_id': client_id
        })

    @socketio.on('disconnect')
    def handle_disconnect():
        # Find and remove the disconnected client
        for client_id in list(connected_clients):
            # In a real implementation, you'd track client IDs properly
            pass
        logger.info(f"Client disconnected. Total clients: {len(connected_clients)}")

    @socketio.on('subscribe')
    def handle_subscribe(data):
        client_id = data.get('client_id')
        subscriptions = data.get('subscriptions', [])

        if client_id in client_subscriptions:
            client_subscriptions[client_id].update(subscriptions)
            join_room(f"subscription_{client_id}")
            logger.info(f"Client {client_id} subscribed to: {subscriptions}")

            emit('subscription_confirmed', {
                'subscriptions': list(client_subscriptions[client_id])
            })

    @socketio.on('unsubscribe')
    def handle_unsubscribe(data):
        client_id = data.get('client_id')
        subscriptions = data.get('subscriptions', [])

        if client_id in client_subscriptions:
            client_subscriptions[client_id].difference_update(subscriptions)
            logger.info(f"Client {client_id} unsubscribed from: {subscriptions}")

    @socketio.on('ping')
    def handle_ping():
        emit('pong', {
            'timestamp': datetime.now().isoformat()
        })

def broadcast_update(socketio, data_type: str, data: Dict[str, Any]):
    """Broadcast updates to all connected clients"""
    global data_cache

    message = {
        'type': 'update',
        'data_type': data_type,
        'timestamp': datetime.now().isoformat(),
        'data': data
    }

    # Update cache
    data_cache[data_type] = data

    # Broadcast to all connected clients
    socketio.emit('dashboard_update', message, broadcast=True)
    logger.debug(f"Broadcasted {data_type} update to {len(connected_clients)} clients")

def start_data_generators(socketio):
    """Start background tasks for generating mock real-time data"""
    import random

    def generate_performance_data():
        """Generate mock performance data"""
        while True:
            data = {
                'total_return': round(random.uniform(-5, 15), 2),
                'daily_return': round(random.uniform(-2, 3), 2),
                'sharpe_ratio': round(random.uniform(1.2, 2.8), 2),
                'max_drawdown': round(random.uniform(-8, -2), 2),
                'win_rate': round(random.uniform(55, 75), 1),
                'active_positions': random.randint(8, 25),
                'timestamp': datetime.now().isoformat()
            }
            broadcast_update(socketio, 'performance', data)
            eventlet.sleep(5)  # Update every 5 seconds

    def generate_trade_data():
        """Generate mock trade data"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META']
        actions = ['BUY', 'SELL']

        while True:
            data = {
                'symbol': random.choice(symbols),
                'action': random.choice(actions),
                'quantity': random.randint(10, 200),
                'price': round(random.uniform(100, 500), 2),
                'timestamp': datetime.now().isoformat(),
                'confidence': round(random.uniform(0.7, 0.95), 3)
            }

            # Add to trades cache (keep last 50 trades)
            global data_cache
            if 'trades' not in data_cache:
                data_cache['trades'] = []
            data_cache['trades'].append(data)
            if len(data_cache['trades']) > 50:
                data_cache['trades'] = data_cache['trades'][-50:]

            broadcast_update(socketio, 'trades', data)
            eventlet.sleep(2)  # Update every 2 seconds

    def generate_ml_metrics():
        """Generate mock ML metrics"""
        while True:
            data = {
                'ensemble_accuracy': round(random.uniform(85, 92), 1),
                'prediction_confidence': round(random.uniform(80, 95), 1),
                'models_training': random.randint(0, 2),
                'cache_hit_rate': round(random.uniform(0.8, 0.95), 3),
                'active_predictions': random.randint(15, 35),
                'timestamp': datetime.now().isoformat()
            }
            broadcast_update(socketio, 'ml_metrics', data)
            eventlet.sleep(10)  # Update every 10 seconds

    def generate_system_health():
        """Generate mock system health data"""
        while True:
            data = {
                'cpu_usage': round(random.uniform(20, 80), 1),
                'memory_usage': round(random.uniform(40, 85), 1),
                'network_io': round(random.uniform(10, 50), 1),
                'active_connections': len(connected_clients),
                'api_response_time': round(random.uniform(25, 100), 0),
                'timestamp': datetime.now().isoformat()
            }
            broadcast_update(socketio, 'system_health', data)
            eventlet.sleep(5)  # Update every 5 seconds

    # Start background tasks
    socketio.start_background_task(generate_performance_data)
    socketio.start_background_task(generate_trade_data)
    socketio.start_background_task(generate_ml_metrics)
    socketio.start_background_task(generate_system_health)

    logger.info("Started real-time data generators")

def create_socketio_app(bot_orchestrator):
    """
    Creates the Flask application and SocketIO server.
    """
    flask_app = create_flask_app(bot_orchestrator)
    socketio = SocketIO(flask_app, async_mode='eventlet', cors_allowed_origins="*")

    # Initialize WebSocket events from the API modules
    init_websocket_events(socketio)
    if KNOWLEDGE_GRAPH_AVAILABLE:
        init_kg_websocket_events(socketio)

    # Initialize real-time dashboard events
    init_realtime_events(socketio)

    # Start data generators for real-time updates
    start_data_generators(socketio)

    return flask_app, socketio

if __name__ == "__main__":
    """Run the dashboard server directly for testing"""
    import sys
    import os

    # Add the parent directory to the path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Create a mock orchestrator for testing
    class MockOrchestrator:
        def __init__(self):
            self.running = True

    mock_orchestrator = MockOrchestrator()
    app, socketio = create_socketio_app(mock_orchestrator)

    logger.info("Starting dashboard server on http://localhost:8080")
    socketio.run(app, host='0.0.0.0', port=8080, debug=False)
