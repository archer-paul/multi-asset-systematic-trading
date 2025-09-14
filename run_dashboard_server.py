#!/usr/bin/env python3.11
"""
Dashboard Server for Trading Bot
Runs Flask app with SocketIO for real-time dashboard
"""

import asyncio
import logging
import os
import sys
import threading
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flask import Flask
from flask_socketio import SocketIO
from api.dashboard_api import dashboard_api, data_provider, init_websocket_events
from core.config import Config
from core.utils import setup_logging

def create_app():
    """Create Flask app with dashboard API"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'trading-bot-dashboard-secret-key-2024'

    # Register dashboard API blueprint
    app.register_blueprint(dashboard_api)

    return app

async def initialize_data_provider():
    """Initialize dashboard data provider"""
    try:
        config = Config()
        success = await data_provider.initialize()
        if success:
            logging.info("Dashboard data provider initialized successfully")
        else:
            logging.error("Failed to initialize dashboard data provider")
        return success
    except Exception as e:
        logging.error(f"Error initializing data provider: {e}")
        return False

def start_real_time_updates(socketio):
    """Start background thread for real-time updates"""
    def update_loop():
        while True:
            try:
                # Emit portfolio updates every 30 seconds
                socketio.emit('portfolio_update', {'timestamp': time.time()})

                # Emit ML updates every 60 seconds
                socketio.emit('ml_update', {'timestamp': time.time()})

                # Emit risk updates every 45 seconds
                socketio.emit('risk_update', {'timestamp': time.time()})

                time.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logging.error(f"Error in real-time update loop: {e}")
                time.sleep(60)  # Wait longer on error

    # Start the update thread
    update_thread = threading.Thread(target=update_loop, daemon=True)
    update_thread.start()
    logging.info("Real-time update thread started")

def run_dashboard_server():
    """Run the dashboard server"""

    # Setup logging
    setup_logging('dashboard_server')

    # Initialize data provider
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    success = loop.run_until_complete(initialize_data_provider())
    if not success:
        logging.error("Failed to initialize data provider, exiting")
        sys.exit(1)

    # Create Flask app
    app = create_app()

    # Initialize SocketIO
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode='threading',
        logger=True,
        engineio_logger=True
    )

    # Initialize WebSocket events
    init_websocket_events(socketio)

    # Start real-time updates
    start_real_time_updates(socketio)

    # Get port from environment or default to 5000
    port = int(os.environ.get('DASHBOARD_PORT', 5000))
    host = os.environ.get('DASHBOARD_HOST', '0.0.0.0')

    logging.info(f"Starting dashboard server on {host}:{port}")

    try:
        # Run the SocketIO server
        socketio.run(
            app,
            host=host,
            port=port,
            debug=False,
            use_reloader=False,
            log_output=True
        )
    except KeyboardInterrupt:
        logging.info("Dashboard server stopped by user")
    except Exception as e:
        logging.error(f"Dashboard server error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    run_dashboard_server()