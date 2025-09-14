"""
Unified Web Server Setup
This module configures and bridges the aiohttp and Flask applications
to run as a single, robust service, especially for Cloud Run.
"""

import logging
from aiohttp import web
from flask import Flask
from flask_socketio import SocketIO
import aiohttp_wsgi
from datetime import datetime

# Import API blueprints and initializers
from api.dashboard_api import dashboard_api, init_websocket_events, set_orchestrator_instance

# We need to check if the module is available before using it
try:
    from knowledge_graph.kg_api import kg_api, init_kg_websocket_events
    KNOWLEDGE_GRAPH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    KNOWLEDGE_GRAPH_AVAILABLE = False

logger = logging.getLogger(__name__)

async def setup_server(bot_orchestrator):
    """
    Sets up the combined aiohttp/Flask web server.
    """
    logger.info("Setting up unified aiohttp/Flask server...")

    # 1. Create main aiohttp app
    aiohttp_app = web.Application()

    # 2. Create Flask app and register blueprints
    flask_app = Flask(__name__)
    
    # Register API blueprints
    flask_app.register_blueprint(dashboard_api, url_prefix='/api')
    logger.info("Registered dashboard_api blueprint under /api.")
    
    if KNOWLEDGE_GRAPH_AVAILABLE:
        # The kg_api already has a url_prefix='/api/knowledge-graph'
        flask_app.register_blueprint(kg_api)
        logger.info("Registered kg_api blueprint.")
    else:
        logger.warning("Knowledge Graph API not available, skipping blueprint registration.")

    # 3. Setup Socket.IO and attach to aiohttp app
    # Use the aiohttp async mode for flask-socketio
    socketio = SocketIO(flask_app, async_mode='aiohttp', cors_allowed_origins="*")
    
    # Initialize WebSocket events from the API modules
    init_websocket_events(socketio)
    if KNOWLEDGE_GRAPH_AVAILABLE:
        init_kg_websocket_events(socketio)
    
    # Attach the Socket.IO server to the aiohttp application
    socketio.attach(aiohttp_app)
    logger.info("Socket.IO server attached to aiohttp application.")

    # 4. Setup WSGI bridge for all other Flask routes
    wsgi_handler = aiohttp_wsgi.WSGIHandler(flask_app)
    # Route all requests to the wsgi handler. The flask app will handle routing from there.
    aiohttp_app.router.add_route("*", "/{path_info:.*}", wsgi_handler)
    logger.info("WSGI handler for Flask app registered with aiohttp.")

    # 5. Add native aiohttp health checks for Cloud Run
    async def health_check(request):
        """Native aiohttp health check endpoint."""
        return web.json_response({
            'status': 'healthy',
            'bot_initialized': bot_orchestrator.is_initialized,
            'cycle_count': bot_orchestrator.cycle_count,
            'timestamp': datetime.now().isoformat()
        })
    
    # We need to add the native aiohttp routes *before* the catch-all WSGI handler.
    # To do this, we create a separate app for the health checks and compose them.
    health_app = web.Application()
    health_app.router.add_get('/health', health_check)
    health_app.router.add_get('/', health_check) # Root health check
    
    # Add the health check app as a sub-app to the main one.
    # This ensures its routes are checked before the catch-all WSGI handler.
    aiohttp_app.add_subapp('/', health_app)

    logger.info("Native aiohttp health check endpoints configured.")

    # Pass the orchestrator instance to the dashboard API
    set_orchestrator_instance(bot_orchestrator)

    return aiohttp_app
