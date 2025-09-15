import logging
from flask import Flask
from flask_socketio import SocketIO

# Import API blueprints and initializers
from api.dashboard_api import dashboard_api, init_websocket_events, set_orchestrator_instance

try:
    from knowledge_graph.kg_api import kg_api, init_kg_websocket_events
    KNOWLEDGE_GRAPH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    KNOWLEDGE_GRAPH_AVAILABLE = False

logger = logging.getLogger(__name__)

def create_flask_app(bot_orchestrator):
    """
    Creates the Flask application and registers blueprints.
    """
    flask_app = Flask(__name__)
    flask_app.register_blueprint(dashboard_api, url_prefix='/api')
    logger.info("Registered dashboard_api blueprint under /api.")

    if KNOWLEDGE_GRAPH_AVAILABLE:
        flask_app.register_blueprint(kg_api)
        logger.info("Registered kg_api blueprint.")
    else:
        logger.warning("Knowledge Graph API not available, skipping blueprint registration.")

    # Pass the orchestrator instance to the dashboard API
    set_orchestrator_instance(bot_orchestrator)

    return flask_app

def run_dashboard_server(bot_orchestrator, host='0.0.0.0', port=5000):
    """
    Runs the Flask-SocketIO server.
    """
    flask_app = create_flask_app(bot_orchestrator)
    socketio = SocketIO(flask_app, async_mode='threading', cors_allowed_origins="*")

    # Initialize WebSocket events from the API modules
    init_websocket_events(socketio)
    if KNOWLEDGE_GRAPH_AVAILABLE:
        init_kg_websocket_events(socketio)

    logger.info(f"Starting Flask-SocketIO server on {host}:{port}")
    socketio.run(flask_app, host=host, port=port, allow_unsafe_werkzeug=True)
