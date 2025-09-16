"""
WSGI entry point for Gunicorn
This script initializes the bot, starts the main trading loop in the background,
and exposes the Flask app for the WSGI server.
"""

import asyncio
import logging
import os
import sys
import threading

# Add project root to path to ensure imports work correctly
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from api.dashboard_server import create_socketio_app
from core.config import Config
from core.utils import setup_logging, create_directories
from enhanced_main import EnhancedTradingBot

# --- 1. Basic Setup ---
setup_logging()
create_directories()
config = Config()
logger = logging.getLogger(__name__)

# --- 2. Global Bot Instance ---
# Create a single instance of the bot to be shared
logger.info("Creating EnhancedTradingBot instance...")
bot = EnhancedTradingBot(config)

# --- 3. Background Task for the Main Trading Loop ---
def run_bot_in_background():
    """Runs the bot's main continuous loop."""
    logger.info("Background thread started for continuous trading loop.")
    try:
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Initialize and run the bot
        loop.run_until_complete(bot.initialize())
        loop.run_until_complete(bot.run_continuous())
        
    except Exception as e:
        logger.critical(f"Critical error in bot background thread: {e}", exc_info=True)
    finally:
        logger.info("Bot background thread finished.")

# --- 4. Create the WSGI App Object ---
# The bot_orchestrator is inside the `bot` instance, created by its __init__
if bot.bot_orchestrator is None:
     # This is a fallback, __init__ should have already created it.
    from core.bot_orchestrator import TradingBotOrchestrator
    bot.bot_orchestrator = TradingBotOrchestrator(config)

logger.info("Creating Flask and SocketIO app...")
# Pass the orchestrator from our single bot instance to the app factory
flask_app, socketio = create_socketio_app(bot.bot_orchestrator)

# The `app` variable is what Gunicorn looks for by default
app = flask_app 

# --- 5. Start the Background Thread ---
logger.info("Starting the main trading loop in a background thread...")
bot_thread = threading.Thread(target=run_bot_in_background, daemon=True)
bot_thread.start()

logger.info("WSGI app object created. Gunicorn can now serve the application.")
