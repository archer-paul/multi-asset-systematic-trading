import logging
from aiohttp import web
from datetime import datetime

logger = logging.getLogger(__name__)

async def setup_server(bot_orchestrator):
    """
    Sets up the aiohttp web server for health checks.
    """
    logger.info("Setting up aiohttp health check server...")

    app = web.Application()

    async def health_check(request):
        """Native aiohttp health check endpoint."""
        return web.json_response({
            'status': 'healthy',
            'bot_initialized': bot_orchestrator.is_initialized,
            'cycle_count': bot_orchestrator.cycle_count,
            'timestamp': datetime.now().isoformat()
        })

    app.router.add_get('/health', health_check)
    app.router.add_get('/', health_check)
    logger.info("Native aiohttp health check endpoints configured.")

    return app