#!/usr/bin/env python3
"""
Launch script for Trading Bot Dashboard
Starts the ultra-detailed dark mode dashboard
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from dashboard.app import TradingDashboard
from core.config import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def main():
    """Main function to launch the dashboard"""
    try:
        logger.info("Starting Trading Bot Dashboard...")
        
        # Create dashboard instance
        dashboard = TradingDashboard(config)
        
        # Initialize dashboard components
        initialized = await dashboard.initialize()
        if not initialized:
            logger.error("Dashboard initialization failed. Some features may not work properly.")
        
        # Start dashboard server
        logger.info("Dashboard initialized successfully!")
        logger.info("Opening dashboard at http://localhost:5000")
        logger.info("Press Ctrl+C to stop the dashboard")
        
        # Start the server (this will block)
        dashboard.start(host='localhost', port=5000, debug=config.debug_mode)
        
    except KeyboardInterrupt:
        logger.info("Dashboard shutdown requested by user")
    except Exception as e:
        logger.error(f"Dashboard startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())