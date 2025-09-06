#!/usr/bin/env python3
"""
Advanced Trading Bot - Main Entry Point
Multi-source sentiment analysis with dual ML approaches
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config import Config
from core.bot_orchestrator import TradingBotOrchestrator
from core.utils import setup_logging, create_directories

class GracefulKiller:
    """Handle graceful shutdown on SIGTERM/SIGINT"""
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        logging.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.kill_now = True

async def main():
    """Main execution function"""
    
    # Setup logging and directories
    setup_logging()
    create_directories()
    
    # Initialize graceful shutdown handler
    killer = GracefulKiller()
    
    # Load configuration
    config = Config()
    
    logging.info("=" * 60)
    logging.info("Advanced News-Based Trading Bot Starting...")
    logging.info("=" * 60)
    logging.info(f"Initial Capital: EUR {config.INITIAL_CAPITAL:,.2f}")
    logging.info(f"Traditional ML: {'Enabled' if config.ENABLE_TRADITIONAL_ML else 'Disabled'}")
    logging.info(f"Transformer ML: {'Enabled' if config.ENABLE_TRANSFORMER_ML else 'Disabled'}")
    logging.info(f"Social Sentiment: {'Enabled' if config.ENABLE_SOCIAL_SENTIMENT else 'Disabled'}")
    logging.info(f"Trading Symbols: {len(config.US_SYMBOLS + config.EU_SYMBOLS)} total")
    logging.info("=" * 60)
    
    try:
        # Initialize the trading bot orchestrator
        bot_orchestrator = TradingBotOrchestrator(config)
        
        # Initialize all components
        await bot_orchestrator.initialize()
        
        # Start the main trading loop
        cycle_count = 0
        max_cycles = config.MAX_CYCLES if hasattr(config, 'MAX_CYCLES') else float('inf')
        
        while not killer.kill_now and cycle_count < max_cycles:
            cycle_count += 1
            
            logging.info(f"Starting Trading Cycle {cycle_count}")
            
            try:
                # Run trading cycle
                cycle_result = await bot_orchestrator.run_trading_cycle()
                
                if 'error' in cycle_result:
                    logging.error(f"Cycle {cycle_count} failed: {cycle_result['error']}")
                else:
                    logging.info(f"Cycle {cycle_count} completed successfully")
                    
                    # Log cycle summary
                    summary = cycle_result.get('portfolio_summary', {})
                    logging.info(f"Portfolio Value: EUR {summary.get('total_value', 0):,.2f}")
                    logging.info(f"Total Return: {summary.get('total_return_pct', 0):.2f}%")
                    logging.info(f"Signals Generated: {cycle_result.get('signals_generated', 0)}")
                    logging.info(f"Trades Executed: {cycle_result.get('trades_executed', 0)}")
                
                # Wait before next cycle
                if not killer.kill_now:
                    logging.info(f"Waiting {config.REFRESH_INTERVAL} seconds before next cycle...")
                    await asyncio.sleep(config.REFRESH_INTERVAL)
                
            except Exception as e:
                logging.error(f"Error in trading cycle {cycle_count}: {e}", exc_info=True)
                # Wait a bit before retrying
                await asyncio.sleep(60)
        
        # Generate final report
        if cycle_count > 0:
            logging.info("=" * 60)
            logging.info("Generating Final Performance Report...")
            final_report = await bot_orchestrator.get_detailed_performance_report()
            
            # Log final performance
            portfolio = final_report.get('portfolio_summary', {})
            performance = final_report.get('performance_metrics', {})
            
            logging.info("FINAL PERFORMANCE REPORT")
            logging.info("-" * 40)
            logging.info(f"Final Portfolio Value: EUR {portfolio.get('total_value', 0):,.2f}")
            logging.info(f"Total Return: {portfolio.get('total_return_pct', 0):.2f}%")
            logging.info(f"Total Trades: {len(final_report.get('trading_history', []))}")
            
            if performance:
                logging.info(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.3f}")
                logging.info(f"Max Drawdown: {performance.get('max_drawdown', 0)*100:.2f}%")
                logging.info(f"Volatility: {performance.get('volatility', 0)*100:.2f}%")
                logging.info(f"Win Rate: {performance.get('win_rate', 0)*100:.1f}%")
            
            logging.info("=" * 60)
            
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logging.error(f"Critical error in main execution: {e}", exc_info=True)
        return 1
    finally:
        # Cleanup
        if 'bot_orchestrator' in locals():
            await bot_orchestrator.cleanup()
        logging.info("Trading bot shutdown complete")
    
    return 0

if __name__ == "__main__":
    # Run the bot
    exit_code = asyncio.run(main())
    sys.exit(exit_code)