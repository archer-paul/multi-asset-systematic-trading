#!/usr/bin/env python3.13
"""
Enhanced Trading Bot - Version élargie avec univers global et analyse long terme
Multi-horizon: trading court terme, émergent, et investissement long terme (3-5 ans)
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config import Config
from core.bot_orchestrator import TradingBotOrchestrator
from core.utils import setup_logging, create_directories
from analysis.emerging_detector import EmergingStockDetector
from analysis.long_term_analyzer import LongTermAnalyzer
from analysis.social_media_v2 import SocialMediaAnalyzerV2
from analysis.congress_trading import CongressTradingAnalyzer
from dashboard.recommendations_dashboard import RecommendationsDashboard

class GracefulKiller:
    """Handle graceful shutdown on SIGTERM/SIGINT"""
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        logging.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.kill_now = True

class EnhancedTradingBot:
    """Bot de trading amélioré avec analyse multi-horizon"""
    
    def __init__(self, config: Config):
        self.config = config
        self.killer = GracefulKiller()
        
        # Composants principaux
        self.bot_orchestrator = None
        self.emerging_detector = None
        self.long_term_analyzer = None
        self.social_analyzer_v2 = None
        self.congress_analyzer = None
        self.dashboard = None
        
        # État
        self.is_initialized = False
        self.cycle_count = 0
        
    async def initialize(self):
        """Initialise tous les composants du bot amélioré"""
        
        if self.is_initialized:
            return
            
        logging.info("=" * 70)
        logging.info("ENHANCED TRADING BOT - GLOBAL MARKET ANALYSIS")
        logging.info("=" * 70)
        logging.info(f"Trading Mode: {getattr(self.config, 'TRADING_MODE', 'fast_mode')}")
        logging.info(f"Universe Size: {len(self.config.ALL_SYMBOLS)} symbols")
        logging.info(f"Regions: US({len(self.config.US_SYMBOLS)}), EU({len(getattr(self.config, 'EU_SYMBOLS', []))}), UK({len(getattr(self.config, 'UK_SYMBOLS', []))}), Asia({len(getattr(self.config, 'ASIA_SYMBOLS', []))})")
        logging.info("=" * 70)
        
        try:
            # 1. Bot orchestrateur principal (trading court terme)
            self.bot_orchestrator = TradingBotOrchestrator(self.config)
            await self.bot_orchestrator.initialize()
            logging.info("Short-term trading orchestrator initialized")
            
            # 2. Détecteur d'actions émergentes
            self.emerging_detector = EmergingStockDetector(
                self.config, 
                self.bot_orchestrator.sentiment_analyzer
            )
            logging.info("Emerging stock detector initialized")
            
            # 3. Analyseur long terme
            self.long_term_analyzer = LongTermAnalyzer(
                self.config,
                self.bot_orchestrator.sentiment_analyzer
            )
            logging.info("Long-term analyzer initialized")
            
            # 4. Analyseur social media V2 (avec gestion rate limits)
            self.social_analyzer_v2 = SocialMediaAnalyzerV2(self.config)
            logging.info("Enhanced social media analyzer initialized")
            
            # 5. Analyseur des transactions du Congrès
            enable_congress = getattr(self.config, 'ENABLE_CONGRESS_TRACKING', True)
            if enable_congress:
                self.congress_analyzer = CongressTradingAnalyzer(self.config)
                congress_status = self.congress_analyzer.get_api_status()
                logging.info(f"Congress trading analyzer initialized - Status: {congress_status['status']}")
            else:
                self.congress_analyzer = None
                logging.info("Congress trading analysis disabled")
            
            # 6. Dashboard de recommandations
            self.dashboard = RecommendationsDashboard(self.config)
            logging.info("Recommendations dashboard initialized")
            
            self.is_initialized = True
            logging.info("Enhanced Trading Bot fully initialized!")
            
        except Exception as e:
            logging.error(f"Failed to initialize enhanced bot: {e}", exc_info=True)
            raise
    
    async def run_enhanced_cycle(self):
        """Exécute un cycle complet d'analyse multi-horizon"""
        
        if not self.is_initialized:
            raise RuntimeError("Bot must be initialized first")
        
        self.cycle_count += 1
        cycle_start = datetime.now()
        
        logging.info(f"Starting Enhanced Cycle #{self.cycle_count}")
        
        try:
            # Phase 1: Trading court terme (cycle normal)
            logging.info("Phase 1: Short-term trading analysis...")
            short_term_result = await self.bot_orchestrator.run_trading_cycle()
            
            # Phase 2: Détection d'opportunités émergentes (tous les cycles)
            logging.info("Phase 2: Scanning for emerging opportunities...")
            emerging_opportunities = []
            if self.config.get('ENABLE_EMERGING_DETECTION', True):
                try:
                    # Utiliser les news du cycle court terme
                    news_data = getattr(short_term_result, 'news_data', [])
                    market_data = getattr(short_term_result, 'market_data', {})
                    
                    emerging_opportunities = await self.emerging_detector.scan_emerging_opportunities(
                        news_data, market_data
                    )
                    logging.info(f"Found {len(emerging_opportunities)} emerging opportunities")
                except Exception as e:
                    logging.error(f"Emerging detection failed: {e}")
            
            # Phase 3: Analyse long terme (tous les 3 cycles pour économiser les ressources)
            logging.info("Phase 3: Long-term investment analysis...")
            long_term_recommendations = []
            if self.cycle_count % 3 == 1 and self.config.get('ENABLE_LONG_TERM_ANALYSIS', True):
                try:
                    # Analyser un sous-ensemble des symboles à chaque fois
                    symbols_batch = self._get_symbols_batch_for_long_term()
                    
                    long_term_recommendations = await self.long_term_analyzer.analyze_long_term_opportunities(
                        symbols_batch
                    )
                    logging.info(f"Generated {len(long_term_recommendations)} long-term recommendations")
                except Exception as e:
                    logging.error(f"Long-term analysis failed: {e}")
            
            # Phase 4: Analyse des transactions du Congrès
            congress_signals = []
            if self.congress_analyzer:
                logging.info("Phase 4: Congress trading analysis...")
                try:
                    # Analyser les signaux du Congrès pour nos symboles
                    congress_analysis = await self.congress_analyzer.analyze_congress_signals(self.config.ALL_SYMBOLS[:50])
                    congress_signals = congress_analysis.get('top_stocks', [])
                    
                    if congress_signals:
                        logging.info(f"Found {len(congress_signals)} Congress trading signals")
                        for signal in congress_signals[:3]:
                            logging.info(f"  {signal['symbol']}: {signal['sentiment']} (net: ${signal['net_activity']:,.0f})")
                    else:
                        logging.info("No significant Congress trading activity found")
                        
                except Exception as e:
                    logging.error(f"Congress analysis failed: {e}")
            
            # Phase 5: Génération du dashboard complet
            logging.info("Phase 5: Generating investment dashboard...")
            try:
                dashboard_data = await self.dashboard.generate_complete_dashboard(
                    short_term_signals=short_term_result.get('signals', []),
                    emerging_opportunities=emerging_opportunities,
                    long_term_recommendations=long_term_recommendations,
                    congress_signals=congress_signals
                )
                
                # Afficher un résumé
                summary_text = self.dashboard.get_dashboard_summary_text()
                logging.info("Dashboard Summary:")
                for line in summary_text.split('\n'):
                    if line.strip():
                        logging.info(line)
                
            except Exception as e:
                logging.error(f"Dashboard generation failed: {e}")
            
            # Phase 5: Sauvegarde et reporting
            await self._save_enhanced_results(
                short_term_result, emerging_opportunities, 
                long_term_recommendations, dashboard_data
            )
            
            # Statistiques du cycle
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            logging.info(f"Enhanced Cycle #{self.cycle_count} completed in {cycle_duration:.1f}s")
            
            return {
                'cycle_number': self.cycle_count,
                'duration_seconds': cycle_duration,
                'short_term': short_term_result,
                'emerging_count': len(emerging_opportunities),
                'long_term_count': len(long_term_recommendations),
                'dashboard_generated': True,
                'timestamp': cycle_start
            }
            
        except Exception as e:
            logging.error(f"Error in enhanced cycle {self.cycle_count}: {e}", exc_info=True)
            return {
                'cycle_number': self.cycle_count,
                'error': str(e),
                'timestamp': cycle_start
            }
    
    def _get_symbols_batch_for_long_term(self) -> List[str]:
        """Retourne un batch de symboles pour l'analyse long terme"""
        
        # Rotation des symboles pour analyser tout l'univers progressivement
        batch_size = min(30, len(self.config.ALL_SYMBOLS) // 3)  # 1/3 de l'univers par cycle
        start_idx = ((self.cycle_count - 1) * batch_size) % len(self.config.ALL_SYMBOLS)
        end_idx = start_idx + batch_size
        
        if end_idx > len(self.config.ALL_SYMBOLS):
            # Wrap around
            batch = self.config.ALL_SYMBOLS[start_idx:] + self.config.ALL_SYMBOLS[:end_idx - len(self.config.ALL_SYMBOLS)]
        else:
            batch = self.config.ALL_SYMBOLS[start_idx:end_idx]
        
        logging.info(f"Long-term analysis batch: {len(batch)} symbols (indices {start_idx}-{end_idx})")
        return batch
    
    async def _save_enhanced_results(self, short_term_result, emerging_opportunities, 
                                   long_term_recommendations, dashboard_data):
        """Sauvegarde les résultats du cycle amélioré"""
        
        try:
            # Sauvegarder via le bot orchestrateur principal
            if hasattr(short_term_result, 'signals'):
                await self.bot_orchestrator._save_cycle_results(
                    short_term_result.get('signals', []),
                    short_term_result.get('execution_results', []),
                    short_term_result.get('portfolio_summary', {}),
                    short_term_result.get('performance_metrics', {})
                )
            
            # Sauvegarder les opportunités émergentes dans la base
            for opportunity in emerging_opportunities:
                await self._save_emerging_opportunity(opportunity)
            
            # Sauvegarder les recommandations long terme
            for recommendation in long_term_recommendations:
                await self._save_long_term_recommendation(recommendation)
            
            logging.debug("Enhanced cycle results saved successfully")
            
        except Exception as e:
            logging.error(f"Error saving enhanced results: {e}")
    
    async def _save_emerging_opportunity(self, opportunity):
        """Sauvegarde une opportunité émergente en base"""
        
        try:
            if self.bot_orchestrator.db_manager.connection:
                cursor = self.bot_orchestrator.db_manager.connection.cursor()
                cursor.execute("""
                    INSERT INTO trading_signals (symbol, signal_type, confidence, price, strategy, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    opportunity.symbol,
                    'emerging_opportunity',
                    opportunity.confidence,
                    0.0,  # Prix à récupérer
                    'emerging_detector',
                    {
                        'score': opportunity.score,
                        'growth_potential': opportunity.growth_potential,
                        'key_drivers': opportunity.key_drivers,
                        'sector': opportunity.sector
                    }
                ))
                cursor.close()
        except Exception as e:
            logging.debug(f"Could not save emerging opportunity: {e}")
    
    async def _save_long_term_recommendation(self, recommendation):
        """Sauvegarde une recommandation long terme en base"""
        
        try:
            if self.bot_orchestrator.db_manager.connection:
                cursor = self.bot_orchestrator.db_manager.connection.cursor()
                cursor.execute("""
                    INSERT INTO trading_signals (symbol, signal_type, confidence, price, strategy, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    recommendation.symbol,
                    'long_term_buy' if recommendation.recommendation in ['Buy', 'Strong Buy'] else 'long_term_hold',
                    recommendation.confidence,
                    recommendation.target_price_3y,
                    'long_term_analyzer',
                    {
                        'overall_score': recommendation.overall_score,
                        'expected_return_3y': recommendation.expected_return_3y,
                        'investment_thesis': recommendation.investment_thesis,
                        'sector': recommendation.sector,
                        'recommendation': recommendation.recommendation
                    }
                ))
                cursor.close()
        except Exception as e:
            logging.debug(f"Could not save long-term recommendation: {e}")
    
    async def run_continuous(self):
        """Exécute le bot en continu avec cycles améliorés"""
        
        max_cycles = getattr(self.config, 'MAX_CYCLES', 0)
        refresh_interval = getattr(self.config, 'REFRESH_INTERVAL', 300)  # 5 minutes par défaut
        
        logging.info(f"Starting continuous enhanced trading (max_cycles: {max_cycles}, interval: {refresh_interval}s)")
        
        while not self.killer.kill_now and (max_cycles == 0 or self.cycle_count < max_cycles):
            try:
                # Exécuter le cycle amélioré
                cycle_result = await self.run_enhanced_cycle()
                
                if 'error' in cycle_result:
                    logging.error(f"Cycle {self.cycle_count} failed: {cycle_result['error']}")
                else:
                    logging.info(f"Enhanced cycle {self.cycle_count} completed successfully")
                
                # Attendre avant le prochain cycle
                if not self.killer.kill_now:
                    logging.info(f"Waiting {refresh_interval}s before next enhanced cycle...")
                    await asyncio.sleep(refresh_interval)
                
            except Exception as e:
                logging.error(f"Critical error in enhanced cycle {self.cycle_count}: {e}", exc_info=True)
                await asyncio.sleep(300)  # Attendre 5 minutes en cas d'erreur critique
    
    async def cleanup(self):
        """Nettoyage des ressources"""
        
        logging.info("Starting enhanced bot cleanup...")
        
        try:
            if self.bot_orchestrator:
                await self.bot_orchestrator.cleanup()
            
            # Export final dashboard if available
            if self.dashboard:
                try:
                    self.dashboard.export_to_csv()
                    logging.info("Final dashboard exported to CSV")
                except Exception as e:
                    logging.error(f"Dashboard export failed: {e}")
            
            logging.info("Enhanced bot cleanup completed")
            
        except Exception as e:
            logging.error(f"Error during enhanced cleanup: {e}")

async def main():
    """Point d'entrée principal pour le bot amélioré"""
    
    # Configuration et logging
    setup_logging()
    create_directories()
    
    # Créer répertoires supplémentaires
    Path("dashboard_data").mkdir(exist_ok=True)
    Path("exports").mkdir(exist_ok=True)
    
    # Charger configuration
    config = Config()
    
    # Vérifier les nouvelles fonctionnalités
    trading_mode = getattr(config, 'TRADING_MODE', 'fast_mode')
    logging.info(f"Enhanced Trading Bot starting in {trading_mode} mode")
    
    enhanced_bot = None
    
    try:
        # Créer et initialiser le bot amélioré
        enhanced_bot = EnhancedTradingBot(config)
        await enhanced_bot.initialize()
        
        # Lancer en mode continu
        await enhanced_bot.run_continuous()
        
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt, shutting down enhanced bot...")
    except Exception as e:
        logging.error(f"Critical error in enhanced main: {e}", exc_info=True)
        return 1
    finally:
        if enhanced_bot:
            await enhanced_bot.cleanup()
        logging.info("Enhanced Trading Bot shutdown complete")
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)