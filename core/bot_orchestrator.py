"""
Trading Bot Orchestrator - Main coordination layer
Manages all components and orchestrates the trading process
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any

from core.config import Config
from core.database import DatabaseManager
from data.data_collector import DataCollector
from analysis.sentiment_analyzer import SentimentAnalyzer
from analysis.social_media_v2 import SocialMediaAnalyzerV2 as SocialMediaAnalyzer
from analysis.multi_timeframe import MultiTimeframeAnalyzer
from ml.traditional_ml import TraditionalMLPredictor
from ml.transformer_ml import TransformerMLPredictor
from ml.ensemble import EnsemblePredictor
from trading.strategy import TradingStrategy
from trading.risk_manager import RiskManager
from trading.portfolio_manager import PortfolioManager
from analytics.performance_analyzer import PerformanceAnalyzer

class TradingBotOrchestrator:
    """
    Main orchestrator that coordinates all trading bot components
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # State tracking
        self.is_initialized = False
        self.is_running = False
        self.cycle_count = 0
        self.last_cycle_time = None
        
    def _initialize_components(self):
        """Initialize all bot components"""
        
        # Core infrastructure
        self.db_manager = DatabaseManager(self.config)
        self.data_collector = DataCollector(self.config)
        
        # Analysis components
        self.sentiment_analyzer = SentimentAnalyzer(self.config)
        self.social_media_analyzer = SocialMediaAnalyzer(self.config) if self.config.ENABLE_SOCIAL_SENTIMENT else None
        self.multi_timeframe_analyzer = MultiTimeframeAnalyzer(self.config)
        
        # ML components
        self.traditional_ml = TraditionalMLPredictor(self.config) if self.config.ENABLE_TRADITIONAL_ML else None
        self.transformer_ml = TransformerMLPredictor(self.config) if self.config.ENABLE_TRANSFORMER_ML else None
        self.ensemble_predictor = EnsemblePredictor(self.config, self.traditional_ml, self.transformer_ml)
        
        # Trading components
        self.risk_manager = RiskManager(self.config)
        self.portfolio_manager = PortfolioManager(self.config)
        self.trading_strategy = TradingStrategy(self.config)
        
        # Analytics
        self.performance_analyzer = PerformanceAnalyzer(self.config)
        
        self.logger.info("All components initialized successfully")
    
    async def initialize(self):
        """Initialize the trading bot with historical data and model training"""
        if self.is_initialized:
            self.logger.warning("Bot already initialized")
            return
        
        self.logger.info("Starting bot initialization...")
        
        try:
            # Initialize database
            await self.db_manager.initialize()
            
            # Initialize data collection
            await self.data_collector.initialize()
            
            # Collect historical data for training
            self.logger.info("Collecting historical market data...")
            historical_market_data = await self.data_collector.collect_historical_market_data(
                symbols=self.config.ALL_SYMBOLS,
                days=180  # 6 months for training
            )
            
            self.logger.info("Collecting historical news data...")
            historical_news = await self.data_collector.collect_historical_news(
                symbols=self.config.ALL_SYMBOLS,
                days=self.config.ANALYSIS_LOOKBACK_DAYS * 4  # More news for training
            )
            
            # Process sentiment for historical news
            self.logger.info("Processing sentiment analysis for historical news...")
            await self._process_news_sentiment(historical_news)
            
            # Collect social media data if enabled
            social_data = []
            if self.social_media_analyzer:
                self.logger.info("Collecting historical social media sentiment...")
                social_data = await self.social_media_analyzer.collect_historical_sentiment(
                    symbols=self.config.ALL_SYMBOLS,
                    days=30
                )
            
            # Train ML models
            self.logger.info("Training machine learning models...")
            await self._train_ml_models(historical_market_data, historical_news, social_data)
            
            # Initialize portfolio
            await self.portfolio_manager.initialize()
            
            # Save initial state
            await self._save_initial_state()
            
            self.is_initialized = True
            self.logger.info("Bot initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize bot: {e}", exc_info=True)
            raise
    
    async def run_trading_cycle(self) -> Dict[str, Any]:
        """Execute one complete trading cycle"""
        if not self.is_initialized:
            raise RuntimeError("Bot must be initialized before running trading cycles")
        
        cycle_start_time = datetime.now()
        self.cycle_count += 1
        
        self.logger.info(f"Starting trading cycle {self.cycle_count}")
        
        try:
            # Step 1: Collect current market data
            self.logger.debug("Collecting current market data...")
            current_market_data = await self.data_collector.collect_current_market_data(
                symbols=self.config.ALL_SYMBOLS
            )
            
            # Step 2: Collect latest news
            self.logger.debug("Collecting latest news...")
            latest_news = await self.data_collector.collect_latest_news(
                symbols=self.config.ALL_SYMBOLS
            )
            
            # Step 3: Process news sentiment
            self.logger.debug("Processing news sentiment...")
            await self._process_news_sentiment(latest_news)
            
            # Step 4: Collect social media sentiment if enabled
            social_sentiment = {}
            if self.social_media_analyzer:
                self.logger.debug("Collecting social media sentiment...")
                social_sentiment = await self.social_media_analyzer.collect_current_sentiment(
                    symbols=self.config.ALL_SYMBOLS
                )
            
            # Step 5: Perform multi-timeframe analysis
            self.logger.debug("Performing multi-timeframe analysis...")
            multi_timeframe_analysis = await self.multi_timeframe_analyzer.analyze_multi_timeframe(
                symbols=self.config.ALL_SYMBOLS[:20]  # Limit to 20 symbols for performance
            )
            
            # Step 6: Generate predictions for all symbols
            self.logger.debug("Generating ML predictions...")
            predictions = await self._generate_predictions(
                current_market_data, latest_news, social_sentiment, multi_timeframe_analysis
            )
            
            # Step 6: Generate trading signals
            self.logger.debug("Generating trading signals...")
            signals = await self._generate_trading_signals(
                predictions, current_market_data, latest_news, social_sentiment, multi_timeframe_analysis
            )
            
            # Step 7: Execute trades
            self.logger.debug("Executing trades...")
            execution_results = await self._execute_trades(signals)
            
            # Step 8: Update portfolio and performance metrics
            self.logger.debug("Updating portfolio and performance metrics...")
            portfolio_summary = await self.portfolio_manager.get_portfolio_summary()
            performance_metrics = await self.performance_analyzer.calculate_current_metrics()
            
            # Step 9: Save cycle results
            await self._save_cycle_results(
                signals, execution_results, portfolio_summary, performance_metrics
            )
            
            # Prepare cycle result
            cycle_result = {
                'cycle_number': self.cycle_count,
                'timestamp': cycle_start_time,
                'duration_seconds': (datetime.now() - cycle_start_time).total_seconds(),
                'signals_generated': len(signals),
                'trades_executed': sum(1 for r in execution_results if r.get('executed', False)),
                'portfolio_summary': portfolio_summary,
                'performance_metrics': performance_metrics,
                'signals': signals,
                'execution_results': execution_results
            }
            
            self.last_cycle_time = cycle_start_time
            self.logger.info(f"Trading cycle {self.cycle_count} completed successfully")
            
            return cycle_result
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle {self.cycle_count}: {e}", exc_info=True)
            return {
                'cycle_number': self.cycle_count,
                'timestamp': cycle_start_time,
                'error': str(e),
                'duration_seconds': (datetime.now() - cycle_start_time).total_seconds()
            }
    
    async def _process_news_sentiment(self, news_data: List[Dict]):
        """Process sentiment analysis for news data"""
        for i, news_item in enumerate(news_data):
            try:
                companies = news_item.get('companies_mentioned', [])
                company = companies[0] if companies else None
                region = news_item.get('region', 'US')
                
                text = (news_item.get('title', '') or '') + ' ' + (news_item.get('content', '') or '')
                
                sentiment_data = await self.sentiment_analyzer.analyze_financial_sentiment(
                    text=text,
                    company=company,
                    region=region
                )
                
                news_item['sentiment_data'] = sentiment_data
                
                # Log progress for large batches
                if (i + 1) % 50 == 0:
                    self.logger.debug(f"Processed sentiment for {i + 1}/{len(news_data)} articles")
                    
            except Exception as e:
                self.logger.error(f"Error processing sentiment for news item {i}: {e}")
                # Fallback sentiment data
                news_item['sentiment_data'] = {
                    'sentiment_score': 0.0,
                    'confidence': 0.5,
                    'market_impact': 0.5,
                    'urgency': 0.5,
                    'key_themes': [],
                    'risk_factors': [],
                    'timeframe': 'short-term',
                    'sector_impact': 0.5,
                    'reasoning': 'Error in sentiment processing'
                }
    
    async def _train_ml_models(self, market_data: Dict, news_data: List[Dict], social_data: List[Dict]):
        """Train all ML models with historical data"""
        training_results = {}
        
        for symbol in self.config.ALL_SYMBOLS:
            if symbol not in market_data:
                continue
                
            market_df = market_data[symbol]
            if market_df.empty or len(market_df) < 100:
                self.logger.warning(f"Insufficient market data for {symbol}, skipping")
                continue
            
            # Get symbol-specific news and social data
            symbol_news = [
                item for item in news_data 
                if symbol in item.get('companies_mentioned', [])
            ]
            
            symbol_social = [
                item for item in social_data 
                if symbol in item.get('symbols', [])
            ] if social_data else []
            
            region = self.config.get_symbol_region(symbol)
            
            # Train traditional ML
            if self.traditional_ml:
                try:
                    trad_result = await self.traditional_ml.train_model(market_df)
                    if trad_result:
                        training_results[f'{symbol}_traditional'] = trad_result
                        self.logger.info(f"Trained traditional ML for {symbol}")
                except Exception as e:
                    self.logger.error(f"Error training traditional ML for {symbol}: {e}")
            
            # Train transformer ML
            if self.transformer_ml:
                try:
                    trans_result = await self.transformer_ml.train_model(
                        symbol=symbol,
                        market_data=market_df,
                        news_data=symbol_news,
                        social_data=symbol_social,
                        region=region
                    )
                    if trans_result:
                        training_results[f'{symbol}_transformer'] = trans_result
                        self.logger.info(f"Trained transformer ML for {symbol}")
                except Exception as e:
                    self.logger.error(f"Error training transformer ML for {symbol}: {e}")
        
        self.logger.info(f"Model training completed. Trained {len(training_results)} models")
        return training_results
    
    async def _generate_predictions(self, market_data: Dict, news_data: List[Dict], 
                                  social_data: Dict, multi_timeframe_data: Dict) -> Dict[str, Dict]:
        """Generate ML predictions for all symbols"""
        predictions = {}
        
        for symbol in self.config.ALL_SYMBOLS:
            if symbol not in market_data:
                continue
                
            try:
                # Get symbol-specific data
                market_df = market_data[symbol]
                symbol_news = [
                    item for item in news_data 
                    if symbol in item.get('companies_mentioned', [])
                ]
                symbol_social = social_data.get(symbol, {})
                symbol_mtf = multi_timeframe_data.get(symbol, {})
                region = self.config.get_symbol_region(symbol)
                
                # Generate ensemble prediction with multi-timeframe analysis
                prediction = await self.ensemble_predictor.predict(
                    symbol=symbol,
                    market_data=market_df,
                    news_data=symbol_news,
                    social_data=symbol_social,
                    multi_timeframe_data=symbol_mtf,
                    region=region
                )
                
                predictions[symbol] = prediction
                
            except Exception as e:
                self.logger.error(f"Error generating prediction for {symbol}: {e}")
                predictions[symbol] = {
                    'error': str(e),
                    'prediction': 2,  # Hold
                    'confidence': 0.0
                }
        
        return predictions
    
    async def _generate_trading_signals(self, predictions: Dict, market_data: Dict, 
                                      news_data: List[Dict], social_data: Dict, 
                                      multi_timeframe_data: Dict) -> List[Dict]:
        """Generate trading signals based on predictions and other factors"""
        signals = []
        
        for symbol in self.config.ALL_SYMBOLS:
            if symbol not in predictions or symbol not in market_data:
                continue
                
            try:
                prediction = predictions[symbol]
                market_df = market_data[symbol]
                mtf_analysis = multi_timeframe_data.get(symbol, {})
                
                if market_df.empty:
                    continue
                
                # Get current price
                price_col = 'Close_EUR' if 'Close_EUR' in market_df.columns else 'Close'
                current_price = market_df[price_col].iloc[-1]
                
                # Get multi-timeframe signal strength
                mtf_score = mtf_analysis.get('composite_score', 0.0)
                mtf_confidence = mtf_analysis.get('confidence', 0.5)
                mtf_signal = mtf_analysis.get('signal', 'HOLD')
                
                # Aggregate sentiment data
                symbol_news = [
                    item for item in news_data 
                    if symbol in item.get('companies_mentioned', [])
                ]
                
                sentiment_data = self._aggregate_sentiment_data(symbol_news)
                social_sentiment = social_data.get(symbol, {})
                
                # Generate signal with multi-timeframe analysis
                signal = await self.trading_strategy.generate_signal(
                    symbol=symbol,
                    prediction=prediction,
                    current_price=current_price,
                    market_data=market_df,
                    sentiment_data=sentiment_data,
                    social_sentiment=social_sentiment,
                    multi_timeframe_analysis=mtf_analysis,
                    region=self.config.get_symbol_region(symbol)
                )
                
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    def _aggregate_sentiment_data(self, news_items: List[Dict]) -> Dict:
        """Aggregate sentiment data from multiple news items"""
        if not news_items:
            return {
                'sentiment_score': 0.0,
                'market_impact': 0.5,
                'urgency': 0.5,
                'confidence': 0.5,
                'news_count': 0
            }
        
        # Extract sentiment scores
        sentiment_scores = []
        market_impacts = []
        urgencies = []
        confidences = []
        
        for item in news_items:
            sentiment_data = item.get('sentiment_data', {})
            sentiment_scores.append(sentiment_data.get('sentiment_score', 0.0))
            market_impacts.append(sentiment_data.get('market_impact', 0.5))
            urgencies.append(sentiment_data.get('urgency', 0.5))
            confidences.append(sentiment_data.get('confidence', 0.5))
        
        # Calculate aggregated values
        import numpy as np
        
        return {
            'sentiment_score': np.mean(sentiment_scores),
            'market_impact': np.mean(market_impacts),
            'urgency': np.mean(urgencies),
            'confidence': np.mean(confidences),
            'news_count': len(news_items),
            'sentiment_std': np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0.0
        }
    
    async def _execute_trades(self, signals: List[Dict]) -> List[Dict]:
        """Execute trading signals"""
        execution_results = []
        
        for signal in signals:
            try:
                # Risk management check
                risk_approved = await self.risk_manager.approve_trade(signal)
                
                if risk_approved:
                    # Execute trade
                    result = await self.portfolio_manager.execute_trade(signal)
                    execution_results.append(result)
                    
                    if result.get('executed'):
                        self.logger.info(
                            f"Executed {signal['signal_type']} for {signal['symbol']}: "
                            f"{result.get('quantity', 0)} shares at EUR {result.get('price', 0):.2f}"
                        )
                    else:
                        self.logger.debug(f"Trade not executed for {signal['symbol']}: {result.get('reason', 'Unknown')}")
                else:
                    execution_results.append({
                        'symbol': signal['symbol'],
                        'executed': False,
                        'reason': 'Risk management rejection',
                        'timestamp': datetime.now()
                    })
                    
            except Exception as e:
                self.logger.error(f"Error executing trade for {signal['symbol']}: {e}")
                execution_results.append({
                    'symbol': signal['symbol'],
                    'executed': False,
                    'error': str(e),
                    'timestamp': datetime.now()
                })
        
        return execution_results
    
    async def _save_initial_state(self):
        """Save initial bot state to database"""
        try:
            initial_portfolio = await self.portfolio_manager.get_portfolio_summary()
            await self.db_manager.save_portfolio_snapshot(initial_portfolio)
            self.logger.info("Initial state saved to database")
        except Exception as e:
            self.logger.error(f"Error saving initial state: {e}")
    
    async def _save_cycle_results(self, signals: List[Dict], execution_results: List[Dict],
                                portfolio_summary: Dict, performance_metrics: Dict):
        """Save trading cycle results to database"""
        try:
            # Save signals
            for signal in signals:
                await self.db_manager.save_trading_signal(signal)
            
            # Save execution results
            for result in execution_results:
                if result.get('executed'):
                    await self.db_manager.save_trade_execution(result)
            
            # Save portfolio snapshot
            await self.db_manager.save_portfolio_snapshot(portfolio_summary)
            
            # Save performance metrics
            await self.db_manager.save_performance_metrics(performance_metrics)
            
            self.logger.debug("Cycle results saved to database")
            
        except Exception as e:
            self.logger.error(f"Error saving cycle results: {e}")
    
    async def get_detailed_performance_report(self) -> Dict[str, Any]:
        """Generate detailed performance report"""
        try:
            portfolio_summary = await self.portfolio_manager.get_portfolio_summary()
            performance_metrics = await self.performance_analyzer.calculate_comprehensive_metrics()
            
            # Get trading history
            trading_history = await self.db_manager.get_trading_history(limit=100)
            
            # Get model performance
            model_performance = {}
            if self.traditional_ml:
                model_performance['traditional'] = await self.traditional_ml.get_performance_summary()
            if self.transformer_ml:
                model_performance['transformer'] = await self.transformer_ml.get_performance_summary()
            
            return {
                'timestamp': datetime.now(),
                'cycle_count': self.cycle_count,
                'portfolio_summary': portfolio_summary,
                'performance_metrics': performance_metrics,
                'trading_history': trading_history,
                'model_performance': model_performance,
                'config_summary': {
                    'initial_capital': self.config.INITIAL_CAPITAL,
                    'symbols_count': len(self.config.ALL_SYMBOLS),
                    'traditional_ml_enabled': self.config.ENABLE_TRADITIONAL_ML,
                    'transformer_ml_enabled': self.config.ENABLE_TRANSFORMER_ML,
                    'social_sentiment_enabled': self.config.ENABLE_SOCIAL_SENTIMENT
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {'error': str(e), 'timestamp': datetime.now()}
    
    async def cleanup(self):
        """Cleanup resources and save final state"""
        self.logger.info("Starting bot cleanup...")
        
        try:
            if self.is_initialized:
                # Save final portfolio state
                final_portfolio = await self.portfolio_manager.get_portfolio_summary()
                await self.db_manager.save_portfolio_snapshot(final_portfolio)
                
                # Generate final performance report
                final_report = await self.get_detailed_performance_report()
                await self.db_manager.save_final_report(final_report)
            
            # Close database connections
            await self.db_manager.cleanup()
            
            # Close other resources
            if hasattr(self.data_collector, 'cleanup'):
                await self.data_collector.cleanup()
            
            self.is_running = False
            self.logger.info("Bot cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        return {
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'cycle_count': self.cycle_count,
            'last_cycle_time': self.last_cycle_time,
            'config': {
                'initial_capital': self.config.INITIAL_CAPITAL,
                'symbols': self.config.ALL_SYMBOLS,
                'traditional_ml': self.config.ENABLE_TRADITIONAL_ML,
                'transformer_ml': self.config.ENABLE_TRANSFORMER_ML,
                'social_sentiment': self.config.ENABLE_SOCIAL_SENTIMENT
            }
        }
    
    async def emergency_stop(self):
        """Emergency stop with immediate cleanup"""
        self.logger.warning("Emergency stop initiated")
        self.is_running = False
        await self.cleanup()
        self.logger.warning("Emergency stop completed")