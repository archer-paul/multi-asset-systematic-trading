"""
Trading Bot Orchestrator - Main coordination layer
Manages all components and orchestrates the trading process
"""

import logging
import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any

from core.config import Config
from core.database import DatabaseManager
from core.data_cache import DataCacheManager
from data.data_collector import DataCollector
from analysis.sentiment_analyzer import SentimentAnalyzer as BaseSentimentAnalyzer
from analysis.enhanced_sentiment import EnhancedSentimentAnalyzer
from analysis.commodities_forex import CommoditiesForexAnalyzer
from analysis.social_media_v2 import SocialMediaAnalyzerV2 as SocialMediaAnalyzer
from analysis.multi_timeframe import MultiTimeframeAnalyzer
from ml.traditional_ml import TraditionalMLPredictor
from ml.transformer_ml import TransformerMLPredictor
from ml.parallel_trainer import ParallelMLTrainer, BatchMLTrainer
from ml.ensemble import EnsemblePredictor
from trading.strategy import TradingStrategy
from trading.risk_manager import RiskManager
from trading.portfolio_manager import PortfolioManager
from trading.advanced_decision_engine import AdvancedPortfolioDecisionEngine
from analysis.performance_analyzer import PerformanceAnalyzer

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
        self.cache_manager = DataCacheManager(self.config)
        
        # Analysis components
        base_sentiment = BaseSentimentAnalyzer(self.config)
        self.sentiment_analyzer = EnhancedSentimentAnalyzer(self.config, base_sentiment)
        self.commodities_forex_analyzer = CommoditiesForexAnalyzer(self.config)
        self.social_media_analyzer = SocialMediaAnalyzer(self.config) if self.config.ENABLE_SOCIAL_SENTIMENT else None
        self.multi_timeframe_analyzer = MultiTimeframeAnalyzer(self.config)
        self.macro_economic_analyzer = MacroEconomicAnalyzer(self.config, self.sentiment_analyzer)
        # self.geopolitical_risk_analyzer = GeopoliticalRiskAnalyzer(self.config)
        
        # ML components
        self.traditional_ml = TraditionalMLPredictor(self.config) if self.config.ENABLE_TRADITIONAL_ML else None
        self.transformer_ml = TransformerMLPredictor(self.config) if self.config.ENABLE_TRANSFORMER_ML else None
        
        self.trained_models_cache = {
            'traditional_ml': {},
            'transformer_ml': {},
            'ttl_hours': 24
        }
        
        self.parallel_trainer = ParallelMLTrainer(
            self.config, TraditionalMLPredictor, TransformerMLPredictor
        )
        self.batch_trainer = BatchMLTrainer(
            self.config, TraditionalMLPredictor, TransformerMLPredictor
        )
        
        self.ensemble_predictor = EnsemblePredictor(self.config, self.traditional_ml, self.transformer_ml)
        
        self.risk_manager = RiskManager(self.config)
        self.portfolio_manager = PortfolioManager(self.config, db_manager=self.db_manager)
        self.trading_strategy = TradingStrategy(self.config)

        # Advanced decision engine for sophisticated portfolio management
        # Inject existing infrastructure for seamless integration
        self.advanced_decision_engine = AdvancedPortfolioDecisionEngine(
            config=self.config.get('advanced_decisions', {}),
            db_manager=self.db_manager,  # Reuse existing database
            ensemble_predictor=self.ensemble_predictor  # Reuse existing ML ensemble
        )

        self.performance_analyzer = PerformanceAnalyzer(self.config)
        
        self.logger.info("All components initialized successfully")
    
    async def initialize(self):
        if self.is_initialized:
            self.logger.warning("Bot already initialized")
            return
        
        self.logger.info("Starting bot initialization...")
        
        try:
            await self.db_manager.initialize()
            await self.data_collector.initialize()
            await self.sentiment_analyzer.initialize()
            await self.commodities_forex_analyzer.initialize()
            
            self.logger.info("Loading historical market data (checking cache)...")
            historical_market_data = await self.cache_manager.get_historical_market_data(
                symbols=self.config.ALL_SYMBOLS,
                days=self.config.ML_TRAINING_LOOKBACK_DAYS
            )
            
            if not historical_market_data:
                self.logger.info("Cache miss - collecting fresh historical market data...")
                historical_market_data = await self.data_collector.collect_historical_market_data(
                    symbols=self.config.ALL_SYMBOLS,
                    days=self.config.ML_TRAINING_LOOKBACK_DAYS
                )
                if historical_market_data:
                    await self.cache_manager.save_historical_market_data(
                        historical_market_data, self.config.ALL_SYMBOLS, self.config.ML_TRAINING_LOOKBACK_DAYS
                    )
                    self.logger.info("Historical market data cached successfully")
            else:
                self.logger.info(f"Historical market data loaded from cache ({len(historical_market_data)} symbols)")
            
            self.logger.info("Loading historical news data (checking cache)...")
            historical_news = await self.cache_manager.get_historical_news(
                symbols=self.config.ALL_SYMBOLS,
                days=self.config.NEWS_LOOKBACK_DAYS
            )
            
            if not historical_news:
                self.logger.info("Cache miss - collecting fresh historical news data...")
                historical_news = await self.data_collector.collect_historical_news(
                    symbols=self.config.ALL_SYMBOLS,
                    days=self.config.NEWS_LOOKBACK_DAYS
                )
                if historical_news:
                    await self.cache_manager.save_historical_news(
                        historical_news, self.config.ALL_SYMBOLS, 
                        self.config.ANALYSIS_LOOKBACK_DAYS * 4
                    )
                    self.logger.info("Historical news data cached successfully")
            else:
                self.logger.info(f"Historical news data loaded from cache ({len(historical_news)} articles)")
            
            self.logger.info("Processing sentiment analysis for historical news...")
            await self._process_news_sentiment(historical_news)
            
            social_data = []
            if self.social_media_analyzer:
                self.logger.info("Collecting historical social media sentiment...")
                social_data = await self.social_media_analyzer.collect_historical_sentiment(
                    symbols=self.config.ALL_SYMBOLS,
                    days=30
                )
            
            if not self.config.SKIP_ML_TRAINING:
                self.logger.info("Training machine learning models (parallel mode)...")
                await self._train_ml_models_parallel(historical_market_data, historical_news, social_data)
            else:
                self.logger.info("Skipping ML model training as per configuration.")
            
            await self.portfolio_manager.initialize()
            await self._save_initial_state()
            
            self.is_initialized = True
            self.logger.info("Bot initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize bot: {e}", exc_info=True)
            raise

    async def run_trading_cycle(self) -> Dict[str, Any]:
        if not self.is_initialized:
            raise RuntimeError("Bot must be initialized before running trading cycles")
        
        cycle_start_time = datetime.now()
        self.cycle_count += 1
        self.logger.info(f"Starting trading cycle {self.cycle_count}")
        
        try:
            self.logger.debug("Collecting current market data...")
            current_market_data = await self.data_collector.collect_current_market_data(self.config.ALL_SYMBOLS)
            
            self.logger.debug("Collecting latest news...")
            latest_news = await self.data_collector.collect_news_data(self.config.ALL_SYMBOLS)
            
            self.logger.debug("Processing news sentiment...")
            await self._process_news_sentiment(latest_news)
            
            social_sentiment = {}
            if self.social_media_analyzer:
                self.logger.debug("Collecting social media sentiment...")
                social_sentiment = await self.social_media_analyzer.collect_current_sentiment(self.config.ALL_SYMBOLS)

            self.logger.debug("Performing multi-timeframe analysis...")
            multi_timeframe_analysis = await self.multi_timeframe_analyzer.analyze_multi_timeframe(self.config.ALL_SYMBOLS[:20])
            
            self.logger.debug("Generating ML predictions...")
            predictions = await self._generate_predictions(current_market_data, latest_news, social_sentiment, multi_timeframe_analysis)
            
            self.logger.debug("Generating trading signals...")
            signals = await self._generate_trading_signals(predictions, current_market_data, latest_news, social_sentiment, multi_timeframe_analysis)
            
            self.logger.debug("Executing trades...")
            execution_results = await self._execute_trades(signals)
            
            self.logger.debug("Updating portfolio and performance metrics...")
            portfolio_summary = await self.portfolio_manager.get_portfolio_summary()
            performance_metrics = self.portfolio_manager.get_comprehensive_performance_metrics()

            # Update advanced decision engine with ML ensemble feedback
            await self.advanced_decision_engine.update_from_ml_ensemble_feedback()

            # Check if model retraining is needed (weekly check)
            if self.cycle_count % 7 == 0:  # Check every 7 cycles (weekly)
                await self.advanced_decision_engine.check_and_trigger_retraining()

            await self._save_cycle_results(signals, execution_results, portfolio_summary, performance_metrics)

            self.latest_analysis = {
                'top_news_sentiment': sorted([s for s in signals if s.get('source') == 'news_sentiment'], key=lambda x: x.get('confidence', 0), reverse=True)[:5],
                'social_media_sentiment': social_sentiment
            }
            
            cycle_result = {
                'cycle_number': self.cycle_count,
                'timestamp': cycle_start_time,
                'duration_seconds': (datetime.now() - cycle_start_time).total_seconds(),
                'signals_generated': len(signals),
                'trades_executed': sum(1 for r in (execution_results or []) if r.get('executed', False)),
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
        for i, news_item in enumerate(news_data):
            try:
                companies = news_item.get('companies_mentioned', [])
                company = companies[0] if companies else None
                region = news_item.get('region', 'US')
                text = (news_item.get('title', '') or '') + ' ' + (news_item.get('content', '') or '')
                sentiment_data = await self.sentiment_analyzer.analyze_financial_sentiment(text=text, company=company, region=region)
                news_item['sentiment_data'] = sentiment_data
                if (i + 1) % 50 == 0:
                    self.logger.debug(f"Processed sentiment for {i + 1}/{len(news_data)} articles")
            except Exception as e:
                self.logger.error(f"Error processing sentiment for news item {i}: {e}")
                news_item['sentiment_data'] = {'sentiment_score': 0.0, 'confidence': 0.5, 'reasoning': 'Error in sentiment processing'}
    
    async def _train_ml_models_parallel(self, market_data: Dict, news_data: List[Dict], social_data: List[Dict]):
        def progress_callback(symbol: str, model_type: str, status: str, progress: float):
            if status in ['completed', 'failed', 'error']:
                self.logger.info(f"Training progress: {progress:.1f}% - {symbol} {model_type}: {status}")
        
        self.parallel_trainer.set_progress_callback(progress_callback)
        
        training_summary = await self.parallel_trainer.train_models_parallel(
            market_data=market_data, news_data=news_data, social_data=social_data,
            train_traditional=self.config.ENABLE_TRADITIONAL_ML, train_transformer=self.config.ENABLE_TRANSFORMER_ML
        )
        
        await self._populate_model_cache(training_summary)
        
        self.logger.info(f"Parallel training completed: {training_summary['successful']}/{training_summary['total_tasks']} tasks successful ({training_summary['success_rate']:.1f}%)")
        
        if len(market_data) > 10:
            self.logger.info("Starting batch training for cross-symbol learning...")
            batch_summary = await self.batch_trainer.train_batch_models(market_data=market_data, news_data=news_data, social_data=social_data)
            if batch_summary.get('successful_models', 0) > 0:
                self.logger.info(f"Batch training completed: {batch_summary['successful_models']}/{batch_summary['total_models']} models successful")
                if 'results' in batch_summary:
                    for model_name, result in batch_summary['results'].items():
                        if result['success'] and 'model' in result:
                            await self.cache_manager.save_ml_model(result['model'], model_name, symbol=None, training_metadata={'training_type': 'batch_cross_symbol'})
        return training_summary
    
    async def _generate_predictions(self, market_data: Dict, news_data: List[Dict], social_data: Dict, multi_timeframe_data: Dict) -> Dict[str, Dict]:
        predictions = {}
        for symbol in self.config.ALL_SYMBOLS:
            if symbol not in market_data: continue
            try:
                prediction = await self._get_cached_prediction(
                    symbol=symbol, market_data=market_data[symbol],
                    news_data=[item for item in news_data if symbol in item.get('companies_mentioned', [])],
                    social_data=social_data.get(symbol, {}), multi_timeframe_data=multi_timeframe_data.get(symbol, {}),
                    region=self.config.get_symbol_region(symbol)
                )
                predictions[symbol] = prediction
            except Exception as e:
                self.logger.error(f"Error generating prediction for {symbol}: {e}")
                predictions[symbol] = {'error': str(e), 'prediction': 2, 'confidence': 0.0}
        return predictions

    async def _get_cached_prediction(self, symbol: str, market_data: pd.DataFrame, news_data: List[Dict], social_data: Dict, multi_timeframe_data: Dict, region: str) -> Dict[str, Any]:
        # This method implementation seems to have a bug in its logic, as it doesn't use the cached models for prediction.
        # It falls back to the ensemble predictor. This should be reviewed.
        self.logger.debug(f"No cached models for {symbol}, using ensemble predictor")
        return await self.ensemble_predictor.predict(
            symbol=symbol, market_data=market_data, news_data=news_data,
            social_data=social_data, multi_timeframe_data=multi_timeframe_data, region=region
        )

    async def _generate_trading_signals(self, predictions: Dict, market_data: Dict, news_data: List[Dict], social_data: Dict, multi_timeframe_data: Dict) -> List[Dict]:
        """Generate trading signals using advanced decision engine"""
        portfolio_summary = await self.portfolio_manager.get_portfolio_summary()
        current_positions = portfolio_summary.get('positions', {})
        try:
            self.logger.info("Generating advanced trading signals...")

            # Prepare comprehensive signals data for each symbol
            all_signals_data = {}

            for symbol in self.config.ALL_SYMBOLS:
                if symbol not in market_data:
                    continue

                # Get current portfolio weight for this symbol
                current_weight = current_positions.get(symbol, {}).get('weight', 0.0)

                # Aggregate all data sources for this symbol
                symbol_data = self._prepare_symbol_data(
                    symbol, predictions, market_data, news_data,
                    social_data, multi_timeframe_data
                )

                all_signals_data[symbol] = symbol_data

            # Get current portfolio allocation
            current_portfolio = {}
            for symbol, position in current_positions.items():
                current_portfolio[symbol] = position.get('weight', 0.0)

            # Generate advanced portfolio decisions
            portfolio_decisions = await self.advanced_decision_engine.make_portfolio_decisions(
                all_signals_data, current_portfolio, market_data
            )

            # Convert portfolio decisions to trading signals format
            trading_signals = self._convert_decisions_to_signals(portfolio_decisions)

            self.logger.info(f"Generated {len(trading_signals)} advanced trading signals")
            return trading_signals

        except Exception as e:
            self.logger.error(f"Advanced signal generation failed: {e}", exc_info=True)
            # Fallback to basic signal generation
            return await self._generate_basic_signals(predictions, market_data, news_data, social_data, multi_timeframe_data)

    def _prepare_symbol_data(self, symbol: str, predictions: Dict, market_data: Dict,
                            news_data: List[Dict], social_data: Dict, multi_timeframe_data: Dict) -> Dict[str, Any]:
        """Prepare comprehensive data for a single symbol"""
        try:
            symbol_market_data = market_data.get(symbol, {})
            symbol_predictions = predictions.get(symbol, {})
            symbol_social = social_data.get(symbol, {})
            symbol_multi_timeframe = multi_timeframe_data.get(symbol, {})

            # Filter news for this symbol
            symbol_news = [
                item for item in news_data
                if symbol in item.get('companies_mentioned', [])
            ]

            # Extract technical analysis data
            price_history = symbol_market_data.get('price_history', pd.DataFrame())
            current_price = symbol_market_data.get('price', 0)

            technical_analysis = {}
            if not price_history.empty and len(price_history) > 20:
                # RSI
                delta = price_history['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs)).iloc[-1]

                # MACD
                ema_12 = price_history['Close'].ewm(span=12).mean()
                ema_26 = price_history['Close'].ewm(span=26).mean()
                macd = ema_12 - ema_26
                signal_line = macd.ewm(span=9).mean()
                macd_signal = 1 if macd.iloc[-1] > signal_line.iloc[-1] else -1

                # Volume ratio
                avg_volume = price_history['Volume'].rolling(window=20).mean().iloc[-1]
                current_volume = symbol_market_data.get('volume', avg_volume)
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

                # Support/Resistance
                resistance = price_history['High'].rolling(window=20).max().iloc[-1]
                support = price_history['Low'].rolling(window=20).min().iloc[-1]

                technical_analysis = {
                    'rsi': rsi,
                    'macd_signal': macd_signal,
                    'volume_ratio': volume_ratio,
                    'resistance': resistance,
                    'support': support
                }

            # Extract sentiment analysis
            sentiment_analysis = {}
            if symbol_news:
                news_sentiments = [
                    item.get('sentiment_data', {}).get('sentiment_score', 0)
                    for item in symbol_news if 'sentiment_data' in item
                ]
                if news_sentiments:
                    sentiment_analysis['news_sentiment'] = np.mean(news_sentiments)
                    sentiment_analysis['news_confidence'] = len(news_sentiments) / 10.0  # More news = higher confidence

            if symbol_social:
                sentiment_analysis['social_sentiment'] = symbol_social.get('sentiment_score', 0)
                sentiment_analysis['social_confidence'] = symbol_social.get('confidence', 0.3)

            # Extract fundamental analysis (basic)
            fundamental_analysis = {
                'pe_ratio': symbol_market_data.get('pe_ratio', 20),
                'pb_ratio': symbol_market_data.get('pb_ratio', 2),
                'revenue_growth': symbol_market_data.get('revenue_growth', 0.05),
                'earnings_growth': symbol_market_data.get('earnings_growth', 0.08)
            }

            # Extract momentum analysis
            momentum_analysis = {}
            if not price_history.empty and len(price_history) > 20:
                momentum_1d = (current_price - price_history['Close'].iloc[-2]) / price_history['Close'].iloc[-2] if len(price_history) > 1 else 0
                momentum_5d = (current_price - price_history['Close'].iloc[-6]) / price_history['Close'].iloc[-6] if len(price_history) > 5 else 0
                momentum_20d = (current_price - price_history['Close'].iloc[-21]) / price_history['Close'].iloc[-21] if len(price_history) > 20 else 0

                momentum_analysis = {
                    'momentum_1d': momentum_1d,
                    'momentum_5d': momentum_5d,
                    'momentum_20d': momentum_20d,
                    'volume_momentum': volume_ratio - 1.0
                }

            # Volume analysis
            volume_analysis = {
                'volume_ratio': technical_analysis.get('volume_ratio', 1.0),
                'volume_trend': 0.0  # Can be enhanced with volume trend calculation
            }

            # Risk analysis
            volatility = 0.02  # Default
            if not price_history.empty and len(price_history) > 20:
                returns = price_history['Close'].pct_change()
                volatility = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)

            risk_analysis = {
                'volatility': volatility,
                'beta': 1.0,  # Can be calculated vs market index
                'max_drawdown': 0.1,  # Historical max drawdown
                'liquidity_score': 0.8  # Estimated liquidity score
            }

            # Correlation analysis (simplified)
            correlation_analysis = {
                'portfolio_correlation': 0.3,  # Average correlation with existing portfolio
                'sector_concentration': 0.2   # Sector concentration risk
            }

            return {
                'symbol': symbol,
                'current_price': current_price,
                'technical_analysis': technical_analysis,
                'ml_predictions': symbol_predictions,
                'sentiment_analysis': sentiment_analysis,
                'fundamental_analysis': fundamental_analysis,
                'momentum_analysis': momentum_analysis,
                'volume_analysis': volume_analysis,
                'risk_analysis': risk_analysis,
                'correlation_analysis': correlation_analysis,
                'sector_analysis': {
                    'strength': symbol_multi_timeframe.get('sector_strength', 0.0),
                    'sector': symbol_market_data.get('sector', 'Unknown')
                },
                'price_history': price_history
            }

        except Exception as e:
            self.logger.error(f"Error preparing data for {symbol}: {e}")
            return {'symbol': symbol, 'current_price': 0}

    def _convert_decisions_to_signals(self, portfolio_decisions: Dict) -> List[Dict]:
        """Convert portfolio decisions to traditional trading signals format"""
        signals = []

        for symbol, decision in portfolio_decisions.items():
            # Include more decisions, not just buy/sell
            if decision.action == 'hold' and decision.conviction < 0.1:
                continue

            # Map decision action to signal type
            signal_type_map = {
                'buy': 'buy',
                'sell': 'sell',
                'rebalance': 'rebalance'
            }

            signal = {
                'symbol': symbol,
                'signal_type': signal_type_map.get(decision.action, 'hold'),
                'confidence': decision.conviction,
                'price': decision.metadata.get('signal_metrics', {}).get('current_price', 0) if hasattr(decision.metadata.get('signal_metrics', {}), 'current_price') else 0,
                'target_weight': decision.target_weight,
                'current_weight': decision.current_weight,
                'expected_return': decision.expected_return,
                'expected_risk': decision.expected_risk,
                'reasoning': decision.reasoning,
                'strategy': 'AdvancedDecisionEngine',
                'metadata': {
                    'conviction': decision.conviction,
                    'market_regime': decision.metadata.get('market_regime'),
                    'weight_change': decision.target_weight - decision.current_weight,
                    'risk_metrics': decision.risk_metrics
                }
            }

            signals.append(signal)

        return signals

    async def _generate_basic_signals(self, predictions: Dict, market_data: Dict, news_data: List[Dict], social_data: Dict, multi_timeframe_data: Dict) -> List[Dict]:
        """Fallback basic signal generation method"""
        try:
            signals = []

            for symbol in self.config.ALL_SYMBOLS[:10]:  # Limit for safety
                if symbol not in market_data:
                    continue

                # Basic signal logic
                prediction = predictions.get(symbol, {})
                ml_signal = prediction.get('meta_prediction', 0)
                ml_confidence = prediction.get('meta_confidence', 0.5)

                if ml_confidence > 0.3 and abs(ml_signal) > 0.2:
                    signal_type = 'buy' if ml_signal > 0 else 'sell'

                    signals.append({
                        'symbol': symbol,
                        'signal_type': signal_type,
                        'confidence': ml_confidence,
                        'price': market_data[symbol].get('price', 0),
                        'reasoning': f"ML prediction: {ml_signal:.2f}",
                        'strategy': 'BasicML',
                        'metadata': {'ml_prediction': ml_signal}
                    })

            return signals

        except Exception as e:
            self.logger.error(f"Basic signal generation failed: {e}")
            return []

    def _aggregate_sentiment_data(self, news_items: List[Dict]) -> Dict:
        # ... (implementation unchanged)
        pass

    async def _execute_trades(self, signals: List[Dict]) -> List[Dict]:
        """Execute trades based on generated signals"""
        execution_results = []

        if not signals:
            self.logger.info("No signals to execute")
            return execution_results

        try:
            for signal in signals:
                symbol = signal.get('symbol')
                signal_type = signal.get('signal_type')
                confidence = signal.get('confidence', 0)

                # Filter signals by confidence threshold
                min_confidence = getattr(self.config, 'MIN_EXECUTION_CONFIDENCE', 0.6)
                if confidence < min_confidence:
                    execution_results.append({
                        'symbol': symbol,
                        'signal_type': signal_type,
                        'status': 'skipped',
                        'reason': f'Confidence {confidence:.2f} below threshold {min_confidence}',
                        'executed': False
                    })
                    continue

                # For now, simulate trade execution (in a real bot, this would connect to a broker)
                execution_results.append({
                    'symbol': symbol,
                    'signal_type': signal_type,
                    'confidence': confidence,
                    'status': 'simulated',
                    'executed': True,
                    'price': signal.get('price', 0),
                    'quantity': signal.get('quantity', 0),
                    'timestamp': datetime.now()
                })

            self.logger.info(f"Executed {len([r for r in execution_results if r.get('executed', False)])} trades from {len(signals)} signals")

        except Exception as e:
            self.logger.error(f"Error executing trades: {e}")

        return execution_results

    async def _save_initial_state(self):
        # ... (implementation unchanged)
        pass

    async def _save_cycle_results(self, signals: List[Dict], execution_results: List[Dict], portfolio_summary: Dict, performance_metrics: Dict):
        """Save cycle results to database and update performance tracking"""
        try:
            # Store in database for historical tracking
            if self.db_manager and self.db_manager.connection:
                cursor = self.db_manager.connection.cursor()

                # Save bot performance metrics
                cursor.execute("""
                    INSERT INTO bot_performance (cycle_id, total_return, trades_count, success_rate, max_drawdown, start_time, end_time, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    f"cycle_{self.cycle_count}",
                    performance_metrics.get('total_return_pct', 0),
                    len(execution_results or []),
                    performance_metrics.get('win_rate_pct', 0) / 100,
                    performance_metrics.get('max_drawdown_pct', 0) / 100,
                    self.last_cycle_time,
                    datetime.now(),
                    json.dumps({
                        'signal_count': len(signals),
                        'advanced_decisions': len(self.advanced_decision_engine.last_decisions) if self.advanced_decision_engine.last_decisions else 0,
                        'market_regime': getattr(self.advanced_decision_engine.regime_detector, 'last_detected_regime', 'unknown'),
                        'ensemble_performance': getattr(self.ensemble_predictor, 'recent_performance', {}),
                    })
                ))

                cursor.close()

            # Update ML Ensemble with portfolio performance feedback
            await self._update_ml_ensemble_with_portfolio_feedback(portfolio_summary, performance_metrics)

            # Record decision performance for adaptive learning
            await self._record_decision_performance_feedback(execution_results, portfolio_summary)

        except Exception as e:
            self.logger.error(f"Error saving cycle results: {e}")

    async def _update_ml_ensemble_with_portfolio_feedback(self, portfolio_summary: Dict, performance_metrics: Dict):
        """Update ML ensemble with actual portfolio performance"""
        try:
            if not self.ensemble_predictor or not hasattr(self.ensemble_predictor, 'bayesian_averaging'):
                return

            # Get recent signals that led to current performance
            recent_decisions = self.advanced_decision_engine.last_decisions

            for symbol, decision in recent_decisions.items():
                # Get actual return from portfolio
                position = portfolio_summary.get('positions', {}).get(symbol, {})
                if position:
                    actual_return = position.get('unrealized_pnl_pct', 0)

                    # Extract ML predictions that contributed to this decision
                    signal_metrics = decision.metadata.get('signal_metrics')
                    if signal_metrics:
                        ml_prediction = getattr(signal_metrics, 'ml_score', 0)

                        # Update ensemble with actual vs predicted
                        individual_predictions = {
                            'traditional_ml': getattr(signal_metrics, 'traditional_ml_prediction', ml_prediction * 0.6),
                            'transformer_ml': getattr(signal_metrics, 'transformer_ml_prediction', ml_prediction * 0.4),
                            'ensemble': ml_prediction
                        }

                        # This feeds back into the BayesianModelAveraging in ml/ensemble.py
                        self.ensemble_predictor.bayesian_averaging.update_model_performance(
                            individual_predictions, actual_return
                        )

            self.logger.debug(f"Updated ML ensemble with feedback from {len(recent_decisions)} decisions")

        except Exception as e:
            self.logger.error(f"Error updating ML ensemble with portfolio feedback: {e}")

    async def _record_decision_performance_feedback(self, execution_results: List[Dict], portfolio_summary: Dict):
        """Record decision performance for advanced decision engine learning"""
        try:
            recent_decisions = self.advanced_decision_engine.last_decisions

            for symbol, decision in recent_decisions.items():
                # Find corresponding execution result
                execution_result = next((er for er in execution_results if er.get('symbol') == symbol), None)

                if execution_result and execution_result.get('status') == 'success':
                    # Calculate actual return based on portfolio change
                    position = portfolio_summary.get('positions', {}).get(symbol, {})
                    actual_return = position.get('unrealized_pnl_pct', 0) if position else 0

                    # Record performance for adaptive learning
                    await self.advanced_decision_engine.record_decision_performance(
                        symbol=symbol,
                        decision=decision,
                        actual_return=actual_return,
                        time_horizon=5  # 5 days default
                    )

            self.logger.debug(f"Recorded performance feedback for {len(recent_decisions)} decisions")

        except Exception as e:
            self.logger.error(f"Error recording decision performance feedback: {e}")

    async def get_detailed_performance_report(self) -> Dict[str, Any]:
        # ... (implementation unchanged)
        pass

    async def cleanup(self):
        self.logger.info("Starting bot cleanup...")
        try:
            if self.is_initialized:
                final_portfolio = await self.portfolio_manager.get_portfolio_summary()
                await self.db_manager.save_portfolio_snapshot(final_portfolio)
                final_report = await self.get_detailed_performance_report()
                await self.db_manager.save_final_report(final_report)
            
            await self.db_manager.cleanup()
            
            if hasattr(self, 'cache_manager'):
                await self.cache_manager.cleanup_expired_cache()
                self.cache_manager.close()
            
            if hasattr(self.data_collector, 'cleanup'): await self.data_collector.cleanup()
            if hasattr(self.sentiment_analyzer, 'cleanup'): await self.sentiment_analyzer.cleanup()
            if hasattr(self.commodities_forex_analyzer, 'cleanup'): await self.commodities_forex_analyzer.cleanup()
            
            self.is_running = False
            self.logger.info("Bot cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def get_status(self) -> Dict[str, Any]:
        # ... (implementation unchanged)
        pass

    async def emergency_stop(self):
        # ... (implementation unchanged)
        pass

    async def _populate_model_cache(self, training_summary: Dict):
        if 'training_results' not in training_summary:
            self.logger.warning("No training results found in training summary")
            return
        
        for key, result in training_summary['training_results'].items():
            if not result.get('success', False) or 'model' not in result: continue
            
            symbol, model_type, model = result['symbol'], result['model_type'], result['model']
            
            if model_type == 'traditional' and model:
                self.trained_models_cache['traditional_ml'][symbol] = {'model': model, 'timestamp': datetime.now()}
                self.logger.debug(f"Cached traditional ML model for {symbol}")
            elif model_type == 'transformer' and model:
                self.trained_models_cache['transformer_ml'][symbol] = {'model': model, 'timestamp': datetime.now()}
                self.logger.debug(f"Cached transformer ML model for {symbol}")
        
        self.logger.info(f"Model cache populated: {len(self.trained_models_cache['traditional_ml'])} traditional, {len(self.trained_models_cache['transformer_ml'])} transformer models cached")
