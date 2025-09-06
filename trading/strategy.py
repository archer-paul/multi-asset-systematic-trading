"""
Advanced Trading Strategy Module for Trading Bot
Implements multiple sophisticated trading strategies with adaptive algorithms,
portfolio optimization, and advanced signal generation techniques
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

class PositionType(Enum):
    """Position types"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    signal_type: SignalType
    strength: float  # 0-1, confidence level
    price: float
    timestamp: datetime
    strategy: str
    metadata: Dict[str, Any]
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None

@dataclass
class Position:
    """Position data structure"""
    symbol: str
    position_type: PositionType
    size: float
    entry_price: float
    current_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    metadata: Dict[str, Any] = None

class BaseStrategy(ABC):
    """Base class for trading strategies"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.signals_history = []
        self.performance_metrics = {}
        
    @abstractmethod
    async def generate_signal(self, data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate trading signal based on data"""
        pass
    
    @abstractmethod
    def update_parameters(self, performance_feedback: Dict[str, Any]):
        """Update strategy parameters based on performance"""
        pass
    
    def calculate_position_size(self, signal: TradingSignal, portfolio_value: float, 
                              risk_per_trade: float = 0.02) -> float:
        """Calculate position size based on risk management"""
        if signal.stop_loss and signal.price:
            risk_per_share = abs(signal.price - signal.stop_loss)
            if risk_per_share > 0:
                max_shares = (portfolio_value * risk_per_trade) / risk_per_share
                return min(max_shares, portfolio_value * 0.1 / signal.price)  # Max 10% of portfolio per trade
        
        # Default to 5% of portfolio
        return (portfolio_value * 0.05) / signal.price if signal.price > 0 else 0

class MomentumStrategy(BaseStrategy):
    """Advanced momentum-based trading strategy"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("MomentumStrategy", config)
        
        # Strategy parameters
        self.short_window = self.config.get('short_window', 12)
        self.long_window = self.config.get('long_window', 26)
        self.signal_window = self.config.get('signal_window', 9)
        self.momentum_threshold = self.config.get('momentum_threshold', 0.02)
        self.volume_threshold = self.config.get('volume_threshold', 1.5)
        
        # Adaptive parameters
        self.success_rate = 0.5
        self.parameter_history = []
    
    async def generate_signal(self, data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate momentum-based trading signal"""
        try:
            market_data = data.get('market_data', {})
            ml_predictions = data.get('ml_predictions', {})
            sentiment_data = data.get('sentiment_data', {})
            
            symbol = data.get('symbol', 'UNKNOWN')
            current_price = market_data.get('price', 0)
            
            if current_price <= 0:
                return None
            
            # Calculate technical indicators
            price_history = market_data.get('price_history', pd.DataFrame())
            if len(price_history) < self.long_window:
                return None
            
            # MACD calculation
            ema_short = price_history['Close'].ewm(span=self.short_window).mean()
            ema_long = price_history['Close'].ewm(span=self.long_window).mean()
            macd = ema_short - ema_long
            signal_line = macd.ewm(span=self.signal_window).mean()
            macd_histogram = macd - signal_line
            
            # RSI calculation
            delta = price_history['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Volume analysis
            avg_volume = price_history['Volume'].rolling(window=20).mean()
            current_volume = market_data.get('volume', 0)
            volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1
            
            # Momentum calculation
            price_change_5d = (current_price - price_history['Close'].iloc[-5]) / price_history['Close'].iloc[-5]
            price_change_20d = (current_price - price_history['Close'].iloc[-20]) / price_history['Close'].iloc[-20]
            
            # ML prediction integration
            ml_signal = 0
            ml_confidence = 0
            if ml_predictions:
                ml_signal = ml_predictions.get('meta_prediction', 0)
                ml_confidence = ml_predictions.get('meta_confidence', 0)
            
            # Sentiment analysis integration
            sentiment_score = sentiment_data.get('average_sentiment', 0) if sentiment_data else 0
            sentiment_confidence = sentiment_data.get('confidence', 0) if sentiment_data else 0
            
            # Composite signal calculation
            technical_signals = []
            
            # MACD signal
            if macd.iloc[-1] > signal_line.iloc[-1] and macd_histogram.iloc[-1] > macd_histogram.iloc[-2]:
                technical_signals.append(('macd_bullish', 0.3))
            elif macd.iloc[-1] < signal_line.iloc[-1] and macd_histogram.iloc[-1] < macd_histogram.iloc[-2]:
                technical_signals.append(('macd_bearish', -0.3))
            
            # RSI signal
            current_rsi = rsi.iloc[-1]
            if current_rsi < 30:
                technical_signals.append(('rsi_oversold', 0.2))
            elif current_rsi > 70:
                technical_signals.append(('rsi_overbought', -0.2))
            
            # Momentum signals
            if price_change_5d > self.momentum_threshold and volume_ratio > self.volume_threshold:
                technical_signals.append(('momentum_strong', 0.4))
            elif price_change_5d < -self.momentum_threshold and volume_ratio > self.volume_threshold:
                technical_signals.append(('momentum_weak', -0.4))
            
            # Price breakout signals
            resistance = price_history['High'].rolling(window=20).max().iloc[-1]
            support = price_history['Low'].rolling(window=20).min().iloc[-1]
            
            if current_price > resistance * 1.001:  # Breakout above resistance
                technical_signals.append(('breakout_up', 0.3))
            elif current_price < support * 0.999:  # Breakdown below support
                technical_signals.append(('breakout_down', -0.3))
            
            # Aggregate signals
            technical_score = sum(score for _, score in technical_signals)
            
            # Weighted composite score
            composite_score = (
                technical_score * 0.4 +
                ml_signal * ml_confidence * 0.4 +
                sentiment_score * sentiment_confidence * 0.2
            )
            
            # Determine signal type and strength
            signal_strength = min(abs(composite_score), 1.0)
            
            if composite_score > 0.15:
                if composite_score > 0.3:
                    signal_type = SignalType.STRONG_BUY
                else:
                    signal_type = SignalType.BUY
            elif composite_score < -0.15:
                if composite_score < -0.3:
                    signal_type = SignalType.STRONG_SELL
                else:
                    signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            # Calculate stop loss and target
            atr = self._calculate_atr(price_history)
            
            if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                stop_loss = current_price - (atr * 2)
                target_price = current_price + (atr * 3)
            elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                stop_loss = current_price + (atr * 2)
                target_price = current_price - (atr * 3)
            else:
                stop_loss = None
                target_price = None
            
            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=signal_strength,
                price=current_price,
                timestamp=datetime.now(),
                strategy=self.name,
                target_price=target_price,
                stop_loss=stop_loss,
                metadata={
                    'technical_signals': technical_signals,
                    'technical_score': technical_score,
                    'ml_signal': ml_signal,
                    'ml_confidence': ml_confidence,
                    'sentiment_score': sentiment_score,
                    'composite_score': composite_score,
                    'rsi': current_rsi,
                    'macd': macd.iloc[-1],
                    'volume_ratio': volume_ratio,
                    'atr': atr
                }
            )
            
            self.signals_history.append(signal)
            return signal
            
        except Exception as e:
            logger.error(f"Momentum strategy signal generation failed: {e}")
            return None
    
    def _calculate_atr(self, price_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high_low = price_data['High'] - price_data['Low']
            high_close = abs(price_data['High'] - price_data['Close'].shift())
            low_close = abs(price_data['Low'] - price_data['Close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else price_data['Close'].iloc[-1] * 0.02
            
        except Exception as e:
            logger.error(f"ATR calculation failed: {e}")
            return price_data['Close'].iloc[-1] * 0.02 if 'Close' in price_data.columns else 0.02
    
    def update_parameters(self, performance_feedback: Dict[str, Any]):
        """Update strategy parameters based on performance"""
        success_rate = performance_feedback.get('success_rate', 0.5)
        avg_return = performance_feedback.get('average_return', 0)
        
        self.success_rate = success_rate
        
        # Adaptive parameter adjustment
        if success_rate < 0.4:  # Poor performance
            self.momentum_threshold = min(self.momentum_threshold * 1.1, 0.05)
            self.volume_threshold = min(self.volume_threshold * 1.1, 3.0)
        elif success_rate > 0.6:  # Good performance
            self.momentum_threshold = max(self.momentum_threshold * 0.9, 0.01)
            self.volume_threshold = max(self.volume_threshold * 0.9, 1.2)
        
        self.parameter_history.append({
            'timestamp': datetime.now(),
            'momentum_threshold': self.momentum_threshold,
            'volume_threshold': self.volume_threshold,
            'success_rate': success_rate
        })

class MeanReversionStrategy(BaseStrategy):
    """Advanced mean reversion trading strategy"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("MeanReversionStrategy", config)
        
        self.lookback_period = self.config.get('lookback_period', 20)
        self.std_threshold = self.config.get('std_threshold', 2.0)
        self.rsi_oversold = self.config.get('rsi_oversold', 25)
        self.rsi_overbought = self.config.get('rsi_overbought', 75)
        self.volume_confirmation = self.config.get('volume_confirmation', True)
        
    async def generate_signal(self, data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate mean reversion trading signal"""
        try:
            market_data = data.get('market_data', {})
            symbol = data.get('symbol', 'UNKNOWN')
            current_price = market_data.get('price', 0)
            
            if current_price <= 0:
                return None
            
            price_history = market_data.get('price_history', pd.DataFrame())
            if len(price_history) < self.lookback_period:
                return None
            
            # Calculate mean and standard deviation
            rolling_mean = price_history['Close'].rolling(window=self.lookback_period).mean().iloc[-1]
            rolling_std = price_history['Close'].rolling(window=self.lookback_period).std().iloc[-1]
            
            # Z-score calculation
            z_score = (current_price - rolling_mean) / rolling_std
            
            # RSI calculation
            delta = price_history['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Volume confirmation
            volume_ratio = 1.0
            if self.volume_confirmation and 'Volume' in price_history.columns:
                avg_volume = price_history['Volume'].rolling(window=20).mean().iloc[-1]
                current_volume = market_data.get('volume', avg_volume)
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Signal generation logic
            signal_type = SignalType.HOLD
            signal_strength = 0.0
            
            # Oversold conditions (Buy signal)
            if (z_score < -self.std_threshold and 
                rsi < self.rsi_oversold and 
                volume_ratio > 1.2):
                
                signal_type = SignalType.BUY
                signal_strength = min(abs(z_score) / self.std_threshold, 1.0) * 0.8
            
            # Overbought conditions (Sell signal)
            elif (z_score > self.std_threshold and 
                  rsi > self.rsi_overbought and 
                  volume_ratio > 1.2):
                
                signal_type = SignalType.SELL
                signal_strength = min(abs(z_score) / self.std_threshold, 1.0) * 0.8
            
            # Calculate stop loss and target
            if signal_type == SignalType.BUY:
                stop_loss = current_price - (rolling_std * 1.5)
                target_price = rolling_mean
            elif signal_type == SignalType.SELL:
                stop_loss = current_price + (rolling_std * 1.5)
                target_price = rolling_mean
            else:
                stop_loss = None
                target_price = None
            
            if signal_type != SignalType.HOLD:
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=signal_strength,
                    price=current_price,
                    timestamp=datetime.now(),
                    strategy=self.name,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    metadata={
                        'z_score': z_score,
                        'rsi': rsi,
                        'rolling_mean': rolling_mean,
                        'rolling_std': rolling_std,
                        'volume_ratio': volume_ratio
                    }
                )
                
                self.signals_history.append(signal)
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Mean reversion strategy signal generation failed: {e}")
            return None
    
    def update_parameters(self, performance_feedback: Dict[str, Any]):
        """Update mean reversion parameters"""
        success_rate = performance_feedback.get('success_rate', 0.5)
        
        if success_rate < 0.4:
            self.std_threshold = min(self.std_threshold * 1.1, 3.0)
        elif success_rate > 0.6:
            self.std_threshold = max(self.std_threshold * 0.95, 1.5)

class BreakoutStrategy(BaseStrategy):
    """Advanced breakout trading strategy"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("BreakoutStrategy", config)
        
        self.lookback_period = self.config.get('lookback_period', 20)
        self.breakout_threshold = self.config.get('breakout_threshold', 0.005)
        self.volume_multiplier = self.config.get('volume_multiplier', 1.5)
        self.consolidation_periods = self.config.get('consolidation_periods', 5)
    
    async def generate_signal(self, data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate breakout trading signal"""
        try:
            market_data = data.get('market_data', {})
            symbol = data.get('symbol', 'UNKNOWN')
            current_price = market_data.get('price', 0)
            
            if current_price <= 0:
                return None
            
            price_history = market_data.get('price_history', pd.DataFrame())
            if len(price_history) < self.lookback_period:
                return None
            
            # Calculate support and resistance levels
            resistance = price_history['High'].rolling(window=self.lookback_period).max().iloc[-1]
            support = price_history['Low'].rolling(window=self.lookback_period).min().iloc[-1]
            
            # Check for consolidation
            recent_range = price_history['High'].iloc[-self.consolidation_periods:].max() - price_history['Low'].iloc[-self.consolidation_periods:].min()
            avg_range = (price_history['High'] - price_history['Low']).rolling(window=self.lookback_period).mean().iloc[-1]
            
            is_consolidating = recent_range < avg_range * 0.8
            
            # Volume analysis
            avg_volume = price_history['Volume'].rolling(window=20).mean().iloc[-1]
            current_volume = market_data.get('volume', avg_volume)
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Breakout detection
            signal_type = SignalType.HOLD
            signal_strength = 0.0
            
            # Upward breakout
            if (current_price > resistance * (1 + self.breakout_threshold) and
                volume_ratio >= self.volume_multiplier and
                is_consolidating):
                
                signal_type = SignalType.BUY
                signal_strength = min((current_price - resistance) / resistance / self.breakout_threshold, 1.0) * 0.9
            
            # Downward breakout
            elif (current_price < support * (1 - self.breakout_threshold) and
                  volume_ratio >= self.volume_multiplier and
                  is_consolidating):
                
                signal_type = SignalType.SELL
                signal_strength = min((support - current_price) / support / self.breakout_threshold, 1.0) * 0.9
            
            # Calculate targets and stops
            if signal_type == SignalType.BUY:
                range_size = resistance - support
                target_price = current_price + range_size
                stop_loss = support * 0.995
            elif signal_type == SignalType.SELL:
                range_size = resistance - support
                target_price = current_price - range_size
                stop_loss = resistance * 1.005
            else:
                target_price = None
                stop_loss = None
            
            if signal_type != SignalType.HOLD:
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=signal_strength,
                    price=current_price,
                    timestamp=datetime.now(),
                    strategy=self.name,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    metadata={
                        'resistance': resistance,
                        'support': support,
                        'volume_ratio': volume_ratio,
                        'is_consolidating': is_consolidating,
                        'recent_range': recent_range,
                        'avg_range': avg_range
                    }
                )
                
                self.signals_history.append(signal)
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Breakout strategy signal generation failed: {e}")
            return None
    
    def update_parameters(self, performance_feedback: Dict[str, Any]):
        """Update breakout parameters"""
        success_rate = performance_feedback.get('success_rate', 0.5)
        
        if success_rate < 0.4:
            self.breakout_threshold = min(self.breakout_threshold * 1.2, 0.02)
            self.volume_multiplier = min(self.volume_multiplier * 1.1, 3.0)
        elif success_rate > 0.6:
            self.breakout_threshold = max(self.breakout_threshold * 0.9, 0.002)
            self.volume_multiplier = max(self.volume_multiplier * 0.95, 1.2)

class MLEnhancedStrategy(BaseStrategy):
    """ML-enhanced multi-factor trading strategy"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("MLEnhancedStrategy", config)
        
        self.ml_weight = self.config.get('ml_weight', 0.6)
        self.technical_weight = self.config.get('technical_weight', 0.3)
        self.sentiment_weight = self.config.get('sentiment_weight', 0.1)
        self.min_confidence = self.config.get('min_confidence', 0.3)
        
        # Sub-strategies
        self.momentum_strategy = MomentumStrategy(self.config.get('momentum', {}))
        self.mean_reversion_strategy = MeanReversionStrategy(self.config.get('mean_reversion', {}))
        self.breakout_strategy = BreakoutStrategy(self.config.get('breakout', {}))
    
    async def generate_signal(self, data: Dict[str, Any]) -> Optional[TradingSignal]:
        """Generate ML-enhanced trading signal"""
        try:
            ml_predictions = data.get('ml_predictions', {})
            
            # Get ML prediction
            ml_signal = ml_predictions.get('meta_prediction', 0)
            ml_confidence = ml_predictions.get('meta_confidence', 0)
            
            if ml_confidence < self.min_confidence:
                return None
            
            # Get signals from sub-strategies
            momentum_signal = await self.momentum_strategy.generate_signal(data)
            mean_reversion_signal = await self.mean_reversion_strategy.generate_signal(data)
            breakout_signal = await self.breakout_strategy.generate_signal(data)
            
            # Combine signals
            technical_signals = []
            if momentum_signal:
                weight = 1.0 if momentum_signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else -1.0
                technical_signals.append(weight * momentum_signal.strength)
            
            if mean_reversion_signal:
                weight = 1.0 if mean_reversion_signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else -1.0
                technical_signals.append(weight * mean_reversion_signal.strength)
            
            if breakout_signal:
                weight = 1.0 if breakout_signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else -1.0
                technical_signals.append(weight * breakout_signal.strength)
            
            technical_score = np.mean(technical_signals) if technical_signals else 0
            
            # Sentiment analysis
            sentiment_data = data.get('sentiment_data', {})
            sentiment_score = sentiment_data.get('average_sentiment', 0) if sentiment_data else 0
            
            # Weighted combination
            composite_score = (
                ml_signal * ml_confidence * self.ml_weight +
                technical_score * self.technical_weight +
                sentiment_score * self.sentiment_weight
            )
            
            # Determine final signal
            signal_strength = min(abs(composite_score), 1.0) * ml_confidence
            
            if composite_score > 0.2 and signal_strength > 0.3:
                signal_type = SignalType.STRONG_BUY if composite_score > 0.4 else SignalType.BUY
            elif composite_score < -0.2 and signal_strength > 0.3:
                signal_type = SignalType.STRONG_SELL if composite_score < -0.4 else SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            if signal_type != SignalType.HOLD:
                # Use breakout strategy for stop loss and target if available
                stop_loss = breakout_signal.stop_loss if breakout_signal else None
                target_price = breakout_signal.target_price if breakout_signal else None
                
                # Fallback to momentum strategy
                if not stop_loss and momentum_signal:
                    stop_loss = momentum_signal.stop_loss
                    target_price = momentum_signal.target_price
                
                signal = TradingSignal(
                    symbol=data.get('symbol', 'UNKNOWN'),
                    signal_type=signal_type,
                    strength=signal_strength,
                    price=data.get('market_data', {}).get('price', 0),
                    timestamp=datetime.now(),
                    strategy=self.name,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    metadata={
                        'ml_signal': ml_signal,
                        'ml_confidence': ml_confidence,
                        'technical_score': technical_score,
                        'sentiment_score': sentiment_score,
                        'composite_score': composite_score,
                        'contributing_strategies': {
                            'momentum': momentum_signal.signal_type.value if momentum_signal else None,
                            'mean_reversion': mean_reversion_signal.signal_type.value if mean_reversion_signal else None,
                            'breakout': breakout_signal.signal_type.value if breakout_signal else None
                        }
                    }
                )
                
                self.signals_history.append(signal)
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"ML-enhanced strategy signal generation failed: {e}")
            return None
    
    def update_parameters(self, performance_feedback: Dict[str, Any]):
        """Update ML-enhanced strategy parameters"""
        success_rate = performance_feedback.get('success_rate', 0.5)
        
        # Adjust weights based on performance
        if success_rate < 0.4:
            self.min_confidence = min(self.min_confidence * 1.1, 0.8)
        elif success_rate > 0.6:
            self.min_confidence = max(self.min_confidence * 0.95, 0.2)
        
        # Update sub-strategies
        self.momentum_strategy.update_parameters(performance_feedback)
        self.mean_reversion_strategy.update_parameters(performance_feedback)
        self.breakout_strategy.update_parameters(performance_feedback)

class TradingStrategy:
    """Main trading strategy orchestrator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize strategies
        self.strategies = {
            'momentum': MomentumStrategy(self.config.get('momentum', {})),
            'mean_reversion': MeanReversionStrategy(self.config.get('mean_reversion', {})),
            'breakout': BreakoutStrategy(self.config.get('breakout', {})),
            'ml_enhanced': MLEnhancedStrategy(self.config.get('ml_enhanced', {}))
        }
        
        # Active strategies
        self.active_strategies = self.config.get('active_strategies', ['ml_enhanced', 'momentum'])
        
        # Portfolio and position management
        self.positions = {}
        self.portfolio_value = self.config.get('initial_portfolio_value', 100000)
        self.max_positions = self.config.get('max_positions', 5)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.02)
        
        # Performance tracking
        self.trade_history = []
        self.performance_metrics = {}
        
        # Strategy selection
        self.strategy_weights = self.config.get('strategy_weights', {})
        self.adaptive_weighting = self.config.get('adaptive_weighting', True)
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate trading signals from all active strategies"""
        signals = []
        
        for strategy_name in self.active_strategies:
            if strategy_name in self.strategies:
                try:
                    signal = await self.strategies[strategy_name].generate_signal(market_data)
                    if signal:
                        signals.append(signal)
                        logger.info(f"Generated {signal.signal_type.value} signal from {strategy_name} for {signal.symbol}")
                
                except Exception as e:
                    logger.error(f"Strategy {strategy_name} failed to generate signal: {e}")
        
        return signals
    
    def filter_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter and prioritize signals"""
        if not signals:
            return []
        
        # Group by symbol
        symbol_signals = {}
        for signal in signals:
            if signal.symbol not in symbol_signals:
                symbol_signals[signal.symbol] = []
            symbol_signals[signal.symbol].append(signal)
        
        # Select best signal per symbol
        filtered_signals = []
        for symbol, symbol_signal_list in symbol_signals.items():
            
            # Skip if already have max positions
            if len(self.positions) >= self.max_positions and symbol not in self.positions:
                continue
            
            # Aggregate signals for the symbol
            if len(symbol_signal_list) == 1:
                filtered_signals.append(symbol_signal_list[0])
            else:
                # Combine multiple signals for same symbol
                combined_signal = self._combine_signals(symbol_signal_list)
                if combined_signal:
                    filtered_signals.append(combined_signal)
        
        # Sort by signal strength
        filtered_signals.sort(key=lambda x: x.strength, reverse=True)
        
        return filtered_signals
    
    def _combine_signals(self, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """Combine multiple signals for the same symbol"""
        if not signals:
            return None
        
        # Calculate weighted average
        buy_signals = [s for s in signals if s.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]]
        sell_signals = [s for s in signals if s.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]]
        
        if len(buy_signals) > len(sell_signals):
            # More buy signals
            avg_strength = np.mean([s.strength for s in buy_signals])
            strongest_buy = max(buy_signals, key=lambda x: x.strength)
            
            combined_signal = TradingSignal(
                symbol=strongest_buy.symbol,
                signal_type=SignalType.STRONG_BUY if avg_strength > 0.7 else SignalType.BUY,
                strength=avg_strength,
                price=strongest_buy.price,
                timestamp=datetime.now(),
                strategy="Combined",
                target_price=strongest_buy.target_price,
                stop_loss=strongest_buy.stop_loss,
                metadata={
                    'contributing_signals': len(buy_signals),
                    'strategy_mix': [s.strategy for s in buy_signals]
                }
            )
            return combined_signal
            
        elif len(sell_signals) > len(buy_signals):
            # More sell signals
            avg_strength = np.mean([s.strength for s in sell_signals])
            strongest_sell = max(sell_signals, key=lambda x: x.strength)
            
            combined_signal = TradingSignal(
                symbol=strongest_sell.symbol,
                signal_type=SignalType.STRONG_SELL if avg_strength > 0.7 else SignalType.SELL,
                strength=avg_strength,
                price=strongest_sell.price,
                timestamp=datetime.now(),
                strategy="Combined",
                target_price=strongest_sell.target_price,
                stop_loss=strongest_sell.stop_loss,
                metadata={
                    'contributing_signals': len(sell_signals),
                    'strategy_mix': [s.strategy for s in sell_signals]
                }
            )
            return combined_signal
        
        # Equal or conflicting signals - no clear direction
        return None
    
    def update_positions(self, current_prices: Dict[str, float]):
        """Update existing positions with current prices"""
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.current_price = current_prices[symbol]
                
                # Calculate unrealized P&L
                if position.position_type == PositionType.LONG:
                    position.unrealized_pnl = (position.current_price - position.entry_price) * position.size
                elif position.position_type == PositionType.SHORT:
                    position.unrealized_pnl = (position.entry_price - position.current_price) * position.size
    
    def check_exit_conditions(self) -> List[TradingSignal]:
        """Check if any positions should be closed"""
        exit_signals = []
        
        for symbol, position in self.positions.items():
            exit_signal = None
            
            # Stop loss check
            if position.stop_loss:
                if (position.position_type == PositionType.LONG and position.current_price <= position.stop_loss) or \
                   (position.position_type == PositionType.SHORT and position.current_price >= position.stop_loss):
                    
                    exit_signal = TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL if position.position_type == PositionType.LONG else SignalType.BUY,
                        strength=1.0,
                        price=position.current_price,
                        timestamp=datetime.now(),
                        strategy="StopLoss",
                        metadata={'reason': 'stop_loss', 'position_id': position}
                    )
            
            # Take profit check
            if not exit_signal and position.take_profit:
                if (position.position_type == PositionType.LONG and position.current_price >= position.take_profit) or \
                   (position.position_type == PositionType.SHORT and position.current_price <= position.take_profit):
                    
                    exit_signal = TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL if position.position_type == PositionType.LONG else SignalType.BUY,
                        strength=1.0,
                        price=position.current_price,
                        timestamp=datetime.now(),
                        strategy="TakeProfit",
                        metadata={'reason': 'take_profit', 'position_id': position}
                    )
            
            # Time-based exit (optional)
            if not exit_signal:
                position_age = datetime.now() - position.entry_time
                max_hold_time = self.config.get('max_hold_time_hours', 720)  # 30 days default
                
                if position_age > timedelta(hours=max_hold_time):
                    exit_signal = TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.SELL if position.position_type == PositionType.LONG else SignalType.BUY,
                        strength=0.5,
                        price=position.current_price,
                        timestamp=datetime.now(),
                        strategy="TimeExit",
                        metadata={'reason': 'time_exit', 'position_id': position}
                    )
            
            if exit_signal:
                exit_signals.append(exit_signal)
        
        return exit_signals
    
    def calculate_position_sizes(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Calculate appropriate position sizes for signals"""
        for signal in signals:
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY, SignalType.SELL, SignalType.STRONG_SELL]:
                position_size = self.strategies['momentum'].calculate_position_size(
                    signal, self.portfolio_value, self.risk_per_trade
                )
                signal.position_size = position_size
        
        return signals
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get comprehensive strategy performance metrics"""
        performance = {
            'total_trades': len(self.trade_history),
            'active_positions': len(self.positions),
            'portfolio_value': self.portfolio_value,
            'strategy_performance': {}
        }
        
        # Calculate per-strategy performance
        for strategy_name, strategy in self.strategies.items():
            strategy_trades = [t for t in self.trade_history if t.get('strategy') == strategy_name]
            
            if strategy_trades:
                returns = [t.get('return', 0) for t in strategy_trades]
                success_rate = len([r for r in returns if r > 0]) / len(returns)
                avg_return = np.mean(returns)
                
                performance['strategy_performance'][strategy_name] = {
                    'total_trades': len(strategy_trades),
                    'success_rate': success_rate,
                    'average_return': avg_return,
                    'total_return': sum(returns)
                }
        
        return performance
    
    def update_strategy_parameters(self):
        """Update strategy parameters based on performance"""
        performance = self.get_strategy_performance()
        
        for strategy_name, strategy in self.strategies.items():
            strategy_perf = performance['strategy_performance'].get(strategy_name, {})
            if strategy_perf:
                strategy.update_parameters(strategy_perf)
    
    def save_strategy_state(self, filepath: str) -> bool:
        """Save strategy state and configuration"""
        try:
            strategy_state = {
                'config': self.config,
                'active_strategies': self.active_strategies,
                'strategy_weights': self.strategy_weights,
                'portfolio_value': self.portfolio_value,
                'positions': {k: {
                    'symbol': v.symbol,
                    'position_type': v.position_type.value,
                    'size': v.size,
                    'entry_price': v.entry_price,
                    'current_price': v.current_price,
                    'entry_time': v.entry_time.isoformat(),
                    'stop_loss': v.stop_loss,
                    'take_profit': v.take_profit,
                    'unrealized_pnl': v.unrealized_pnl,
                    'metadata': v.metadata
                } for k, v in self.positions.items()},
                'trade_history': self.trade_history[-1000:],  # Keep last 1000 trades
                'performance_metrics': self.performance_metrics
            }
            
            import joblib
            joblib.dump(strategy_state, filepath)
            logger.info(f"Strategy state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save strategy state: {e}")
            return False
    
    def load_strategy_state(self, filepath: str) -> bool:
        """Load strategy state and configuration"""
        try:
            import joblib
            strategy_state = joblib.load(filepath)
            
            self.config = strategy_state.get('config', {})
            self.active_strategies = strategy_state.get('active_strategies', ['ml_enhanced'])
            self.strategy_weights = strategy_state.get('strategy_weights', {})
            self.portfolio_value = strategy_state.get('portfolio_value', 100000)
            self.trade_history = strategy_state.get('trade_history', [])
            self.performance_metrics = strategy_state.get('performance_metrics', {})
            
            # Reconstruct positions
            positions_data = strategy_state.get('positions', {})
            self.positions = {}
            
            for symbol, pos_data in positions_data.items():
                position = Position(
                    symbol=pos_data['symbol'],
                    position_type=PositionType(pos_data['position_type']),
                    size=pos_data['size'],
                    entry_price=pos_data['entry_price'],
                    current_price=pos_data['current_price'],
                    entry_time=datetime.fromisoformat(pos_data['entry_time']),
                    stop_loss=pos_data.get('stop_loss'),
                    take_profit=pos_data.get('take_profit'),
                    unrealized_pnl=pos_data.get('unrealized_pnl', 0.0),
                    metadata=pos_data.get('metadata', {})
                )
                self.positions[symbol] = position
            
            logger.info(f"Strategy state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load strategy state: {e}")
            return False