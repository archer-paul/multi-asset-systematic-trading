"""
Advanced Portfolio Decision Engine for Trading Bot
Sophisticated multi-factor decision making with adaptive weighting,
regime detection, and portfolio optimization
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import asyncio
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from trading.strategy import TradingSignal, SignalType, PositionType

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime types"""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"

@dataclass
class SignalMetrics:
    """Comprehensive signal evaluation metrics"""
    symbol: str
    technical_score: float
    ml_score: float
    sentiment_score: float
    fundamental_score: float
    momentum_score: float
    volume_score: float
    risk_score: float
    composite_score: float
    confidence: float
    market_regime: MarketRegime
    sector_strength: float
    correlation_penalty: float
    timestamp: datetime

@dataclass
class PortfolioDecision:
    """Portfolio-level investment decision"""
    action: str  # 'buy', 'sell', 'hold', 'rebalance'
    symbol: str
    target_weight: float
    current_weight: float
    conviction: float  # 0-1 confidence level
    reasoning: str
    risk_metrics: Dict[str, float]
    expected_return: float
    expected_risk: float
    hold_period: str
    metadata: Dict[str, Any]

class MarketRegimeDetector:
    """Detects current market regime for adaptive strategy selection"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.lookback_days = self.config.get('regime_lookback_days', 60)
        self.volatility_threshold = self.config.get('volatility_threshold', 0.02)
        self.trend_threshold = self.config.get('trend_threshold', 0.05)

    def detect_regime(self, market_data: Dict[str, Any], macro_data: Dict[str, Any] = None) -> MarketRegime:
        """Detect current market regime"""
        try:
            # Calculate market indicators
            if 'price_history' not in market_data or market_data['price_history'].empty:
                return MarketRegime.SIDEWAYS

            price_data = market_data['price_history']

            # Calculate volatility (20-day rolling)
            returns = price_data['Close'].pct_change()
            volatility = returns.rolling(window=20).std().iloc[-1] * np.sqrt(252)

            # Calculate trend (price vs 50-day MA)
            ma_50 = price_data['Close'].rolling(window=min(50, len(price_data))).mean()
            current_price = price_data['Close'].iloc[-1]
            trend_strength = (current_price - ma_50.iloc[-1]) / ma_50.iloc[-1]

            # Volume analysis
            avg_volume = price_data['Volume'].rolling(window=20).mean().iloc[-1]
            recent_volume = price_data['Volume'].iloc[-5:].mean()
            volume_surge = recent_volume / avg_volume if avg_volume > 0 else 1.0

            # VIX-like indicator (synthetic)
            fear_indicator = volatility * 100

            # Regime classification logic
            if fear_indicator > 30 or volume_surge > 2.0:
                return MarketRegime.CRISIS
            elif volatility > self.volatility_threshold * 2:
                return MarketRegime.HIGH_VOLATILITY
            elif volatility < self.volatility_threshold * 0.5:
                return MarketRegime.LOW_VOLATILITY
            elif trend_strength > self.trend_threshold:
                return MarketRegime.BULL_TREND
            elif trend_strength < -self.trend_threshold:
                return MarketRegime.BEAR_TREND
            else:
                return MarketRegime.SIDEWAYS

        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return MarketRegime.SIDEWAYS

class SignalAggregator:
    """Advanced signal aggregation with adaptive weighting"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Base weights for different signal types
        self.base_weights = self.config.get('base_weights', {
            'technical': 0.25,
            'ml': 0.30,
            'sentiment': 0.15,
            'fundamental': 0.20,
            'momentum': 0.10
        })

        # Regime-specific weight adjustments
        self.regime_adjustments = {
            MarketRegime.BULL_TREND: {'momentum': 1.3, 'sentiment': 1.2, 'technical': 0.9},
            MarketRegime.BEAR_TREND: {'fundamental': 1.4, 'risk': 1.5, 'momentum': 0.8},
            MarketRegime.HIGH_VOLATILITY: {'risk': 1.6, 'technical': 1.2, 'sentiment': 0.8},
            MarketRegime.LOW_VOLATILITY: {'momentum': 1.2, 'ml': 1.1, 'fundamental': 1.1},
            MarketRegime.SIDEWAYS: {'mean_reversion': 1.3, 'technical': 1.1, 'momentum': 0.9},
            MarketRegime.CRISIS: {'fundamental': 1.5, 'risk': 2.0, 'sentiment': 0.6}
        }

        # Performance tracking for adaptive weighting
        self.signal_performance_history = {}

    def aggregate_signals(self, signals_data: Dict[str, Any], market_regime: MarketRegime) -> SignalMetrics:
        """Aggregate multiple signal sources into composite score"""
        try:
            symbol = signals_data.get('symbol', 'UNKNOWN')

            # Extract individual signal components
            technical_score = self._extract_technical_score(signals_data)
            ml_score = self._extract_ml_score(signals_data)
            sentiment_score = self._extract_sentiment_score(signals_data)
            fundamental_score = self._extract_fundamental_score(signals_data)
            momentum_score = self._extract_momentum_score(signals_data)
            volume_score = self._extract_volume_score(signals_data)
            risk_score = self._extract_risk_score(signals_data)

            # Extract new macro-economic and geopolitical scores
            macro_score = self._extract_macro_economic_score(signals_data)
            geopolitical_score = self._extract_geopolitical_score(signals_data)
            commodities_score = self._extract_commodities_score(signals_data)
            forex_score = self._extract_forex_score(signals_data)

            # Get adaptive weights for current regime
            weights = self._get_adaptive_weights(market_regime, symbol)

            # Calculate weighted composite score with boost for positive signals
            composite_score = (
                technical_score * weights.get('technical', 0.20) +
                ml_score * weights.get('ml', 0.25) +
                sentiment_score * weights.get('sentiment', 0.12) +
                fundamental_score * weights.get('fundamental', 0.15) +
                momentum_score * weights.get('momentum', 0.08) +
                macro_score * weights.get('macro', 0.10) +
                geopolitical_score * weights.get('geopolitical', 0.05) +
                commodities_score * weights.get('commodities', 0.03) +
                forex_score * weights.get('forex', 0.02)
            )

            # Apply signal amplification to generate more actionable signals
            if abs(composite_score) > 0.02:  # Small boost for any meaningful signal
                composite_score *= 1.5  # Amplify by 50%

            # Calculate confidence based on signal agreement (more generous)
            signal_values = [technical_score, ml_score, sentiment_score, fundamental_score, momentum_score, macro_score, geopolitical_score, commodities_score, forex_score]
            signal_agreement = self._calculate_signal_agreement(signal_values)

            # Boost confidence calculation to generate more actionable signals
            base_confidence = max(0.3, signal_agreement * 1.5)  # Minimum 30% confidence
            signal_strength_bonus = min(0.3, abs(composite_score) * 2)  # Bonus for strong signals
            confidence = min(base_confidence + signal_strength_bonus, 1.0)

            # Apply risk and correlation adjustments
            correlation_penalty = self._calculate_correlation_penalty(signals_data)
            final_score = composite_score * (1 - correlation_penalty)

            # Get sector strength
            sector_strength = signals_data.get('sector_analysis', {}).get('strength', 0.0)

            return SignalMetrics(
                symbol=symbol,
                technical_score=technical_score,
                ml_score=ml_score,
                sentiment_score=sentiment_score,
                fundamental_score=fundamental_score,
                momentum_score=momentum_score,
                volume_score=volume_score,
                risk_score=risk_score,
                composite_score=final_score,
                confidence=confidence,
                market_regime=market_regime,
                sector_strength=sector_strength,
                correlation_penalty=correlation_penalty,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Signal aggregation failed for {symbol}: {e}")
            return self._create_default_metrics(signals_data.get('symbol', 'UNKNOWN'))

    def _extract_technical_score(self, data: Dict[str, Any]) -> float:
        """Extract and normalize technical analysis score"""
        try:
            technical_signals = data.get('technical_analysis', {})

            # RSI score (normalized)
            rsi = technical_signals.get('rsi', 50)
            rsi_score = (50 - abs(rsi - 50)) / 50  # Higher when closer to 50

            # MACD score
            macd_signal = technical_signals.get('macd_signal', 0)  # -1 to 1

            # Volume confirmation
            volume_ratio = technical_signals.get('volume_ratio', 1.0)
            volume_score = min(volume_ratio / 2.0, 1.0)  # Normalize to 0-1

            # Support/Resistance proximity
            price = data.get('current_price', 0)
            resistance = technical_signals.get('resistance', price * 1.05)
            support = technical_signals.get('support', price * 0.95)

            if price > 0:
                breakout_potential = min((resistance - price) / (resistance - support), 1.0) if resistance != support else 0.5
            else:
                breakout_potential = 0.5

            # Weighted technical score
            technical_score = (
                rsi_score * 0.3 +
                (macd_signal + 1) * 0.5 * 0.25 +  # Normalize MACD to 0-1
                volume_score * 0.2 +
                breakout_potential * 0.25
            )

            return np.clip(technical_score, -1.0, 1.0)

        except Exception as e:
            logger.error(f"Technical score extraction failed: {e}")
            return 0.0

    def _extract_ml_score(self, data: Dict[str, Any]) -> float:
        """Extract ML prediction score"""
        try:
            ml_data = data.get('ml_predictions', {})

            # Meta prediction (ensemble of models)
            meta_prediction = ml_data.get('meta_prediction', 0)
            meta_confidence = ml_data.get('meta_confidence', 0.5)

            # Individual model predictions
            traditional_pred = ml_data.get('traditional_prediction', 0)
            transformer_pred = ml_data.get('transformer_prediction', 0)
            traditional_conf = ml_data.get('traditional_confidence', 0.5)
            transformer_conf = ml_data.get('transformer_confidence', 0.5)

            # Weighted ensemble
            if meta_confidence > 0.3:  # Use meta if confident enough
                ml_score = meta_prediction * meta_confidence
            else:
                # Fallback to individual models
                ml_score = (
                    traditional_pred * traditional_conf * 0.6 +
                    transformer_pred * transformer_conf * 0.4
                )

            return np.clip(ml_score, -1.0, 1.0)

        except Exception as e:
            logger.error(f"ML score extraction failed: {e}")
            return 0.0

    def _extract_sentiment_score(self, data: Dict[str, Any]) -> float:
        """Extract sentiment analysis score"""
        try:
            sentiment_data = data.get('sentiment_analysis', {})

            # News sentiment
            news_sentiment = sentiment_data.get('news_sentiment', 0)
            news_confidence = sentiment_data.get('news_confidence', 0.5)

            # Social media sentiment
            social_sentiment = sentiment_data.get('social_sentiment', 0)
            social_confidence = sentiment_data.get('social_confidence', 0.3)

            # Congress trading sentiment
            congress_sentiment = sentiment_data.get('congress_sentiment', 0)
            congress_activity = sentiment_data.get('congress_activity', 0)

            # Weighted sentiment score
            sentiment_score = (
                news_sentiment * news_confidence * 0.5 +
                social_sentiment * social_confidence * 0.3 +
                congress_sentiment * min(congress_activity / 1000000, 1.0) * 0.2  # Normalize congress activity
            )

            return np.clip(sentiment_score, -1.0, 1.0)

        except Exception as e:
            logger.error(f"Sentiment score extraction failed: {e}")
            return 0.0

    def _extract_fundamental_score(self, data: Dict[str, Any]) -> float:
        """Extract fundamental analysis score"""
        try:
            fundamental_data = data.get('fundamental_analysis', {})

            # Valuation metrics
            pe_ratio = fundamental_data.get('pe_ratio', 20)
            pb_ratio = fundamental_data.get('pb_ratio', 2)
            peg_ratio = fundamental_data.get('peg_ratio', 1)

            # Normalize valuation scores (lower is better for PE, PB; PEG around 1 is good)
            pe_score = max(0, (30 - pe_ratio) / 30) if pe_ratio > 0 else 0.5
            pb_score = max(0, (5 - pb_ratio) / 5) if pb_ratio > 0 else 0.5
            peg_score = max(0, 1 - abs(peg_ratio - 1)) if peg_ratio > 0 else 0.5

            # Growth metrics
            revenue_growth = fundamental_data.get('revenue_growth', 0)
            earnings_growth = fundamental_data.get('earnings_growth', 0)

            # Normalize growth scores
            revenue_score = min(revenue_growth / 0.2, 1.0) if revenue_growth > 0 else max(revenue_growth / -0.2, -1.0)
            earnings_score = min(earnings_growth / 0.3, 1.0) if earnings_growth > 0 else max(earnings_growth / -0.3, -1.0)

            # Financial health
            debt_to_equity = fundamental_data.get('debt_to_equity', 0.5)
            roe = fundamental_data.get('roe', 0.1)

            financial_health = (
                max(0, (2 - debt_to_equity) / 2) * 0.3 +  # Lower debt is better
                min(roe / 0.2, 1.0) * 0.7  # Higher ROE is better
            )

            # Weighted fundamental score
            fundamental_score = (
                pe_score * 0.15 +
                pb_score * 0.1 +
                peg_score * 0.15 +
                revenue_score * 0.25 +
                earnings_score * 0.25 +
                financial_health * 0.1
            )

            return np.clip(fundamental_score * 2 - 1, -1.0, 1.0)  # Convert to -1 to 1 range

        except Exception as e:
            logger.error(f"Fundamental score extraction failed: {e}")
            return 0.0

    def _extract_momentum_score(self, data: Dict[str, Any]) -> float:
        """Extract momentum score"""
        try:
            momentum_data = data.get('momentum_analysis', {})

            # Price momentum (various timeframes)
            momentum_1d = momentum_data.get('momentum_1d', 0)
            momentum_5d = momentum_data.get('momentum_5d', 0)
            momentum_20d = momentum_data.get('momentum_20d', 0)

            # Volume-weighted momentum
            volume_momentum = momentum_data.get('volume_momentum', 0)

            # Relative strength
            relative_strength = momentum_data.get('relative_strength', 0)

            # Weighted momentum score
            momentum_score = (
                momentum_1d * 0.1 +
                momentum_5d * 0.3 +
                momentum_20d * 0.3 +
                volume_momentum * 0.2 +
                relative_strength * 0.1
            )

            return np.clip(momentum_score, -1.0, 1.0)

        except Exception as e:
            logger.error(f"Momentum score extraction failed: {e}")
            return 0.0

    def _extract_volume_score(self, data: Dict[str, Any]) -> float:
        """Extract volume analysis score"""
        try:
            volume_data = data.get('volume_analysis', {})

            volume_ratio = volume_data.get('volume_ratio', 1.0)
            volume_trend = volume_data.get('volume_trend', 0)

            # Volume score (higher volume during price moves is positive)
            volume_score = (
                min((volume_ratio - 1) / 2, 1.0) * 0.7 +  # Volume surge
                volume_trend * 0.3  # Volume trend
            )

            return np.clip(volume_score, -1.0, 1.0)

        except Exception as e:
            logger.error(f"Volume score extraction failed: {e}")
            return 0.0

    def _extract_risk_score(self, data: Dict[str, Any]) -> float:
        """Extract risk assessment score (lower risk = higher score)"""
        try:
            risk_data = data.get('risk_analysis', {})

            # Volatility (lower is better)
            volatility = risk_data.get('volatility', 0.02)
            volatility_score = max(0, (0.05 - volatility) / 0.05)

            # Beta (closer to 1 is better for most strategies)
            beta = risk_data.get('beta', 1.0)
            beta_score = max(0, 1 - abs(beta - 1))

            # Drawdown risk
            max_drawdown = risk_data.get('max_drawdown', 0.1)
            drawdown_score = max(0, (0.2 - max_drawdown) / 0.2)

            # Liquidity
            liquidity = risk_data.get('liquidity_score', 0.8)

            # Combined risk score (higher = lower risk)
            risk_score = (
                volatility_score * 0.3 +
                beta_score * 0.2 +
                drawdown_score * 0.3 +
                liquidity * 0.2
            )

            return risk_score

        except Exception as e:
            logger.error(f"Risk score extraction failed: {e}")
            return 0.5

    def _get_adaptive_weights(self, market_regime: MarketRegime, symbol: str) -> Dict[str, float]:
        """Get adaptive weights based on market regime and ML ensemble performance"""
        weights = self.base_weights.copy()

        # Apply regime adjustments
        regime_adj = self.regime_adjustments.get(market_regime, {})
        for factor, multiplier in regime_adj.items():
            if factor in weights:
                weights[factor] *= multiplier

        # Integrate with existing ML ensemble performance tracking
        # This will be connected to the EnsemblePredictor's BayesianModelAveraging
        try:
            # Get ML model performance from the existing system
            # This integrates with the existing ml/ensemble.py performance tracking
            ml_performance = getattr(self, 'ml_ensemble_performance', {})

            if ml_performance:
                # Traditional ML models performance
                traditional_performance = ml_performance.get('traditional', {}).get('recent_accuracy', 0.5)
                transformer_performance = ml_performance.get('transformer', {}).get('recent_accuracy', 0.5)

                # Adjust ML weight based on actual performance
                ensemble_performance = (traditional_performance + transformer_performance) / 2
                if ensemble_performance > 0.6:
                    weights['ml'] *= 1.2
                elif ensemble_performance < 0.4:
                    weights['ml'] *= 0.8

            # Apply sentiment performance tracking (if available from historical data)
            if hasattr(self, 'sentiment_performance_tracker'):
                sentiment_accuracy = getattr(self.sentiment_performance_tracker, 'recent_accuracy', 0.5)
                if sentiment_accuracy > 0.6:
                    weights['sentiment'] *= 1.15
                elif sentiment_accuracy < 0.4:
                    weights['sentiment'] *= 0.85

        except Exception as e:
            logger.debug(f"Could not apply ML performance adjustments: {e}")

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def _calculate_signal_agreement(self, signal_values: List[float]) -> float:
        """Calculate agreement between different signals"""
        if len(signal_values) < 2:
            return 0.5

        # Remove zero values (missing signals)
        active_signals = [s for s in signal_values if abs(s) > 0.01]

        if len(active_signals) < 2:
            return 0.5

        # Calculate pairwise correlation
        correlations = []
        for i in range(len(active_signals)):
            for j in range(i + 1, len(active_signals)):
                # Simple agreement measure
                if active_signals[i] * active_signals[j] > 0:  # Same direction
                    correlations.append(min(abs(active_signals[i]), abs(active_signals[j])))
                else:  # Opposite direction
                    correlations.append(-min(abs(active_signals[i]), abs(active_signals[j])))

        if not correlations:
            return 0.5

        agreement = np.mean(correlations)
        return max(0, (agreement + 1) / 2)  # Normalize to 0-1 range

    def _calculate_correlation_penalty(self, data: Dict[str, Any]) -> float:
        """Calculate penalty for high correlation with existing positions"""
        try:
            correlations = data.get('correlation_analysis', {})
            portfolio_correlation = correlations.get('portfolio_correlation', 0.0)
            sector_concentration = correlations.get('sector_concentration', 0.0)

            # Penalty increases with correlation and concentration
            penalty = (
                min(abs(portfolio_correlation) * 0.3, 0.2) +
                min(sector_concentration * 0.2, 0.1)
            )

            return penalty

        except Exception as e:
            logger.error(f"Correlation penalty calculation failed: {e}")
            return 0.0

    def _extract_macro_economic_score(self, data: Dict[str, Any]) -> float:
        """Extract macro-economic analysis score"""
        try:
            macro_analysis = data.get('macro_economic_analysis', {})

            # Economic strength and macro score
            economic_strength = macro_analysis.get('economic_strength', 0.0)
            macro_score = macro_analysis.get('macro_score', 0.5)

            # Inflation impact (negative impact reduces score)
            inflation_impact = macro_analysis.get('inflation_impact', 0.0)
            gdp_growth = macro_analysis.get('gdp_growth', 0.0)

            # Combined macro score (-1 to 1)
            combined_score = (
                economic_strength * 0.4 +
                (macro_score - 0.5) * 2 * 0.3 +  # Convert 0-1 to -1 to 1
                gdp_growth * 0.2 +
                -inflation_impact * 0.1  # Negative inflation impact
            )

            return np.clip(combined_score, -1.0, 1.0)

        except Exception as e:
            logger.error(f"Macro-economic score extraction failed: {e}")
            return 0.0

    def _extract_geopolitical_score(self, data: Dict[str, Any]) -> float:
        """Extract geopolitical risk score"""
        try:
            geopolitical_analysis = data.get('geopolitical_risk_analysis', {})

            # Overall risk (higher risk = lower score)
            overall_risk = geopolitical_analysis.get('overall_risk', 0.0)
            sector_risk_multiplier = geopolitical_analysis.get('sector_risk_multiplier', 1.0)
            risk_count = geopolitical_analysis.get('risk_count', 0)

            # Convert risks to negative scores
            risk_score = -overall_risk * sector_risk_multiplier

            # Risk count penalty
            risk_count_penalty = min(risk_count * 0.1, 0.5)

            # Combined geopolitical score
            geo_score = risk_score - risk_count_penalty

            return np.clip(geo_score, -1.0, 1.0)

        except Exception as e:
            logger.error(f"Geopolitical score extraction failed: {e}")
            return 0.0

    def _extract_commodities_score(self, data: Dict[str, Any]) -> float:
        """Extract commodities market score"""
        try:
            commodities_analysis = data.get('commodities_analysis', {})

            # Gold correlation and inflation hedge
            gold_correlation = commodities_analysis.get('gold_correlation', 0.0)
            inflation_hedge_score = commodities_analysis.get('inflation_hedge_score', 0.0)
            commodities_momentum = commodities_analysis.get('commodities_momentum', 0.0)
            volatility_score = commodities_analysis.get('volatility_score', 0.5)

            # Combined commodities score
            commodities_score = (
                gold_correlation * 0.3 +
                inflation_hedge_score * 0.3 +
                commodities_momentum * 0.3 +
                (volatility_score - 0.5) * 2 * 0.1  # Convert volatility to -1 to 1
            )

            return np.clip(commodities_score, -1.0, 1.0)

        except Exception as e:
            logger.error(f"Commodities score extraction failed: {e}")
            return 0.0

    def _extract_forex_score(self, data: Dict[str, Any]) -> float:
        """Extract forex market score"""
        try:
            forex_analysis = data.get('forex_analysis', {})

            # Currency strength and volatility
            currency_strength = forex_analysis.get('currency_strength', 0.0)
            currency_volatility = forex_analysis.get('currency_volatility', 0.0)
            carry_trade_signal = forex_analysis.get('carry_trade_signal', 0.0)

            # Combined forex score
            forex_score = (
                currency_strength * 0.5 +
                carry_trade_signal * 0.3 +
                -min(currency_volatility, 1.0) * 0.2  # High volatility is negative
            )

            return np.clip(forex_score, -1.0, 1.0)

        except Exception as e:
            logger.error(f"Forex score extraction failed: {e}")
            return 0.0

    def _create_default_metrics(self, symbol: str) -> SignalMetrics:
        """Create default metrics when aggregation fails"""
        return SignalMetrics(
            symbol=symbol,
            technical_score=0.0,
            ml_score=0.0,
            sentiment_score=0.0,
            fundamental_score=0.0,
            momentum_score=0.0,
            volume_score=0.0,
            risk_score=0.5,
            composite_score=0.0,
            confidence=0.0,
            market_regime=MarketRegime.SIDEWAYS,
            sector_strength=0.0,
            correlation_penalty=0.0,
            timestamp=datetime.now()
        )

class PortfolioOptimizer:
    """Modern portfolio optimization with Black-Litterman and risk budgeting"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.target_return = self.config.get('target_return', 0.12)  # 12% annual
        self.max_weight = self.config.get('max_weight', 0.15)  # Max 15% per position
        self.min_weight = self.config.get('min_weight', 0.02)  # Min 2% per position
        self.risk_free_rate = self.config.get('risk_free_rate', 0.03)  # 3% risk-free rate

    def optimize_portfolio(self, signal_metrics: List[SignalMetrics],
                         current_portfolio: Dict[str, float],
                         market_data: Dict[str, Any]) -> Dict[str, PortfolioDecision]:
        """Optimize portfolio allocation based on signal metrics"""
        try:
            if not signal_metrics:
                return {}

            # Debug: Log signal quality distribution
            logger.info(f"Signal metrics distribution:")
            high_conf = [s for s in signal_metrics if s.confidence > 0.5]
            med_conf = [s for s in signal_metrics if 0.3 <= s.confidence <= 0.5]
            low_conf = [s for s in signal_metrics if s.confidence < 0.3]
            high_score = [s for s in signal_metrics if abs(s.composite_score) > 0.3]
            med_score = [s for s in signal_metrics if 0.1 <= abs(s.composite_score) <= 0.3]

            logger.info(f"  Confidence: High(>0.5)={len(high_conf)}, Med(0.3-0.5)={len(med_conf)}, Low(<0.3)={len(low_conf)}")
            logger.info(f"  Score: High(>0.3)={len(high_score)}, Med(0.1-0.3)={len(med_score)}")

            # Show top 5 signals
            top_signals = sorted(signal_metrics, key=lambda x: x.confidence * abs(x.composite_score), reverse=True)[:5]
            for i, s in enumerate(top_signals):
                logger.info(f"  Top {i+1}: {s.symbol} - conf:{s.confidence:.2f}, score:{s.composite_score:.2f}")

            # Filter strong signals (very reduced thresholds to generate more trades)
            strong_signals = [s for s in signal_metrics if s.confidence > 0.15 and abs(s.composite_score) > 0.05]

            if not strong_signals:
                logger.warning("No signals passed reduced filtering criteria")
                return {}

            # Build expected returns vector
            expected_returns = self._build_expected_returns(strong_signals)

            # Build covariance matrix
            cov_matrix = self._build_covariance_matrix(strong_signals, market_data)

            # Apply Black-Litterman adjustments
            bl_returns, bl_cov = self._apply_black_litterman(expected_returns, cov_matrix, strong_signals)

            # Optimize weights
            optimal_weights = self._optimize_weights(bl_returns, bl_cov)

            # Generate portfolio decisions
            decisions = self._generate_decisions(optimal_weights, strong_signals, current_portfolio)

            return decisions

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return {}

    def _build_expected_returns(self, signals: List[SignalMetrics]) -> np.ndarray:
        """Build expected returns vector from signal metrics"""
        returns = []
        for signal in signals:
            # Convert composite score to expected annual return
            base_return = signal.composite_score * 0.2  # Max 20% from signals
            confidence_adj = base_return * signal.confidence
            returns.append(confidence_adj)

        return np.array(returns)

    def _build_covariance_matrix(self, signals: List[SignalMetrics],
                               market_data: Dict[str, Any]) -> np.ndarray:
        """Build covariance matrix from historical data or use simple model"""
        n_assets = len(signals)

        # Simple diagonal covariance matrix (can be enhanced with historical data)
        cov_matrix = np.eye(n_assets) * 0.04  # Base 4% volatility

        # Add sector correlations
        sectors = {}
        for i, signal in enumerate(signals):
            sector = market_data.get(signal.symbol, {}).get('sector', 'Unknown')
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(i)

        # Add sector correlation (0.3 within sector)
        for sector_indices in sectors.values():
            for i in sector_indices:
                for j in sector_indices:
                    if i != j:
                        cov_matrix[i, j] = 0.012  # 30% correlation * sqrt(0.04 * 0.04)

        return cov_matrix

    def _apply_black_litterman(self, expected_returns: np.ndarray,
                             cov_matrix: np.ndarray,
                             signals: List[SignalMetrics]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Black-Litterman model for better return estimates"""
        try:
            n_assets = len(expected_returns)

            # Market implied returns (using reverse optimization)
            market_weights = np.ones(n_assets) / n_assets  # Equal weight as market proxy
            lambda_risk = 3.0  # Risk aversion parameter

            implied_returns = lambda_risk * np.dot(cov_matrix, market_weights)

            # Views and confidence
            P = np.eye(n_assets)  # Each asset is a view
            Q = expected_returns  # Our view on expected returns

            # Confidence in views (based on signal confidence)
            tau = 0.025  # Scales the uncertainty of the prior
            confidence_weights = np.array([s.confidence for s in signals])
            omega = np.diag(1 / confidence_weights) * tau

            # Black-Litterman calculation
            precision_prior = np.linalg.inv(tau * cov_matrix)
            precision_views = np.dot(P.T, np.dot(np.linalg.inv(omega), P))

            bl_precision = precision_prior + precision_views
            bl_cov = np.linalg.inv(bl_precision)

            bl_returns = np.dot(bl_cov,
                               np.dot(precision_prior, implied_returns) +
                               np.dot(P.T, np.dot(np.linalg.inv(omega), Q)))

            return bl_returns, bl_cov

        except Exception as e:
            logger.error(f"Black-Litterman application failed: {e}")
            return expected_returns, cov_matrix

    def _optimize_weights(self, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Optimize portfolio weights using robust mean-variance optimization"""
        try:
            n_assets = len(expected_returns)

            # Input validation
            if n_assets == 0:
                logger.warning("No assets to optimize")
                return np.array([])

            if n_assets == 1:
                logger.info("Single asset portfolio - using full allocation")
                return np.array([1.0])

            # Validate expected returns
            if np.any(np.isnan(expected_returns)) or np.any(np.isinf(expected_returns)):
                logger.warning("Invalid expected returns, using equal weights")
                return np.ones(n_assets) / n_assets

            # Validate and regularize covariance matrix
            cov_matrix = self._regularize_covariance_matrix(cov_matrix)

            # Try multiple optimization methods in order of preference
            methods = ['risk_parity', 'mean_variance', 'equal_weight']

            for method in methods:
                try:
                    if method == 'risk_parity':
                        weights = self._risk_parity_optimization(cov_matrix)
                    elif method == 'mean_variance':
                        weights = self._mean_variance_optimization(expected_returns, cov_matrix)
                    else:  # equal_weight
                        weights = np.ones(n_assets) / n_assets

                    # Validate weights
                    if self._validate_weights(weights):
                        logger.info(f"Portfolio optimization successful using {method} method")
                        return weights
                    else:
                        logger.warning(f"{method} optimization produced invalid weights")

                except Exception as method_error:
                    logger.warning(f"{method} optimization failed: {method_error}")
                    continue

            # Fallback to equal weights
            logger.warning("All optimization methods failed, using equal weights")
            return np.ones(n_assets) / n_assets

        except Exception as e:
            logger.error(f"Portfolio optimization completely failed: {e}")
            return np.ones(len(expected_returns)) / len(expected_returns)

    def _generate_decisions(self, optimal_weights: np.ndarray,
                          signals: List[SignalMetrics],
                          current_portfolio: Dict[str, float]) -> Dict[str, PortfolioDecision]:
        """Generate portfolio decisions from optimal weights"""
        decisions = {}

        for i, (weight, signal) in enumerate(zip(optimal_weights, signals)):
            current_weight = current_portfolio.get(signal.symbol, 0.0)
            weight_diff = weight - current_weight

            # Determine action (more sensitive thresholds)
            if abs(weight_diff) < 0.005:  # Less than 0.5% difference
                action = 'hold'
            elif weight_diff > 0.01:  # Increase by more than 1%
                action = 'buy'
            elif weight_diff < -0.01:  # Decrease by more than 1%
                action = 'sell'
            else:
                action = 'hold'

            # Calculate expected risk and return
            expected_return = signal.composite_score * signal.confidence * 0.15
            expected_risk = np.sqrt(0.04 * (1 + abs(signal.composite_score)))  # Risk increases with conviction

            # Generate reasoning
            reasoning = self._generate_reasoning(signal, weight, current_weight, action)

            decisions[signal.symbol] = PortfolioDecision(
                action=action,
                symbol=signal.symbol,
                target_weight=weight,
                current_weight=current_weight,
                conviction=signal.confidence,
                reasoning=reasoning,
                risk_metrics={
                    'expected_volatility': expected_risk,
                    'correlation_penalty': signal.correlation_penalty,
                    'risk_score': signal.risk_score
                },
                expected_return=expected_return,
                expected_risk=expected_risk,
                hold_period='medium',  # Can be enhanced with more logic
                metadata={
                    'signal_metrics': signal,
                    'weight_change': weight_diff,
                    'market_regime': signal.market_regime.value
                }
            )

        return decisions

    def _generate_reasoning(self, signal: SignalMetrics, target_weight: float,
                          current_weight: float, action: str) -> str:
        """Generate human-readable reasoning for the decision"""

        reasoning_parts = []

        # Signal strength
        if signal.composite_score > 0.3:
            reasoning_parts.append("Strong bullish signals")
        elif signal.composite_score < -0.3:
            reasoning_parts.append("Strong bearish signals")
        else:
            reasoning_parts.append("Mixed signals")

        # Best performing factors
        factors = {
            'Technical': signal.technical_score,
            'ML': signal.ml_score,
            'Sentiment': signal.sentiment_score,
            'Fundamental': signal.fundamental_score,
            'Momentum': signal.momentum_score
        }

        best_factor = max(factors.items(), key=lambda x: abs(x[1]))
        reasoning_parts.append(f"driven by {best_factor[0].lower()} analysis")

        # Market regime
        reasoning_parts.append(f"in {signal.market_regime.value.replace('_', ' ')} market")

        # Confidence
        if signal.confidence > 0.7:
            reasoning_parts.append("with high confidence")
        elif signal.confidence > 0.5:
            reasoning_parts.append("with moderate confidence")
        else:
            reasoning_parts.append("with low confidence")

        # Weight change explanation
        weight_change = target_weight - current_weight
        if abs(weight_change) > 0.05:
            direction = "increase" if weight_change > 0 else "reduce"
            reasoning_parts.append(f"suggesting to {direction} position significantly")

        return "; ".join(reasoning_parts).capitalize() + "."

    def _regularize_covariance_matrix(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Regularize covariance matrix to ensure numerical stability"""
        try:
            # Check for NaN/Inf values
            if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
                logger.warning("Invalid covariance matrix detected, using identity matrix")
                return np.eye(cov_matrix.shape[0]) * 0.04  # 4% volatility assumption

            # Ensure symmetry
            cov_matrix = (cov_matrix + cov_matrix.T) / 2

            # Check positive definiteness
            eigenvals = np.linalg.eigvals(cov_matrix)
            if np.any(eigenvals <= 0):
                logger.debug("Covariance matrix not positive definite, applying regularization")
                # Add small value to diagonal for regularization
                regularization = max(0.001, -np.min(eigenvals) + 0.001)
                cov_matrix += regularization * np.eye(cov_matrix.shape[0])

            # Check condition number
            cond_num = np.linalg.cond(cov_matrix)
            if cond_num > 1e12:
                logger.debug(f"High condition number ({cond_num:.2e}), applying stronger regularization")
                cov_matrix += 0.01 * np.eye(cov_matrix.shape[0])

            return cov_matrix

        except Exception as e:
            logger.error(f"Covariance regularization failed: {e}")
            n = cov_matrix.shape[0] if len(cov_matrix.shape) > 0 else 1
            return np.eye(n) * 0.04

    def _risk_parity_optimization(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Risk parity optimization - equal risk contribution"""
        n_assets = cov_matrix.shape[0]

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        def risk_parity_objective(weights):
            # Portfolio volatility
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

            # Marginal risk contributions
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol

            # Risk contributions
            risk_contrib = weights * marginal_contrib

            # Target equal risk contribution
            target_contrib = portfolio_vol / n_assets

            # Minimize deviation from equal risk contribution
            return np.sum((risk_contrib - target_contrib) ** 2)

        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0.001, 0.5) for _ in range(n_assets)]  # Relaxed bounds

        result = minimize(risk_parity_objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 500, 'ftol': 1e-9})

        if result.success:
            return result.x
        else:
            raise ValueError(f"Risk parity optimization failed: {result.message}")

    def _mean_variance_optimization(self, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Traditional mean-variance optimization with relaxed constraints"""
        n_assets = len(expected_returns)

        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            # Reduced risk aversion for more aggressive allocation
            return -(portfolio_return - 1.0 * portfolio_variance)

        # Relaxed constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]

        # More flexible bounds
        min_weight = max(0.001, 1.0 / (n_assets * 2))  # Dynamic minimum
        max_weight = min(0.3, 3.0 / n_assets)  # Dynamic maximum
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]

        # Better initial guess based on expected returns
        if np.std(expected_returns) > 0:
            # Weight by relative expected returns
            positive_returns = np.maximum(expected_returns, 0)
            if np.sum(positive_returns) > 0:
                x0 = positive_returns / np.sum(positive_returns)
            else:
                x0 = np.ones(n_assets) / n_assets
        else:
            x0 = np.ones(n_assets) / n_assets

        # Multiple optimization attempts with different methods
        methods = ['SLSQP', 'trust-constr']

        for method in methods:
            try:
                result = minimize(objective, x0, method=method, bounds=bounds,
                                constraints=constraints, options={'maxiter': 1000})

                if result.success and self._validate_weights(result.x):
                    return result.x

            except Exception as e:
                logger.debug(f"Method {method} failed: {e}")
                continue

        raise ValueError("All mean-variance optimization methods failed")

    def _validate_weights(self, weights: np.ndarray) -> bool:
        """Validate portfolio weights"""
        try:
            # Check for NaN/Inf
            if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                return False

            # Check non-negative
            if np.any(weights < -1e-6):  # Small tolerance for numerical errors
                return False

            # Check sum approximately equals 1
            weight_sum = np.sum(weights)
            if abs(weight_sum - 1.0) > 1e-3:
                return False

            # Check reasonable bounds
            if np.any(weights > 0.8):  # No single position > 80%
                return False

            return True

        except Exception:
            return False

class AdvancedPortfolioDecisionEngine:
    """Main class orchestrating advanced portfolio decisions"""

    def __init__(self, config: Dict[str, Any] = None, db_manager=None, ensemble_predictor=None):
        self.config = config or {}

        # Integrate with existing infrastructure
        self.db_manager = db_manager  # Reuse existing database manager
        self.ensemble_predictor = ensemble_predictor  # Reuse existing ML ensemble

        # Initialize components
        self.regime_detector = MarketRegimeDetector(self.config.get('regime_detection', {}))
        self.signal_aggregator = SignalAggregator(self.config.get('signal_aggregation', {}))
        self.portfolio_optimizer = PortfolioOptimizer(self.config.get('portfolio_optimization', {}))

        # Decision thresholds (more aggressive to generate trades)
        self.min_conviction = self.config.get('min_conviction', 0.15)
        self.rebalance_threshold = self.config.get('rebalance_threshold', 0.02)

        # State tracking
        self.last_decisions = {}
        self.decision_history = []

        # Performance tracking integration
        self.performance_feedback_enabled = True

        # Auto-retraining scheduler
        self.last_full_retrain = None
        self.retrain_interval_days = config.get('retrain_interval_days', 30)  # Monthly by default
        self.retrain_threshold_accuracy = config.get('retrain_threshold_accuracy', 0.65)  # Trigger if below 65%

    async def make_portfolio_decisions(self,
                                     all_signals_data: Dict[str, Any],
                                     current_portfolio: Dict[str, float],
                                     market_data: Dict[str, Any]) -> Dict[str, PortfolioDecision]:
        """Main method to make comprehensive portfolio decisions"""
        try:
            logger.info("Starting advanced portfolio decision process...")

            # Step 1: Detect market regime
            market_regime = self.regime_detector.detect_regime(market_data)
            logger.info(f"Detected market regime: {market_regime.value}")

            # Step 2: Aggregate signals for each symbol
            signal_metrics = []

            for symbol, symbol_data in all_signals_data.items():
                try:
                    metrics = self.signal_aggregator.aggregate_signals(symbol_data, market_regime)
                    # More lenient filtering to capture more opportunities
                    if metrics.confidence > 0.1 or abs(metrics.composite_score) > 0.08:
                        signal_metrics.append(metrics)
                except Exception as e:
                    logger.error(f"Failed to aggregate signals for {symbol}: {e}")

            logger.info(f"Generated {len(signal_metrics)} qualified signal metrics")

            # Step 3: Optimize portfolio
            decisions = self.portfolio_optimizer.optimize_portfolio(
                signal_metrics, current_portfolio, market_data
            )

            # Step 4: Filter and validate decisions
            final_decisions = self._validate_decisions(decisions, current_portfolio)

            # Step 5: Update state
            self.last_decisions = final_decisions
            self.decision_history.append({
                'timestamp': datetime.now(),
                'decisions': final_decisions,
                'market_regime': market_regime,
                'signal_count': len(signal_metrics)
            })

            # Keep only recent history
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-1000:]

            logger.info(f"Generated {len(final_decisions)} portfolio decisions")
            return final_decisions

        except Exception as e:
            logger.error(f"Portfolio decision making failed: {e}")
            return {}

    def _validate_decisions(self, decisions: Dict[str, PortfolioDecision],
                          current_portfolio: Dict[str, float]) -> Dict[str, PortfolioDecision]:
        """Validate and filter decisions based on risk constraints"""
        validated = {}

        total_target_weight = sum(d.target_weight for d in decisions.values())

        # Normalize if total weight exceeds 1.0
        if total_target_weight > 1.0:
            for symbol, decision in decisions.items():
                decision.target_weight /= total_target_weight

        # Filter decisions based on conviction and risk
        for symbol, decision in decisions.items():
            # More lenient conviction threshold
            if decision.conviction < 0.1:
                continue

            # More aggressive rebalancing threshold
            weight_change = abs(decision.target_weight - decision.current_weight)
            if weight_change < 0.01 and decision.action != 'sell':  # Only 1% threshold
                decision.action = 'hold'

            # Validate risk constraints
            if decision.expected_risk > 0.3:  # Max 30% expected volatility
                logger.warning(f"High risk decision for {symbol}: {decision.expected_risk:.2f}")
                decision.target_weight *= 0.8  # Reduce position size

            validated[symbol] = decision

        return validated

    def _regularize_covariance_matrix(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Regularize covariance matrix to ensure numerical stability"""
        try:
            # Check for NaN/Inf values
            if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
                logger.warning("Invalid covariance matrix detected, using identity matrix")
                return np.eye(cov_matrix.shape[0]) * 0.04  # 4% volatility assumption

            # Ensure symmetry
            cov_matrix = (cov_matrix + cov_matrix.T) / 2

            # Check positive definiteness
            eigenvals = np.linalg.eigvals(cov_matrix)
            if np.any(eigenvals <= 0):
                logger.debug("Covariance matrix not positive definite, applying regularization")
                # Add small value to diagonal for regularization
                regularization = max(0.001, -np.min(eigenvals) + 0.001)
                cov_matrix += regularization * np.eye(cov_matrix.shape[0])

            # Check condition number
            cond_num = np.linalg.cond(cov_matrix)
            if cond_num > 1e12:
                logger.debug(f"High condition number ({cond_num:.2e}), applying stronger regularization")
                cov_matrix += 0.01 * np.eye(cov_matrix.shape[0])

            return cov_matrix

        except Exception as e:
            logger.error(f"Covariance regularization failed: {e}")
            n = cov_matrix.shape[0] if len(cov_matrix.shape) > 0 else 1
            return np.eye(n) * 0.04

    def _risk_parity_optimization(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Risk parity optimization - equal risk contribution"""
        n_assets = cov_matrix.shape[0]

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        def risk_parity_objective(weights):
            # Portfolio volatility
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

            # Marginal risk contributions
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol

            # Risk contributions
            risk_contrib = weights * marginal_contrib

            # Target equal risk contribution
            target_contrib = portfolio_vol / n_assets

            # Minimize deviation from equal risk contribution
            return np.sum((risk_contrib - target_contrib) ** 2)

        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0.001, 0.5) for _ in range(n_assets)]  # Relaxed bounds

        result = minimize(risk_parity_objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 500, 'ftol': 1e-9})

        if result.success:
            return result.x
        else:
            raise ValueError(f"Risk parity optimization failed: {result.message}")

    def _mean_variance_optimization(self, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Traditional mean-variance optimization with relaxed constraints"""
        n_assets = len(expected_returns)

        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            # Reduced risk aversion for more aggressive allocation
            return -(portfolio_return - 1.0 * portfolio_variance)

        # Relaxed constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]

        # More flexible bounds
        min_weight = max(0.001, 1.0 / (n_assets * 2))  # Dynamic minimum
        max_weight = min(0.3, 3.0 / n_assets)  # Dynamic maximum
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]

        # Better initial guess based on expected returns
        if np.std(expected_returns) > 0:
            # Weight by relative expected returns
            positive_returns = np.maximum(expected_returns, 0)
            if np.sum(positive_returns) > 0:
                x0 = positive_returns / np.sum(positive_returns)
            else:
                x0 = np.ones(n_assets) / n_assets
        else:
            x0 = np.ones(n_assets) / n_assets

        # Multiple optimization attempts with different methods
        methods = ['SLSQP', 'trust-constr']

        for method in methods:
            try:
                result = minimize(objective, x0, method=method, bounds=bounds,
                                constraints=constraints, options={'maxiter': 1000})

                if result.success and self._validate_weights(result.x):
                    return result.x

            except Exception as e:
                logger.debug(f"Method {method} failed: {e}")
                continue

        raise ValueError("All mean-variance optimization methods failed")

    def _validate_weights(self, weights: np.ndarray) -> bool:
        """Validate portfolio weights"""
        try:
            # Check for NaN/Inf
            if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                return False

            # Check non-negative
            if np.any(weights < -1e-6):  # Small tolerance for numerical errors
                return False

            # Check sum approximately equals 1
            weight_sum = np.sum(weights)
            if abs(weight_sum - 1.0) > 1e-3:
                return False

            # Check reasonable bounds
            if np.any(weights > 0.8):  # No single position > 80%
                return False

            return True

        except Exception:
            return False

    def get_decision_summary(self) -> Dict[str, Any]:
        """Get summary of current decisions for dashboard/logging"""
        if not self.last_decisions:
            return {'no_decisions': True}

        summary = {
            'total_decisions': len(self.last_decisions),
            'buy_decisions': len([d for d in self.last_decisions.values() if d.action == 'buy']),
            'sell_decisions': len([d for d in self.last_decisions.values() if d.action == 'sell']),
            'hold_decisions': len([d for d in self.last_decisions.values() if d.action == 'hold']),
            'avg_conviction': np.mean([d.conviction for d in self.last_decisions.values()]),
            'total_target_allocation': sum(d.target_weight for d in self.last_decisions.values() if d.action != 'sell'),
            'decisions': []
        }

        # Add top decisions
        sorted_decisions = sorted(
            self.last_decisions.values(),
            key=lambda x: x.conviction,
            reverse=True
        )[:10]

        for decision in sorted_decisions:
            summary['decisions'].append({
                'symbol': decision.symbol,
                'action': decision.action,
                'target_weight': f"{decision.target_weight:.1%}",
                'conviction': f"{decision.conviction:.1%}",
                'expected_return': f"{decision.expected_return:.1%}",
                'reasoning': decision.reasoning
            })

        return summary

    def export_decisions_to_dashboard(self) -> Dict[str, Any]:
        """Export decisions in format compatible with dashboard"""
        if not self.last_decisions:
            return {}

        dashboard_data = {
            'portfolio_recommendations': [],
            'risk_analysis': {
                'total_positions': len(self.last_decisions),
                'avg_conviction': np.mean([d.conviction for d in self.last_decisions.values()]),
                'high_conviction_count': len([d for d in self.last_decisions.values() if d.conviction > 0.7])
            },
            'sector_allocation': {},
            'expected_returns': {}
        }

        for symbol, decision in self.last_decisions.items():
            dashboard_data['portfolio_recommendations'].append({
                'symbol': symbol,
                'action': decision.action.upper(),
                'current_weight': f"{decision.current_weight:.1%}",
                'target_weight': f"{decision.target_weight:.1%}",
                'conviction': f"{decision.conviction:.1%}",
                'expected_return': f"{decision.expected_return:.1%}",
                'expected_risk': f"{decision.expected_risk:.1%}",
                'reasoning': decision.reasoning,
                'market_regime': decision.metadata.get('market_regime', 'unknown')
            })

            # Expected returns by symbol
            dashboard_data['expected_returns'][symbol] = decision.expected_return

        return dashboard_data

    async def record_decision_performance(self, symbol: str, decision: PortfolioDecision,
                                        actual_return: float, time_horizon: int = 5):
        """Record decision performance using existing database infrastructure"""
        try:
            if not self.db_manager or not self.performance_feedback_enabled:
                return

            # Use existing trading_signals table to store decision performance
            cursor = self.db_manager.connection.cursor()

            # Store performance record
            cursor.execute("""
                INSERT INTO trading_signals (symbol, signal_type, confidence, price, strategy, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                symbol,
                'decision_performance',  # Special signal type for tracking
                decision.conviction,
                0.0,  # Not applicable for performance records
                'AdvancedDecisionEngine',
                {
                    'actual_return': actual_return,
                    'expected_return': decision.expected_return,
                    'time_horizon': time_horizon,
                    'market_regime': decision.metadata.get('market_regime'),
                    'decision_action': decision.action,
                    'target_weight': decision.target_weight,
                    'reasoning': decision.reasoning,
                    'prediction_accuracy': 1.0 if (decision.expected_return > 0 and actual_return > 0) or (decision.expected_return < 0 and actual_return < 0) else 0.0
                }
            ))

            cursor.close()

            # Update ML ensemble with performance feedback
            if self.ensemble_predictor and hasattr(self.ensemble_predictor, 'update_model_performance'):
                # Extract individual model contributions from decision metadata
                signal_metrics = decision.metadata.get('signal_metrics')
                if signal_metrics:
                    model_predictions = {
                        'technical': getattr(signal_metrics, 'technical_score', 0),
                        'ml': getattr(signal_metrics, 'ml_score', 0),
                        'sentiment': getattr(signal_metrics, 'sentiment_score', 0),
                        'fundamental': getattr(signal_metrics, 'fundamental_score', 0),
                        'momentum': getattr(signal_metrics, 'momentum_score', 0)
                    }

                    # Update ensemble performance tracking
                    await self.ensemble_predictor.update_model_performance(model_predictions, actual_return)

            # Update signal aggregator performance tracking
            await self._update_signal_aggregator_performance(symbol, decision, actual_return)

        except Exception as e:
            logger.error(f"Error recording decision performance: {e}")

    async def _update_signal_aggregator_performance(self, symbol: str, decision: PortfolioDecision,
                                                  actual_return: float):
        """Update signal aggregator with performance feedback"""
        try:
            # Update ML ensemble performance data that will be used by signal aggregator
            signal_metrics = decision.metadata.get('signal_metrics')
            if signal_metrics:
                # Extract individual signal performances
                individual_performances = {
                    'technical': {
                        'prediction': getattr(signal_metrics, 'technical_score', 0),
                        'actual': actual_return,
                        'success': (getattr(signal_metrics, 'technical_score', 0) > 0 and actual_return > 0) or
                                 (getattr(signal_metrics, 'technical_score', 0) < 0 and actual_return < 0)
                    },
                    'ml': {
                        'prediction': getattr(signal_metrics, 'ml_score', 0),
                        'actual': actual_return,
                        'success': (getattr(signal_metrics, 'ml_score', 0) > 0 and actual_return > 0) or
                                 (getattr(signal_metrics, 'ml_score', 0) < 0 and actual_return < 0)
                    },
                    'sentiment': {
                        'prediction': getattr(signal_metrics, 'sentiment_score', 0),
                        'actual': actual_return,
                        'success': (getattr(signal_metrics, 'sentiment_score', 0) > 0 and actual_return > 0) or
                                 (getattr(signal_metrics, 'sentiment_score', 0) < 0 and actual_return < 0)
                    }
                }

                # Update signal aggregator's performance tracking
                if not hasattr(self.signal_aggregator, 'ml_ensemble_performance'):
                    self.signal_aggregator.ml_ensemble_performance = {}

                for signal_type, perf in individual_performances.items():
                    if signal_type not in self.signal_aggregator.ml_ensemble_performance:
                        self.signal_aggregator.ml_ensemble_performance[signal_type] = {
                            'recent_accuracy': 0.5,
                            'total_predictions': 0,
                            'correct_predictions': 0
                        }

                    perf_data = self.signal_aggregator.ml_ensemble_performance[signal_type]
                    perf_data['total_predictions'] += 1

                    if perf['success']:
                        perf_data['correct_predictions'] += 1

                    # Exponential moving average for recent accuracy
                    new_accuracy = perf_data['correct_predictions'] / perf_data['total_predictions']
                    perf_data['recent_accuracy'] = perf_data['recent_accuracy'] * 0.8 + new_accuracy * 0.2

        except Exception as e:
            logger.error(f"Error updating signal aggregator performance: {e}")

    async def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics using existing database"""
        try:
            if not self.db_manager or not self.db_manager.connection:
                return {'error': 'Database not available'}

            cursor = self.db_manager.connection.cursor()

            # Get decision performance records from last 90 days
            cursor.execute("""
                SELECT symbol, confidence, metadata, timestamp
                FROM trading_signals
                WHERE signal_type = 'decision_performance'
                AND timestamp > NOW() - INTERVAL '90 days'
                ORDER BY timestamp DESC
            """)

            performance_records = cursor.fetchall()
            cursor.close()

            if not performance_records:
                return {'message': 'No performance data available yet'}

            # Calculate analytics
            analytics = {
                'total_decisions': len(performance_records),
                'accuracy_by_signal_type': {},
                'accuracy_by_regime': {},
                'avg_conviction': 0.0,
                'recent_performance': {},
                'improvement_over_time': {}
            }

            # Process records
            accurate_predictions = 0
            total_conviction = 0
            signal_type_performance = {}
            regime_performance = {}

            for record in performance_records:
                symbol, confidence, metadata, timestamp = record
                total_conviction += confidence

                # Extract performance data
                if isinstance(metadata, dict):
                    accuracy = metadata.get('prediction_accuracy', 0)
                    regime = metadata.get('market_regime', 'unknown')

                    if accuracy > 0.5:  # Consider as accurate
                        accurate_predictions += 1

                    # Track by regime
                    if regime not in regime_performance:
                        regime_performance[regime] = {'accurate': 0, 'total': 0}

                    regime_performance[regime]['total'] += 1
                    if accuracy > 0.5:
                        regime_performance[regime]['accurate'] += 1

            # Calculate final metrics
            analytics['overall_accuracy'] = accurate_predictions / len(performance_records)
            analytics['avg_conviction'] = total_conviction / len(performance_records)

            # Accuracy by regime
            for regime, perf in regime_performance.items():
                analytics['accuracy_by_regime'][regime] = perf['accurate'] / perf['total'] if perf['total'] > 0 else 0

            # Recent performance (last 30 days)
            recent_records = [r for r in performance_records if (datetime.now() - r[3]).days <= 30]
            if recent_records:
                recent_accurate = sum(1 for r in recent_records if r[2].get('prediction_accuracy', 0) > 0.5)
                analytics['recent_performance'] = {
                    'accuracy': recent_accurate / len(recent_records),
                    'total_decisions': len(recent_records)
                }

            return analytics

        except Exception as e:
            logger.error(f"Error getting performance analytics: {e}")
            return {'error': str(e)}

    async def update_from_ml_ensemble_feedback(self):
        """Update decision engine based on ML ensemble performance feedback"""
        try:
            if not self.ensemble_predictor:
                return

            # Get ML ensemble performance metrics
            if hasattr(self.ensemble_predictor, 'get_performance_metrics'):
                ml_metrics = await self.ensemble_predictor.get_performance_metrics()

                if ml_metrics:
                    # Update signal aggregator with ML performance data
                    if not hasattr(self.signal_aggregator, 'ml_ensemble_performance'):
                        self.signal_aggregator.ml_ensemble_performance = {}

                    self.signal_aggregator.ml_ensemble_performance.update({
                        'traditional': ml_metrics.get('traditional_ml', {}),
                        'transformer': ml_metrics.get('transformer_ml', {}),
                        'ensemble': ml_metrics.get('ensemble', {}),
                        'last_updated': datetime.now()
                    })

                    logger.info("Updated decision engine with ML ensemble feedback")

        except Exception as e:
            logger.error(f"Error updating from ML ensemble feedback: {e}")

    async def check_and_trigger_retraining(self) -> bool:
        """Check if full model retraining is needed and trigger if necessary"""
        try:
            # Check time-based retraining
            time_based_retrain = False
            if self.last_full_retrain:
                days_since_retrain = (datetime.now() - self.last_full_retrain).days
                time_based_retrain = days_since_retrain >= self.retrain_interval_days

            # Check performance-based retraining
            performance_analytics = await self.get_performance_analytics()
            performance_based_retrain = False

            if isinstance(performance_analytics, dict) and 'overall_accuracy' in performance_analytics:
                current_accuracy = performance_analytics['overall_accuracy']
                performance_based_retrain = current_accuracy < self.retrain_threshold_accuracy

                logger.info(f"Current model accuracy: {current_accuracy:.1%} (threshold: {self.retrain_threshold_accuracy:.1%})")

            # Trigger retraining if needed
            if time_based_retrain or performance_based_retrain:
                reason = []
                if time_based_retrain:
                    reason.append(f"time-based ({days_since_retrain} days)")
                if performance_based_retrain:
                    reason.append(f"performance-based ({current_accuracy:.1%} < {self.retrain_threshold_accuracy:.1%})")

                logger.warning(f"Model retraining triggered: {', '.join(reason)}")

                # Try to trigger automatic retraining
                success = await self._trigger_automatic_retraining()

                if success:
                    self.last_full_retrain = datetime.now()
                    logger.info("Automatic retraining completed successfully")
                else:
                    logger.error("Automatic retraining failed - manual intervention required")
                    await self._send_retraining_alert()

                return success

            return False

        except Exception as e:
            logger.error(f"Error checking retraining needs: {e}")
            return False

    async def _trigger_automatic_retraining(self) -> bool:
        """Trigger automatic ML model retraining"""
        try:
            logger.info("Attempting automatic model retraining...")

            # Option 1: Use parallel trainer if available
            if hasattr(self, 'parallel_trainer') and self.parallel_trainer:
                # Trigger parallel training for all symbols
                training_results = await self.parallel_trainer.train_all_symbols_async()
                success_rate = sum(1 for r in training_results.values() if r.get('status') == 'success') / len(training_results)

                if success_rate > 0.7:  # 70% success rate threshold
                    logger.info(f"Parallel retraining completed with {success_rate:.1%} success rate")
                    return True

            # Option 2: Use Cloud VM if configured
            vm_training_success = await self._trigger_cloud_vm_training()
            if vm_training_success:
                return True

            # Option 3: Local incremental retraining
            logger.info("Falling back to local incremental retraining...")
            incremental_success = await self._perform_incremental_retraining()
            return incremental_success

        except Exception as e:
            logger.error(f"Automatic retraining failed: {e}")
            return False

    async def _trigger_cloud_vm_training(self) -> bool:
        """Trigger training on GCP VM using PowerShell script"""
        try:
            import subprocess
            import os

            # Check if PowerShell script exists
            script_path = Path(__file__).parent.parent / "gcp" / "ml-training-setup.ps1"
            if not script_path.exists():
                logger.debug("GCP training script not found")
                return False

            logger.info("Triggering cloud VM training...")

            # Execute PowerShell script to start training VM
            result = subprocess.run([
                "powershell.exe", "-ExecutionPolicy", "Bypass",
                "-File", str(script_path)
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                logger.info("Cloud VM training initiated successfully")
                return True
            else:
                logger.error(f"Cloud VM training failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error triggering cloud VM training: {e}")
            return False

    async def _perform_incremental_retraining(self) -> bool:
        """Perform incremental model retraining with available data"""
        try:
            if not self.ensemble_predictor:
                return False

            logger.info("Starting incremental model retraining...")

            # Get recent performance data for retraining
            if self.db_manager and self.db_manager.connection:
                cursor = self.db_manager.connection.cursor()

                # Get recent trading signals and their performance
                cursor.execute("""
                    SELECT symbol, metadata, timestamp
                    FROM trading_signals
                    WHERE signal_type = 'decision_performance'
                    AND timestamp > NOW() - INTERVAL '30 days'
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """)

                recent_data = cursor.fetchall()
                cursor.close()

                if len(recent_data) < 50:  # Need minimum data for retraining
                    logger.warning("Insufficient data for incremental retraining")
                    return False

                # Extract features and labels for retraining
                training_features = []
                training_labels = []

                for symbol, metadata, timestamp in recent_data:
                    if isinstance(metadata, dict):
                        # Extract features from metadata
                        features = self._extract_features_from_metadata(metadata)
                        actual_return = metadata.get('actual_return', 0)

                        if features:
                            training_features.append(features)
                            training_labels.append(actual_return)

                if len(training_features) >= 20:
                    # Perform incremental update of Bayesian weights
                    for i, (features, label) in enumerate(zip(training_features, training_labels)):
                        # Update ensemble with recent performance
                        model_predictions = {
                            'traditional_ml': features.get('traditional_prediction', 0),
                            'transformer_ml': features.get('transformer_prediction', 0),
                            'ensemble': features.get('ensemble_prediction', 0)
                        }

                        if hasattr(self.ensemble_predictor, 'bayesian_averaging'):
                            self.ensemble_predictor.bayesian_averaging.update_model_performance(
                                model_predictions, label
                            )

                    logger.info(f"Incremental retraining completed with {len(training_features)} samples")
                    return True

            return False

        except Exception as e:
            logger.error(f"Incremental retraining failed: {e}")
            return False

    def _extract_features_from_metadata(self, metadata: Dict) -> Optional[Dict]:
        """Extract training features from decision metadata"""
        try:
            if not isinstance(metadata, dict):
                return None

            # Extract relevant features for retraining
            features = {
                'traditional_prediction': metadata.get('traditional_ml_score', 0),
                'transformer_prediction': metadata.get('transformer_ml_score', 0),
                'ensemble_prediction': metadata.get('expected_return', 0),
                'technical_score': metadata.get('technical_score', 0),
                'sentiment_score': metadata.get('sentiment_score', 0),
                'fundamental_score': metadata.get('fundamental_score', 0),
                'market_regime': metadata.get('market_regime', 'unknown'),
                'conviction': metadata.get('conviction', 0)
            }

            return features

        except Exception as e:
            logger.debug(f"Error extracting features: {e}")
            return None

    async def _send_retraining_alert(self):
        """Send alert when manual retraining intervention is needed"""
        try:
            alert_message = {
                'type': 'retraining_alert',
                'timestamp': datetime.now().isoformat(),
                'message': 'ML model performance below threshold - manual retraining recommended',
                'details': {
                    'current_accuracy': getattr(self, 'last_accuracy_check', 'unknown'),
                    'threshold': self.retrain_threshold_accuracy,
                    'last_retrain': self.last_full_retrain.isoformat() if self.last_full_retrain else 'never',
                    'recommended_action': 'Run gcp/ml-training-setup.ps1 for full retraining'
                }
            }

            # Log the alert
            logger.error(f"RETRAINING ALERT: {alert_message['message']}")
            logger.error(f"Recommended action: {alert_message['details']['recommended_action']}")

            # Store alert in database if available
            if self.db_manager and self.db_manager.connection:
                cursor = self.db_manager.connection.cursor()
                cursor.execute("""
                    INSERT INTO trading_signals (symbol, signal_type, confidence, price, strategy, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    'SYSTEM',
                    'retraining_alert',
                    0.0,
                    0.0,
                    'AdvancedDecisionEngine',
                    alert_message
                ))
                cursor.close()

        except Exception as e:
            logger.error(f"Error sending retraining alert: {e}")