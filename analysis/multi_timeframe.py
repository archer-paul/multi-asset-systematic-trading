"""
Multi-Timeframe Analysis Module for Trading Bot
Analyzes multiple time horizons to improve prediction accuracy
"""

import logging
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import talib

logger = logging.getLogger(__name__)

class MultiTimeframeAnalyzer:
    """
    Analyzes multiple timeframes to provide comprehensive market insights
    Combines short-term (1D), medium-term (1W), and long-term (1M) analysis
    """
    
    def __init__(self, config=None):
        self.config = config
        
        # Define timeframes for analysis
        self.timeframes = {
            'short_term': {'period': '1d', 'interval': '5m', 'lookback_days': 5},
            'medium_term': {'period': '1mo', 'interval': '1h', 'lookback_days': 30}, 
            'long_term': {'period': '3mo', 'interval': '1d', 'lookback_days': 90}
        }
        
        # Technical indicators configuration
        self.indicators = {
            'trend': ['SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD', 'ICHIMOKU'],
            'momentum': ['RSI', 'STOCH', 'WILLIAMS_R', 'CCI'],
            'volatility': ['BBANDS', 'ATR', 'VOLATILITY'],
            'volume': ['OBV', 'AD', 'VOLUME_SMA']
        }
        
        # Scoring weights for different timeframes
        self.timeframe_weights = {
            'short_term': 0.2,    # 20% weight for short-term signals
            'medium_term': 0.5,   # 50% weight for medium-term signals
            'long_term': 0.3      # 30% weight for long-term signals
        }
        
        self.scalers = {}
        self.models = {}
        
    async def analyze_multi_timeframe(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Perform multi-timeframe analysis for given symbols with improved error handling
        """
        results = {}

        # Pre-filter known problematic symbols
        valid_symbols = await self._filter_valid_symbols(symbols)

        for symbol in valid_symbols:
            try:
                logger.info(f"Starting multi-timeframe analysis for {symbol}")

                # Collect data for all timeframes
                timeframe_data = await self._collect_timeframe_data(symbol)

                if not timeframe_data:
                    logger.warning(f"No data collected for {symbol}, skipping analysis")
                    continue
                
                # Calculate technical indicators for each timeframe
                technical_analysis = await asyncio.to_thread(self._calculate_technical_indicators, timeframe_data)
                
                # Generate signals for each timeframe
                timeframe_signals = await asyncio.to_thread(self._generate_timeframe_signals, technical_analysis)
                
                # Combine signals into final score
                final_analysis = await asyncio.to_thread(self._combine_timeframe_analysis, timeframe_signals)
                
                # Add market regime detection
                market_regime = await asyncio.to_thread(self._detect_market_regime, timeframe_data)
                final_analysis['market_regime'] = market_regime
                
                # Calculate confidence score
                confidence = await asyncio.to_thread(self._calculate_confidence_score, timeframe_signals, market_regime)
                final_analysis['confidence'] = confidence
                
                results[symbol] = final_analysis
                
                logger.info(f"Completed multi-timeframe analysis for {symbol}: Score={final_analysis['composite_score']:.2f}, Confidence={confidence:.2f}")
                
            except Exception as e:
                error_msg = str(e).lower()
                if 'possibly delisted' in error_msg or 'no price data found' in error_msg:
                    logger.warning(f"Symbol {symbol} appears to be delisted, skipping")
                else:
                    logger.error(f"Multi-timeframe analysis failed for {symbol}: {e}")
                    results[symbol] = self._get_neutral_analysis()
        
        return results

    async def _filter_valid_symbols(self, symbols: List[str]) -> List[str]:
        """
        Pre-filter symbols to remove obviously invalid ones
        """
        valid_symbols = []

        # More relaxed filtering - only skip truly invalid symbols
        skip_patterns = [
            '--', '???', 'N/A',  # Removed '.' and '' to allow valid symbols
        ]

        # Known delisted or problematic symbols (can be expanded)
        known_delisted = {
            # Add more as discovered
        }

        logger.debug(f"Starting symbol filtering for {len(symbols)} symbols: {symbols[:10]}...")

        for symbol in symbols:
            # More lenient checking
            if not symbol or len(symbol.strip()) == 0:
                logger.debug(f"Skipping empty symbol: '{symbol}'")
                continue

            if any(pattern in symbol for pattern in skip_patterns):
                logger.debug(f"Skipping invalid symbol format: {symbol}")
                continue

            # Skip known delisted symbols
            if symbol in known_delisted:
                logger.debug(f"Skipping known problematic symbol: {symbol}")
                continue

            valid_symbols.append(symbol)
            logger.debug(f"Added valid symbol: {symbol}")

        logger.info(f"Filtered {len(symbols)} symbols to {len(valid_symbols)} valid symbols")
        return valid_symbols

    async def _collect_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Collect price data for all timeframes with robust error handling
        """
        timeframe_data = {}

        # Try to get basic info first to validate symbol
        try:
            ticker = yf.Ticker(symbol)
            info = await asyncio.to_thread(lambda: ticker.info)
            if not info or info.get('regularMarketPrice') is None:
                logger.warning(f"Symbol {symbol} appears to be invalid or delisted")
                return timeframe_data
        except Exception as e:
            logger.warning(f"Cannot validate symbol {symbol}: {e}")
            # Continue anyway, but be more careful

        for tf_name, tf_config in self.timeframes.items():
            try:
                ticker = yf.Ticker(symbol)

                # Get historical data with retries
                data = None
                for attempt in range(2):
                    try:
                        data = await asyncio.to_thread(
                            ticker.history,
                            period=tf_config['period'],
                            interval=tf_config['interval'],
                            auto_adjust=True,
                            prepost=False,  # Disable premarket/afterhours for better reliability
                            timeout=30
                        )
                        break
                    except Exception as retry_e:
                        if attempt == 0:
                            logger.debug(f"Retry {attempt + 1} for {symbol} {tf_name}: {retry_e}")
                            await asyncio.sleep(1)
                        else:
                            raise retry_e

                if data is not None and not data.empty and len(data) > 10:
                    # Validate data quality
                    if data['Close'].isna().sum() / len(data) < 0.5:  # Less than 50% NaN
                        timeframe_data[tf_name] = data
                        logger.debug(f"Collected {len(data)} bars for {symbol} on {tf_name} timeframe")
                    else:
                        logger.warning(f"Poor data quality for {symbol} on {tf_name} timeframe")
                else:
                    logger.warning(f"No data for {symbol} on {tf_name} timeframe")

            except Exception as e:
                error_msg = str(e).lower()
                if 'possibly delisted' in error_msg or 'no price data found' in error_msg:
                    logger.warning(f"Symbol {symbol} appears to be delisted or suspended")
                    # Try alternative data sources or skip this symbol
                    break  # Skip other timeframes for this symbol
                else:
                    logger.error(f"Failed to collect {tf_name} data for {symbol}: {e}")

        return timeframe_data
    
    def _calculate_technical_indicators(self, timeframe_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate technical indicators for each timeframe
        """
        technical_analysis = {}
        
        for tf_name, df in timeframe_data.items():
            if df.empty:
                continue
                
            try:
                indicators = {}
                
                # Price data
                high = df['High'].values.astype(np.double)
                low = df['Low'].values.astype(np.double)
                close = df['Close'].values.astype(np.double)
                volume = df['Volume'].values.astype(np.double)
                
                # Trend Indicators
                indicators['SMA_20'] = talib.SMA(close, timeperiod=20)
                indicators['SMA_50'] = talib.SMA(close, timeperiod=min(50, len(close)-1))
                indicators['EMA_12'] = talib.EMA(close, timeperiod=12)
                indicators['EMA_26'] = talib.EMA(close, timeperiod=26)
                
                # MACD
                macd, macd_signal, macd_hist = talib.MACD(close)
                indicators['MACD'] = macd
                indicators['MACD_SIGNAL'] = macd_signal
                indicators['MACD_HIST'] = macd_hist
                
                # Momentum Indicators
                indicators['RSI'] = talib.RSI(close, timeperiod=14)
                indicators['STOCH_K'], indicators['STOCH_D'] = talib.STOCH(high, low, close)
                indicators['WILLIAMS_R'] = talib.WILLR(high, low, close, timeperiod=14)
                indicators['CCI'] = talib.CCI(high, low, close, timeperiod=14)
                
                # Volatility Indicators
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
                indicators['BB_UPPER'] = bb_upper
                indicators['BB_MIDDLE'] = bb_middle
                indicators['BB_LOWER'] = bb_lower
                indicators['ATR'] = talib.ATR(high, low, close, timeperiod=14)
                
                # Volume Indicators
                indicators['OBV'] = talib.OBV(close, volume)
                indicators['AD'] = talib.AD(high, low, close, volume)

                # Ichimoku Cloud Indicators
                indicators['ICHIMOKU'] = self._calculate_ichimoku_cloud(high, low, close)

                # Custom volatility calculation
                returns = np.log(close[1:] / close[:-1])
                indicators['VOLATILITY'] = pd.Series(returns).rolling(20).std() * np.sqrt(252)
                
                technical_analysis[tf_name] = {
                    'data': df,
                    'indicators': indicators,
                    'latest_price': close[-1] if len(close) > 0 else 0
                }
                
            except Exception as e:
                logger.error(f"Technical indicator calculation failed for {tf_name}: {e}")
                technical_analysis[tf_name] = {'data': df, 'indicators': {}, 'latest_price': 0}
        
        return technical_analysis

    def _calculate_ichimoku_cloud(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate Ichimoku Cloud indicators
        """
        try:
            # Conversion Line (Tenkan-sen): (9-period high + 9-period low) / 2
            tenkan_high = pd.Series(high).rolling(window=9).max()
            tenkan_low = pd.Series(low).rolling(window=9).min()
            tenkan_sen = (tenkan_high + tenkan_low) / 2

            # Base Line (Kijun-sen): (26-period high + 26-period low) / 2
            kijun_high = pd.Series(high).rolling(window=26).max()
            kijun_low = pd.Series(low).rolling(window=26).min()
            kijun_sen = (kijun_high + kijun_low) / 2

            # Leading Span A (Senkou Span A): (Conversion Line + Base Line) / 2, displaced 26 periods ahead
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

            # Leading Span B (Senkou Span B): (52-period high + 52-period low) / 2, displaced 26 periods ahead
            senkou_high = pd.Series(high).rolling(window=52).max()
            senkou_low = pd.Series(low).rolling(window=52).min()
            senkou_span_b = ((senkou_high + senkou_low) / 2).shift(26)

            # Lagging Span (Chikou Span): Close displaced 26 periods behind
            chikou_span = pd.Series(close).shift(-26)

            return {
                'tenkan_sen': tenkan_sen.values,
                'kijun_sen': kijun_sen.values,
                'senkou_span_a': senkou_span_a.values,
                'senkou_span_b': senkou_span_b.values,
                'chikou_span': chikou_span.values,
                'cloud_color': np.where(senkou_span_a > senkou_span_b, 'bullish', 'bearish')
            }
        except Exception as e:
            logger.error(f"Ichimoku calculation failed: {e}")
            return {}

    def _generate_timeframe_signals(self, technical_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Generate trading signals for each timeframe
        """
        timeframe_signals = {}
        
        for tf_name, analysis in technical_analysis.items():
            try:
                signals = {}
                indicators = analysis['indicators']
                latest_price = analysis['latest_price']
                
                if not indicators or latest_price == 0:
                    timeframe_signals[tf_name] = self._get_neutral_signals()
                    continue
                
                # Trend Signals
                signals['trend_score'] = self._calculate_trend_score(indicators, latest_price)
                
                # Momentum Signals  
                signals['momentum_score'] = self._calculate_momentum_score(indicators)
                
                # Volatility Signals
                signals['volatility_score'] = self._calculate_volatility_score(indicators, latest_price)
                
                # Volume Signals
                signals['volume_score'] = self._calculate_volume_score(indicators)
                
                # Overall timeframe score
                signals['timeframe_score'] = np.mean([
                    signals['trend_score'],
                    signals['momentum_score'], 
                    signals['volatility_score'],
                    signals['volume_score']
                ])
                
                timeframe_signals[tf_name] = signals
                
            except Exception as e:
                logger.error(f"Signal generation failed for {tf_name}: {e}")
                timeframe_signals[tf_name] = self._get_neutral_signals()
        
        return timeframe_signals
    
    def _calculate_trend_score(self, indicators: Dict, latest_price: float) -> float:
        """Calculate trend score from technical indicators"""
        try:
            score = 0.0
            count = 0
            
            # SMA trend
            if 'SMA_20' in indicators and not np.isnan(indicators['SMA_20'][-1]):
                sma20 = indicators['SMA_20'][-1]
                score += 1.0 if latest_price > sma20 else -1.0
                count += 1
                
            if 'SMA_50' in indicators and not np.isnan(indicators['SMA_50'][-1]):
                sma50 = indicators['SMA_50'][-1]
                score += 1.0 if latest_price > sma50 else -1.0
                count += 1
            
            # EMA trend
            if 'EMA_12' in indicators and 'EMA_26' in indicators:
                ema12 = indicators['EMA_12'][-1]
                ema26 = indicators['EMA_26'][-1]
                if not (np.isnan(ema12) or np.isnan(ema26)):
                    score += 1.0 if ema12 > ema26 else -1.0
                    count += 1
            
            # MACD trend
            if 'MACD' in indicators and 'MACD_SIGNAL' in indicators:
                macd = indicators['MACD'][-1]
                macd_signal = indicators['MACD_SIGNAL'][-1]
                if not (np.isnan(macd) or np.isnan(macd_signal)):
                    score += 1.0 if macd > macd_signal else -1.0
                    count += 1

            # Ichimoku Cloud trend
            if 'ICHIMOKU' in indicators and indicators['ICHIMOKU']:
                ichimoku = indicators['ICHIMOKU']
                if len(ichimoku.get('tenkan_sen', [])) > 0 and len(ichimoku.get('kijun_sen', [])) > 0:
                    tenkan = ichimoku['tenkan_sen'][-1]
                    kijun = ichimoku['kijun_sen'][-1]
                    if not (np.isnan(tenkan) or np.isnan(kijun)):
                        # Tenkan above Kijun is bullish
                        score += 1.0 if tenkan > kijun else -1.0
                        count += 1

                    # Price position relative to cloud
                    if (len(ichimoku.get('senkou_span_a', [])) > 0 and
                        len(ichimoku.get('senkou_span_b', [])) > 0):
                        senkou_a = ichimoku['senkou_span_a'][-1]
                        senkou_b = ichimoku['senkou_span_b'][-1]
                        if not (np.isnan(senkou_a) or np.isnan(senkou_b)):
                            cloud_top = max(senkou_a, senkou_b)
                            cloud_bottom = min(senkou_a, senkou_b)

                            if latest_price > cloud_top:
                                score += 1.0  # Above cloud - bullish
                            elif latest_price < cloud_bottom:
                                score += -1.0  # Below cloud - bearish
                            # Inside cloud is neutral (no score adjustment)
                            count += 1

            return score / count if count > 0 else 0.0
            
        except Exception as e:
            logger.debug(f"Trend score calculation error: {e}")
            return 0.0
    
    def _calculate_momentum_score(self, indicators: Dict) -> float:
        """Calculate momentum score from technical indicators"""
        try:
            score = 0.0
            count = 0
            
            # RSI
            if 'RSI' in indicators and not np.isnan(indicators['RSI'][-1]):
                rsi = indicators['RSI'][-1]
                if rsi > 70:
                    score += -1.0  # Overbought
                elif rsi < 30:
                    score += 1.0   # Oversold
                else:
                    score += (50 - rsi) / 20  # Normalized score
                count += 1
            
            # Stochastic
            if 'STOCH_K' in indicators and not np.isnan(indicators['STOCH_K'][-1]):
                stoch_k = indicators['STOCH_K'][-1]
                if stoch_k > 80:
                    score += -1.0  # Overbought
                elif stoch_k < 20:
                    score += 1.0   # Oversold
                else:
                    score += (50 - stoch_k) / 30
                count += 1
            
            # Williams %R
            if 'WILLIAMS_R' in indicators and not np.isnan(indicators['WILLIAMS_R'][-1]):
                wr = indicators['WILLIAMS_R'][-1]
                if wr > -20:
                    score += -1.0  # Overbought
                elif wr < -80:
                    score += 1.0   # Oversold
                else:
                    score += (wr + 50) / 30
                count += 1
            
            return score / count if count > 0 else 0.0
            
        except Exception as e:
            logger.debug(f"Momentum score calculation error: {e}")
            return 0.0
    
    def _calculate_volatility_score(self, indicators: Dict, latest_price: float) -> float:
        """Calculate volatility-based score"""
        try:
            score = 0.0
            count = 0
            
            # Bollinger Bands position
            if all(key in indicators for key in ['BB_UPPER', 'BB_LOWER', 'BB_MIDDLE']):
                bb_upper = indicators['BB_UPPER'][-1]
                bb_lower = indicators['BB_LOWER'][-1]
                bb_middle = indicators['BB_MIDDLE'][-1]
                
                if not any(np.isnan([bb_upper, bb_lower, bb_middle])):
                    if latest_price > bb_upper:
                        score += -0.5  # Price above upper band
                    elif latest_price < bb_lower:
                        score += 1.0   # Price below lower band - potential buy
                    else:
                        # Normalized position within bands
                        position = (latest_price - bb_lower) / (bb_upper - bb_lower)
                        score += 1.0 - position  # Higher score when closer to lower band
                    count += 1
            
            # ATR-based volatility assessment
            if 'ATR' in indicators and not np.isnan(indicators['ATR'][-1]):
                atr = indicators['ATR'][-1]
                atr_pct = (atr / latest_price) * 100
                
                # Moderate volatility is preferred
                if 1.0 < atr_pct < 3.0:
                    score += 0.5
                elif atr_pct > 5.0:
                    score += -0.5  # Too volatile
                count += 1
            
            return score / count if count > 0 else 0.0
            
        except Exception as e:
            logger.debug(f"Volatility score calculation error: {e}")
            return 0.0
    
    def _calculate_volume_score(self, indicators: Dict) -> float:
        """Calculate volume-based score"""
        try:
            score = 0.0
            count = 0
            
            # OBV trend
            if 'OBV' in indicators and len(indicators['OBV']) > 10:
                obv = indicators['OBV']
                obv_sma = np.nanmean(obv[-10:])  # 10-period average
                if not np.isnan(obv[-1]) and not np.isnan(obv_sma):
                    score += 1.0 if obv[-1] > obv_sma else -0.5
                    count += 1
            
            # A/D Line
            if 'AD' in indicators and len(indicators['AD']) > 5:
                ad = indicators['AD']
                if not np.isnan(ad[-1]) and not np.isnan(ad[-5]):
                    score += 1.0 if ad[-1] > ad[-5] else -0.5
                    count += 1
            
            return score / count if count > 0 else 0.0
            
        except Exception as e:
            logger.debug(f"Volume score calculation error: {e}")
            return 0.0
    
    def _combine_timeframe_analysis(self, timeframe_signals: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Combine signals from all timeframes into final analysis
        """
        try:
            # Calculate weighted composite score
            composite_score = 0.0
            total_weight = 0.0
            
            timeframe_scores = {}
            
            for tf_name, signals in timeframe_signals.items():
                if tf_name in self.timeframe_weights:
                    weight = self.timeframe_weights[tf_name]
                    tf_score = signals.get('timeframe_score', 0.0)
                    
                    composite_score += tf_score * weight
                    total_weight += weight
                    timeframe_scores[tf_name] = tf_score
            
            final_composite_score = composite_score / total_weight if total_weight > 0 else 0.0
            
            # Determine signal strength
            if final_composite_score > 0.5:
                signal = 'STRONG_BUY'
            elif final_composite_score > 0.2:
                signal = 'BUY'
            elif final_composite_score > -0.2:
                signal = 'HOLD'
            elif final_composite_score > -0.5:
                signal = 'SELL'
            else:
                signal = 'STRONG_SELL'
            
            return {
                'composite_score': final_composite_score,
                'signal': signal,
                'timeframe_scores': timeframe_scores,
                'timeframe_signals': timeframe_signals,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to combine timeframe analysis: {e}")
            return self._get_neutral_analysis()
    
    def _detect_market_regime(self, timeframe_data: Dict[str, pd.DataFrame]) -> str:
        """
        Detect current market regime (trending, ranging, volatile)
        """
        try:
            # Use long-term data for regime detection
            if 'long_term' not in timeframe_data or timeframe_data['long_term'].empty:
                return 'unknown'
            
            df = timeframe_data['long_term']
            closes = df['Close'].values
            
            if len(closes) < 20:
                return 'insufficient_data'
            
            # Calculate trend strength
            recent_closes = closes[-20:]
            trend_slope = np.polyfit(range(len(recent_closes)), recent_closes, 1)[0]
            price_std = np.std(recent_closes)
            avg_price = np.mean(recent_closes)
            
            trend_strength = abs(trend_slope) / (price_std / avg_price)
            volatility_ratio = price_std / avg_price
            
            if trend_strength > 0.5:
                if trend_slope > 0:
                    return 'uptrending'
                else:
                    return 'downtrending'
            elif volatility_ratio > 0.05:
                return 'volatile'
            else:
                return 'ranging'
                
        except Exception as e:
            logger.debug(f"Market regime detection error: {e}")
            return 'unknown'
    
    def _calculate_confidence_score(self, timeframe_signals: Dict[str, Dict[str, float]], market_regime: str) -> float:
        """
        Calculate confidence score based on signal alignment and market regime
        """
        try:
            confidence = 0.5  # Base confidence
            
            # Check signal alignment across timeframes
            scores = [signals.get('timeframe_score', 0.0) for signals in timeframe_signals.values()]
            
            if len(scores) >= 2:
                # Calculate agreement between timeframes
                score_std = np.std(scores)
                avg_score = np.mean(scores)
                
                # Higher confidence when signals align
                alignment_bonus = max(0, 0.3 - score_std)
                confidence += alignment_bonus
                
                # Higher confidence for stronger signals
                strength_bonus = min(0.2, abs(avg_score) * 0.2)
                confidence += strength_bonus
            
            # Market regime adjustment
            if market_regime in ['uptrending', 'downtrending']:
                confidence += 0.1  # More confident in trending markets
            elif market_regime == 'volatile':
                confidence -= 0.1  # Less confident in volatile markets
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.debug(f"Confidence calculation error: {e}")
            return 0.5
    
    def _get_neutral_signals(self) -> Dict[str, float]:
        """Return neutral signals when calculation fails"""
        return {
            'trend_score': 0.0,
            'momentum_score': 0.0,
            'volatility_score': 0.0,
            'volume_score': 0.0,
            'timeframe_score': 0.0
        }
    
    def _get_neutral_analysis(self) -> Dict[str, Any]:
        """Return neutral analysis when calculation fails"""
        return {
            'composite_score': 0.0,
            'signal': 'HOLD',
            'timeframe_scores': {},
            'timeframe_signals': {},
            'market_regime': 'unknown',
            'confidence': 0.5,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    async def get_market_summary(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get a summary of multi-timeframe analysis across all symbols
        """
        try:
            analysis_results = await self.analyze_multi_timeframe(symbols)
            
            if not analysis_results:
                return {'status': 'no_data', 'summary': 'No analysis data available'}
            
            # Aggregate statistics
            total_symbols = len(analysis_results)
            signals = [result['signal'] for result in analysis_results.values()]
            scores = [result['composite_score'] for result in analysis_results.values()]
            
            signal_distribution = {}
            for signal in ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']:
                signal_distribution[signal] = signals.count(signal)
            
            avg_score = np.mean(scores) if scores else 0.0
            
            # Top opportunities
            sorted_results = sorted(
                [(symbol, result) for symbol, result in analysis_results.items()],
                key=lambda x: x[1]['composite_score'],
                reverse=True
            )
            
            top_opportunities = [
                {
                    'symbol': symbol,
                    'score': result['composite_score'],
                    'signal': result['signal'],
                    'confidence': result['confidence']
                }
                for symbol, result in sorted_results[:10]
            ]
            
            return {
                'status': 'success',
                'total_symbols': total_symbols,
                'average_score': avg_score,
                'signal_distribution': signal_distribution,
                'top_opportunities': top_opportunities,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Market summary generation failed: {e}")
            return {'status': 'error', 'error': str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get analyzer status"""
        return {
            'timeframes_configured': len(self.timeframes),
            'indicators_available': sum(len(indicators) for indicators in self.indicators.values()),
            'status': 'ready'
        }