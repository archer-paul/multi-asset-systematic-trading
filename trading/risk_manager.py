"""
Advanced Risk Management Module for Trading Bot
Implements sophisticated risk management techniques including VaR calculation,
portfolio optimization, dynamic position sizing, and real-time risk monitoring
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import warnings
warnings.filterwarnings('ignore')

from trading.strategy import TradingSignal, Position, SignalType, PositionType

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Risk alert types"""
    POSITION_SIZE = "position_size"
    PORTFOLIO_CONCENTRATION = "portfolio_concentration"
    DRAWDOWN = "drawdown"
    VAR_BREACH = "var_breach"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"

@dataclass
class RiskAlert:
    """Risk alert data structure"""
    alert_type: AlertType
    risk_level: RiskLevel
    message: str
    timestamp: datetime
    affected_symbols: List[str]
    recommended_action: str
    metadata: Dict[str, Any]

@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    portfolio_value: float
    total_exposure: float
    var_1d: float
    var_5d: float
    expected_shortfall: float
    maximum_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    timestamp: datetime

class VaRCalculator:
    """Value at Risk calculator with multiple methodologies"""

    def __init__(self, confidence_levels: List[float] = None):
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.lookback_periods = [30, 60, 252]  # 1 month, 2 months, 1 year
        self.emergency_var_threshold = 0.15  # 15% max single-day VaR
    
    def calculate_parametric_var(self, returns: pd.Series, confidence_level: float = 0.95, 
                                portfolio_value: float = 1.0) -> float:
        """Calculate parametric VaR assuming normal distribution"""
        try:
            if len(returns) < 10:
                return portfolio_value * 0.05  # Default 5% VaR
            
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Z-score for confidence level
            from scipy import stats
            z_score = stats.norm.ppf(1 - confidence_level)
            
            # VaR calculation
            var = portfolio_value * (mean_return + z_score * std_return)
            
            return abs(var)
            
        except Exception as e:
            logger.error(f"Parametric VaR calculation failed: {e}")
            return portfolio_value * 0.05

    def calculate_historical_var(self, returns: pd.Series, confidence_level: float = 0.95,
                                portfolio_value: float = 1.0) -> float:
        """Calculate historical VaR using empirical distribution"""
        try:
            if len(returns) < 10:
                return portfolio_value * 0.05

            # Sort returns and find percentile
            sorted_returns = returns.sort_values()
            percentile_index = int((1 - confidence_level) * len(sorted_returns))

            if percentile_index == 0:
                var_return = sorted_returns.iloc[0]
            else:
                var_return = sorted_returns.iloc[percentile_index - 1]

            var = portfolio_value * abs(var_return)

            # Emergency cap
            max_var = portfolio_value * self.emergency_var_threshold
            return min(var, max_var)

        except Exception as e:
            logger.error(f"Historical VaR calculation failed: {e}")
            return portfolio_value * 0.05

    def calculate_monte_carlo_var(self, returns: pd.Series, confidence_level: float = 0.95,
                                 portfolio_value: float = 1.0, simulations: int = 1000) -> float:
        """Calculate Monte Carlo VaR"""
        try:
            if len(returns) < 10:
                return portfolio_value * 0.05

            mean_return = returns.mean()
            std_return = returns.std()

            # Generate random scenarios
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.normal(mean_return, std_return, simulations)

            # Calculate VaR
            var_return = np.percentile(simulated_returns, (1 - confidence_level) * 100)
            var = portfolio_value * abs(var_return)

            return min(var, portfolio_value * self.emergency_var_threshold)

        except Exception as e:
            logger.error(f"Monte Carlo VaR calculation failed: {e}")
            return portfolio_value * 0.05

class AdvancedRiskManager:
    """Enhanced risk manager with real-time monitoring and dynamic adjustments"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Risk limits
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.02)  # 2% daily VaR
        self.max_position_size = self.config.get('max_position_size', 0.1)  # 10% max position
        self.max_sector_concentration = self.config.get('max_sector_concentration', 0.3)  # 30% per sector
        self.max_drawdown_threshold = self.config.get('max_drawdown_threshold', 0.15)  # 15% max drawdown

        # Dynamic risk adjustment
        self.volatility_adjustment = self.config.get('volatility_adjustment', True)
        self.correlation_adjustment = self.config.get('correlation_adjustment', True)

        # Risk monitoring
        self.var_calculator = VaRCalculator()
        self.active_alerts = []
        self.risk_metrics_history = []

        logger.info("Advanced Risk Manager initialized with enhanced monitoring")

    async def assess_pre_trade_risk(self, signal: TradingSignal, current_positions: Dict[str, Position],
                                   portfolio_value: float, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive pre-trade risk assessment"""
        try:
            risk_assessment = {
                'approved': True,
                'recommended_size': signal.position_size,
                'risk_factors': [],
                'adjustments_made': [],
                'risk_score': 0.0
            }

            # 1. Position size validation
            max_size = self._calculate_max_position_size(signal.symbol, portfolio_value, market_data)
            if signal.position_size > max_size:
                risk_assessment['adjustments_made'].append(f"Position size reduced from {signal.position_size:.3f} to {max_size:.3f}")
                risk_assessment['recommended_size'] = max_size

            # 2. Portfolio concentration check
            concentration_risk = self._check_concentration_risk(signal, current_positions, portfolio_value)
            if concentration_risk['risk_level'] == RiskLevel.HIGH:
                risk_assessment['recommended_size'] *= 0.5  # Reduce by 50%
                risk_assessment['adjustments_made'].append("Position size reduced due to concentration risk")
                risk_assessment['risk_factors'].append(concentration_risk)

            # 3. Correlation risk assessment
            correlation_risk = self._assess_correlation_risk(signal, current_positions, market_data)
            if correlation_risk['high_correlation_count'] > 3:
                risk_assessment['recommended_size'] *= 0.7  # Reduce by 30%
                risk_assessment['adjustments_made'].append("Position size reduced due to correlation risk")
                risk_assessment['risk_factors'].append(correlation_risk)

            # 4. Volatility adjustment
            if self.volatility_adjustment:
                vol_adjustment = self._calculate_volatility_adjustment(signal.symbol, market_data)
                risk_assessment['recommended_size'] *= vol_adjustment
                if vol_adjustment < 1.0:
                    risk_assessment['adjustments_made'].append(f"Position size adjusted for volatility (factor: {vol_adjustment:.2f})")

            # 5. Calculate final risk score
            risk_assessment['risk_score'] = self._calculate_overall_risk_score(
                signal, current_positions, portfolio_value, market_data
            )

            # 6. Final approval check
            if risk_assessment['risk_score'] > 0.8:
                risk_assessment['approved'] = False
                risk_assessment['risk_factors'].append({
                    'type': 'high_risk_score',
                    'message': f"Overall risk score too high: {risk_assessment['risk_score']:.2f}"
                })

            # Ensure minimum position size
            if risk_assessment['recommended_size'] < 0.001:  # 0.1% minimum
                risk_assessment['approved'] = False
                risk_assessment['risk_factors'].append({
                    'type': 'position_too_small',
                    'message': "Adjusted position size below minimum threshold"
                })

            return risk_assessment

        except Exception as e:
            logger.error(f"Pre-trade risk assessment failed: {e}")
            return {
                'approved': False,
                'recommended_size': 0.0,
                'risk_factors': [{'type': 'assessment_error', 'message': str(e)}],
                'adjustments_made': [],
                'risk_score': 1.0
            }

    def _calculate_max_position_size(self, symbol: str, portfolio_value: float,
                                   market_data: Dict[str, Any]) -> float:
        """Calculate maximum allowed position size based on multiple factors"""
        try:
            # Base maximum position size
            base_max = self.max_position_size

            # Adjust for volatility
            symbol_data = market_data.get(symbol, {})
            volatility = symbol_data.get('volatility', 0.02)

            # Higher volatility = smaller position size
            vol_adjustment = min(1.0, 0.02 / max(volatility, 0.005))

            # Adjust for liquidity
            avg_volume = symbol_data.get('average_volume', 1000000)
            liquidity_adjustment = min(1.0, avg_volume / 500000)  # Scale based on 500k volume baseline

            # Combine adjustments
            adjusted_max = base_max * vol_adjustment * liquidity_adjustment

            return max(0.01, adjusted_max)  # Minimum 1%

        except Exception as e:
            logger.error(f"Max position size calculation failed: {e}")
            return 0.01

    def _check_concentration_risk(self, signal: TradingSignal, current_positions: Dict[str, Position],
                                 portfolio_value: float) -> Dict[str, Any]:
        """Check for portfolio concentration risk"""
        try:
            # Get sector information (simplified)
            symbol_sector = self._get_symbol_sector(signal.symbol)

            # Calculate current sector exposure
            sector_exposure = 0.0
            for position in current_positions.values():
                if self._get_symbol_sector(position.symbol) == symbol_sector:
                    sector_exposure += abs(position.position_size)

            # Add new position
            new_sector_exposure = sector_exposure + signal.position_size

            # Check limits
            if new_sector_exposure > self.max_sector_concentration:
                return {
                    'risk_level': RiskLevel.HIGH,
                    'current_exposure': sector_exposure,
                    'new_exposure': new_sector_exposure,
                    'limit': self.max_sector_concentration,
                    'sector': symbol_sector
                }
            elif new_sector_exposure > self.max_sector_concentration * 0.8:
                return {
                    'risk_level': RiskLevel.MEDIUM,
                    'current_exposure': sector_exposure,
                    'new_exposure': new_sector_exposure,
                    'limit': self.max_sector_concentration,
                    'sector': symbol_sector
                }
            else:
                return {
                    'risk_level': RiskLevel.LOW,
                    'current_exposure': sector_exposure,
                    'new_exposure': new_sector_exposure,
                    'limit': self.max_sector_concentration,
                    'sector': symbol_sector
                }

        except Exception as e:
            logger.error(f"Concentration risk check failed: {e}")
            return {'risk_level': RiskLevel.MEDIUM}

    def _assess_correlation_risk(self, signal: TradingSignal, current_positions: Dict[str, Position],
                                market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess correlation risk with existing positions"""
        try:
            high_correlation_count = 0
            correlations = []

            symbol_sector = self._get_symbol_sector(signal.symbol)

            for position in current_positions.values():
                position_sector = self._get_symbol_sector(position.symbol)

                # Simplified correlation model based on sector
                if symbol_sector == position_sector:
                    correlation = 0.7  # High intra-sector correlation
                    if correlation > 0.6:
                        high_correlation_count += 1
                else:
                    correlation = 0.2  # Low inter-sector correlation

                correlations.append({
                    'symbol': position.symbol,
                    'correlation': correlation
                })

            return {
                'high_correlation_count': high_correlation_count,
                'correlations': correlations,
                'average_correlation': np.mean([c['correlation'] for c in correlations]) if correlations else 0.0
            }

        except Exception as e:
            logger.error(f"Correlation risk assessment failed: {e}")
            return {'high_correlation_count': 0, 'correlations': [], 'average_correlation': 0.0}

    def _calculate_volatility_adjustment(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Calculate position size adjustment based on volatility"""
        try:
            symbol_data = market_data.get(symbol, {})
            volatility = symbol_data.get('volatility', 0.02)

            # Target volatility (2% daily)
            target_volatility = 0.02

            # Adjustment factor (inverse relationship)
            adjustment = min(1.5, target_volatility / max(volatility, 0.005))

            return max(0.3, adjustment)  # Cap between 30% and 150%

        except Exception as e:
            logger.error(f"Volatility adjustment calculation failed: {e}")
            return 1.0

    def _calculate_overall_risk_score(self, signal: TradingSignal, current_positions: Dict[str, Position],
                                     portfolio_value: float, market_data: Dict[str, Any]) -> float:
        """Calculate overall risk score for the trade"""
        try:
            risk_score = 0.0

            # Position size risk (0-0.3)
            size_risk = min(0.3, signal.position_size / self.max_position_size)
            risk_score += size_risk

            # Volatility risk (0-0.3)
            symbol_data = market_data.get(signal.symbol, {})
            volatility = symbol_data.get('volatility', 0.02)
            vol_risk = min(0.3, volatility / 0.05)  # 5% volatility = max risk
            risk_score += vol_risk

            # Concentration risk (0-0.2)
            concentration = self._check_concentration_risk(signal, current_positions, portfolio_value)
            if concentration['risk_level'] == RiskLevel.HIGH:
                risk_score += 0.2
            elif concentration['risk_level'] == RiskLevel.MEDIUM:
                risk_score += 0.1

            # Correlation risk (0-0.2)
            correlation = self._assess_correlation_risk(signal, current_positions, market_data)
            corr_risk = min(0.2, correlation['high_correlation_count'] * 0.05)
            risk_score += corr_risk

            return min(1.0, risk_score)

        except Exception as e:
            logger.error(f"Overall risk score calculation failed: {e}")
            return 0.5

    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for symbol (simplified mapping)"""
        # Simplified sector mapping
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'CRM', 'ORCL']
        finance_symbols = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 'MMC']
        healthcare_symbols = ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR', 'BMY', 'AMGN', 'GILD', 'CVS']

        if symbol in tech_symbols:
            return 'Technology'
        elif symbol in finance_symbols:
            return 'Financials'
        elif symbol in healthcare_symbols:
            return 'Healthcare'
        elif '.PA' in symbol:
            return 'Europe'
        elif '.L' in symbol:
            return 'UK'
        else:
            return 'Other'

    async def monitor_portfolio_risk(self, current_positions: Dict[str, Position],
                                   portfolio_value: float, market_data: Dict[str, Any]) -> List[RiskAlert]:
        """Real-time portfolio risk monitoring"""
        alerts = []

        try:
            # 1. Check portfolio VaR
            portfolio_returns = self._calculate_portfolio_returns(current_positions, market_data)
            if len(portfolio_returns) > 0:
                daily_var = self.var_calculator.calculate_historical_var(portfolio_returns, 0.95, portfolio_value)
                var_ratio = daily_var / portfolio_value

                if var_ratio > self.max_portfolio_risk:
                    alerts.append(RiskAlert(
                        alert_type=AlertType.VAR_BREACH,
                        risk_level=RiskLevel.HIGH,
                        message=f"Portfolio VaR ({var_ratio:.1%}) exceeds limit ({self.max_portfolio_risk:.1%})",
                        timestamp=datetime.now(),
                        affected_symbols=list(current_positions.keys()),
                        recommended_action="Reduce position sizes or hedge portfolio",
                        metadata={'var_ratio': var_ratio, 'var_amount': daily_var}
                    ))

            # 2. Check individual position sizes
            for symbol, position in current_positions.items():
                if abs(position.position_size) > self.max_position_size:
                    alerts.append(RiskAlert(
                        alert_type=AlertType.POSITION_SIZE,
                        risk_level=RiskLevel.HIGH,
                        message=f"{symbol} position size ({abs(position.position_size):.1%}) exceeds limit",
                        timestamp=datetime.now(),
                        affected_symbols=[symbol],
                        recommended_action=f"Reduce {symbol} position size",
                        metadata={'current_size': position.position_size, 'limit': self.max_position_size}
                    ))

            # 3. Check sector concentration
            sector_exposures = self._calculate_sector_exposures(current_positions)
            for sector, exposure in sector_exposures.items():
                if exposure > self.max_sector_concentration:
                    affected_symbols = [s for s, p in current_positions.items() if self._get_symbol_sector(s) == sector]
                    alerts.append(RiskAlert(
                        alert_type=AlertType.PORTFOLIO_CONCENTRATION,
                        risk_level=RiskLevel.HIGH,
                        message=f"{sector} sector exposure ({exposure:.1%}) exceeds limit",
                        timestamp=datetime.now(),
                        affected_symbols=affected_symbols,
                        recommended_action=f"Reduce exposure to {sector} sector",
                        metadata={'current_exposure': exposure, 'limit': self.max_sector_concentration}
                    ))

            self.active_alerts = alerts
            return alerts

        except Exception as e:
            logger.error(f"Portfolio risk monitoring failed: {e}")
            return []

    def _calculate_portfolio_returns(self, current_positions: Dict[str, Position],
                                   market_data: Dict[str, Any]) -> pd.Series:
        """Calculate portfolio returns for VaR calculation"""
        try:
            # Simplified return calculation
            returns = []
            for position in current_positions.values():
                symbol_data = market_data.get(position.symbol, {})
                symbol_returns = symbol_data.get('returns', pd.Series())
                if len(symbol_returns) > 0:
                    weighted_returns = symbol_returns * abs(position.position_size)
                    returns.append(weighted_returns)

            if returns:
                portfolio_returns = pd.concat(returns, axis=1).sum(axis=1)
                return portfolio_returns.dropna()
            else:
                return pd.Series()

        except Exception as e:
            logger.error(f"Portfolio returns calculation failed: {e}")
            return pd.Series()

    def _calculate_sector_exposures(self, current_positions: Dict[str, Position]) -> Dict[str, float]:
        """Calculate current sector exposures"""
        sector_exposures = {}

        try:
            for position in current_positions.values():
                sector = self._get_symbol_sector(position.symbol)
                if sector not in sector_exposures:
                    sector_exposures[sector] = 0.0
                sector_exposures[sector] += abs(position.position_size)

            return sector_exposures

        except Exception as e:
            logger.error(f"Sector exposure calculation failed: {e}")
            return {}

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk management summary"""
        return {
            'risk_limits': {
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_position_size': self.max_position_size,
                'max_sector_concentration': self.max_sector_concentration,
                'max_drawdown_threshold': self.max_drawdown_threshold
            },
            'active_alerts_count': len(self.active_alerts),
            'monitoring_enabled': True,
            'last_update': datetime.now().isoformat()
        }
        try:
            if len(returns) < 10:
                return portfolio_value * 0.05
            
            # Sort returns and find percentile
            sorted_returns = returns.sort_values()
            percentile_index = int(len(sorted_returns) * (1 - confidence_level))
            
            var_return = sorted_returns.iloc[percentile_index]
            var = portfolio_value * abs(var_return)
            
            return var
            
        except Exception as e:
            logger.error(f"Historical VaR calculation failed: {e}")
            return portfolio_value * 0.05
    
    def calculate_monte_carlo_var(self, returns: pd.Series, confidence_level: float = 0.95,
                                 portfolio_value: float = 1.0, n_simulations: int = 10000) -> float:
        """Calculate Monte Carlo VaR"""
        try:
            if len(returns) < 10:
                return portfolio_value * 0.05
            
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Generate random scenarios
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            
            # Calculate VaR
            var_return = np.percentile(simulated_returns, (1 - confidence_level) * 100)
            var = portfolio_value * abs(var_return)
            
            return var
            
        except Exception as e:
            logger.error(f"Monte Carlo VaR calculation failed: {e}")
            return portfolio_value * 0.05
    
    def calculate_expected_shortfall(self, returns: pd.Series, confidence_level: float = 0.95,
                                   portfolio_value: float = 1.0) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            if len(returns) < 10:
                return portfolio_value * 0.07
            
            sorted_returns = returns.sort_values()
            cutoff_index = int(len(sorted_returns) * (1 - confidence_level))
            
            # Expected value of losses beyond VaR
            tail_returns = sorted_returns.iloc[:cutoff_index]
            expected_shortfall = portfolio_value * abs(tail_returns.mean()) if len(tail_returns) > 0 else portfolio_value * 0.07
            
            return expected_shortfall
            
        except Exception as e:
            logger.error(f"Expected Shortfall calculation failed: {e}")
            return portfolio_value * 0.07

class PortfolioOptimizer:
    """Portfolio optimization for risk management"""
    
    def __init__(self, max_weight: float = 0.3, min_weight: float = 0.01):
        self.max_weight = max_weight
        self.min_weight = min_weight
    
    def calculate_portfolio_variance(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calculate portfolio variance"""
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    def calculate_portfolio_return(self, weights: np.ndarray, expected_returns: np.ndarray) -> float:
        """Calculate expected portfolio return"""
        return np.dot(weights, expected_returns)
    
    def optimize_portfolio(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                          current_weights: np.ndarray = None) -> Dict[str, Any]:
        """Optimize portfolio using mean-variance optimization"""
        try:
            from scipy.optimize import minimize
            
            n_assets = len(expected_returns)
            
            # Initial weights (equal weight or current weights)
            if current_weights is None:
                x0 = np.ones(n_assets) / n_assets
            else:
                x0 = current_weights
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
            ]
            
            # Bounds
            bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
            
            # Objective function (minimize variance)
            def objective(weights):
                return self.calculate_portfolio_variance(weights, cov_matrix)
            
            # Optimization
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = result.x
                optimal_variance = result.fun
                optimal_return = self.calculate_portfolio_return(optimal_weights, expected_returns)
                optimal_volatility = np.sqrt(optimal_variance)
                
                # Calculate Sharpe ratio (assuming 0 risk-free rate)
                sharpe_ratio = optimal_return / optimal_volatility if optimal_volatility > 0 else 0
                
                return {
                    'success': True,
                    'optimal_weights': optimal_weights,
                    'expected_return': optimal_return,
                    'volatility': optimal_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'optimization_result': result
                }
            else:
                return {'success': False, 'error': 'Optimization failed'}
                
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return {'success': False, 'error': str(e)}

class DrawdownAnalyzer:
    """Maximum Drawdown and drawdown analysis"""
    
    @staticmethod
    def calculate_drawdowns(equity_curve: pd.Series) -> pd.DataFrame:
        """Calculate drawdown series"""
        try:
            # Calculate running maximum
            running_max = equity_curve.expanding().max()
            
            # Calculate drawdown
            drawdown = (equity_curve - running_max) / running_max
            
            # Calculate drawdown duration
            drawdown_duration = pd.Series(index=equity_curve.index, dtype=int)
            duration = 0
            
            for i, dd in enumerate(drawdown):
                if dd < 0:
                    duration += 1
                else:
                    duration = 0
                drawdown_duration.iloc[i] = duration
            
            return pd.DataFrame({
                'equity': equity_curve,
                'running_max': running_max,
                'drawdown': drawdown,
                'drawdown_duration': drawdown_duration
            })
            
        except Exception as e:
            logger.error(f"Drawdown calculation failed: {e}")
            return pd.DataFrame()
    
    @classmethod
    def calculate_maximum_drawdown(cls, equity_curve: pd.Series) -> Dict[str, Any]:
        """Calculate maximum drawdown statistics"""
        try:
            drawdown_df = cls.calculate_drawdowns(equity_curve)
            
            if drawdown_df.empty:
                return {'max_drawdown': 0, 'max_drawdown_duration': 0}
            
            max_drawdown = drawdown_df['drawdown'].min()
            max_drawdown_duration = drawdown_df['drawdown_duration'].max()
            
            # Find the period of maximum drawdown
            max_dd_index = drawdown_df['drawdown'].idxmin()
            max_dd_start = None
            
            # Find start of maximum drawdown period
            for i in range(drawdown_df.index.get_loc(max_dd_index), -1, -1):
                if drawdown_df.iloc[i]['drawdown'] == 0:
                    max_dd_start = drawdown_df.index[i + 1] if i + 1 < len(drawdown_df) else drawdown_df.index[0]
                    break
            
            if max_dd_start is None:
                max_dd_start = drawdown_df.index[0]
            
            return {
                'max_drawdown': abs(max_drawdown),
                'max_drawdown_duration': max_drawdown_duration,
                'max_drawdown_start': max_dd_start,
                'max_drawdown_end': max_dd_index,
                'current_drawdown': abs(drawdown_df['drawdown'].iloc[-1]) if not drawdown_df.empty else 0
            }
            
        except Exception as e:
            logger.error(f"Maximum drawdown calculation failed: {e}")
            return {'max_drawdown': 0, 'max_drawdown_duration': 0, 'current_drawdown': 0}

class CorrelationAnalyzer:
    """Portfolio correlation and concentration risk analysis"""
    
    def __init__(self, lookback_period: int = 60):
        self.lookback_period = lookback_period
    
    def calculate_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix for portfolio assets"""
        try:
            if returns_df.empty or len(returns_df) < 10:
                return pd.DataFrame()
            
            correlation_matrix = returns_df.corr()
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Correlation matrix calculation failed: {e}")
            return pd.DataFrame()
    
    def calculate_concentration_risk(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """Calculate portfolio concentration risk metrics"""
        try:
            if not weights:
                return {'herfindahl_index': 0, 'effective_number_positions': 0}
            
            weight_values = np.array(list(weights.values()))
            
            # Herfindahl-Hirschman Index
            herfindahl_index = np.sum(weight_values ** 2)
            
            # Effective number of positions
            effective_positions = 1 / herfindahl_index if herfindahl_index > 0 else 0
            
            # Concentration ratio (top 3 positions)
            sorted_weights = sorted(weight_values, reverse=True)
            top3_concentration = sum(sorted_weights[:3]) if len(sorted_weights) >= 3 else sum(sorted_weights)
            
            return {
                'herfindahl_index': herfindahl_index,
                'effective_number_positions': effective_positions,
                'top3_concentration_ratio': top3_concentration,
                'max_position_weight': max(weight_values) if len(weight_values) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Concentration risk calculation failed: {e}")
            return {'herfindahl_index': 0, 'effective_number_positions': 0}
    
    def calculate_correlation_risk(self, correlation_matrix: pd.DataFrame, 
                                 weights: Dict[str, float]) -> float:
        """Calculate overall portfolio correlation risk"""
        try:
            if correlation_matrix.empty or not weights:
                return 0.0
            
            # Average correlation weighted by position sizes
            total_correlation = 0
            total_weight = 0
            
            symbols = list(weights.keys())
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i != j and symbol1 in correlation_matrix.columns and symbol2 in correlation_matrix.index:
                        corr = correlation_matrix.loc[symbol1, symbol2]
                        weight = weights[symbol1] * weights[symbol2]
                        total_correlation += abs(corr) * weight
                        total_weight += weight
            
            avg_correlation = total_correlation / total_weight if total_weight > 0 else 0
            
            return avg_correlation
            
        except Exception as e:
            logger.error(f"Correlation risk calculation failed: {e}")
            return 0.0

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Risk limits - handle both dict and object config
        if isinstance(self.config, dict):
            self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.15)  # 15% max daily VaR
            self.max_position_size = self.config.get('max_position_size', 0.20)   # 20% max per position
            self.max_sector_concentration = self.config.get('max_sector_concentration', 0.40)  # 40% max per sector
            self.max_drawdown = self.config.get('max_drawdown', 0.20)  # 20% max drawdown
            self.max_correlation = self.config.get('max_correlation', 0.70)  # 70% max average correlation
        else:
            self.max_portfolio_risk = getattr(self.config, 'max_portfolio_risk', 0.15)  # 15% max daily VaR
            self.max_position_size = getattr(self.config, 'max_position_size', 0.20)   # 20% max per position
            self.max_sector_concentration = getattr(self.config, 'max_sector_concentration', 0.40)  # 40% max per sector
            self.max_drawdown = getattr(self.config, 'max_drawdown', 0.20)  # 20% max drawdown
            self.max_correlation = getattr(self.config, 'max_correlation', 0.70)  # 70% max average correlation
        
        # Risk calculation components
        self.var_calculator = VaRCalculator()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.drawdown_analyzer = DrawdownAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        
        # Risk monitoring
        self.risk_alerts = []
        self.risk_history = []
        self.portfolio_performance = []
        
        # Dynamic parameters
        self.current_risk_level = RiskLevel.MEDIUM
        self.risk_multiplier = 1.0  # Multiplier for position sizing based on current risk
    
    async def evaluate_signal_risk(self, signal: TradingSignal, current_positions: Dict[str, Position],
                                 portfolio_value: float, market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate risk of executing a trading signal"""
        try:
            risk_evaluation = {
                'approved': False,
                'risk_score': 1.0,  # 0 = low risk, 1 = high risk
                'recommended_position_size': 0.0,
                'warnings': [],
                'risk_factors': {}
            }
            
            # 1. Position size risk
            if signal.position_size:
                position_weight = (signal.position_size * signal.price) / portfolio_value
                risk_evaluation['risk_factors']['position_weight'] = position_weight
                
                if position_weight > self.max_position_size:
                    risk_evaluation['warnings'].append(f"Position size ({position_weight:.2%}) exceeds maximum ({self.max_position_size:.2%})")
                    risk_evaluation['risk_score'] += 0.3
            
            # 2. Portfolio concentration risk
            current_weights = self._calculate_portfolio_weights(current_positions, portfolio_value)
            
            if signal.symbol in current_weights:
                new_weight = current_weights[signal.symbol] + (signal.position_size * signal.price) / portfolio_value
            else:
                new_weight = (signal.position_size * signal.price) / portfolio_value if signal.position_size else 0
            
            concentration_metrics = self.correlation_analyzer.calculate_concentration_risk(current_weights)
            risk_evaluation['risk_factors']['concentration'] = concentration_metrics
            
            if new_weight > self.max_position_size:
                risk_evaluation['warnings'].append(f"New position weight would exceed maximum")
                risk_evaluation['risk_score'] += 0.2
            
            # 3. Correlation risk
            if market_data and 'correlation_data' in market_data:
                correlation_matrix = market_data['correlation_data']
                correlation_risk = self.correlation_analyzer.calculate_correlation_risk(
                    correlation_matrix, current_weights
                )
                risk_evaluation['risk_factors']['correlation_risk'] = correlation_risk
                
                if correlation_risk > self.max_correlation:
                    risk_evaluation['warnings'].append(f"Portfolio correlation too high ({correlation_risk:.2%})")
                    risk_evaluation['risk_score'] += 0.2
            
            # 4. Market volatility risk
            if market_data and 'volatility' in market_data:
                volatility = market_data['volatility']
                risk_evaluation['risk_factors']['market_volatility'] = volatility
                
                # Adjust position size based on volatility
                volatility_multiplier = min(1.0, 0.2 / volatility) if volatility > 0 else 1.0
                risk_evaluation['volatility_multiplier'] = volatility_multiplier
                
                if volatility > 0.4:  # High volatility
                    risk_evaluation['warnings'].append(f"High market volatility ({volatility:.2%})")
                    risk_evaluation['risk_score'] += 0.1
            
            # 5. Signal strength and confidence
            signal_confidence = signal.strength
            risk_evaluation['risk_factors']['signal_confidence'] = signal_confidence
            
            if signal_confidence < 0.5:
                risk_evaluation['warnings'].append(f"Low signal confidence ({signal_confidence:.2%})")
                risk_evaluation['risk_score'] += 0.1
            
            # Calculate recommended position size
            base_position_size = signal.position_size or 0
            risk_adjusted_size = base_position_size * (1 - risk_evaluation['risk_score'] * 0.5)
            risk_adjusted_size *= self.risk_multiplier  # Apply current risk regime multiplier
            
            # Apply volatility adjustment if available
            if 'volatility_multiplier' in risk_evaluation:
                risk_adjusted_size *= risk_evaluation['volatility_multiplier']
            
            risk_evaluation['recommended_position_size'] = max(0, risk_adjusted_size)
            
            # Final approval decision
            if (risk_evaluation['risk_score'] < 0.7 and 
                risk_evaluation['recommended_position_size'] > 0 and
                len([w for w in risk_evaluation['warnings'] if 'exceed' in w]) == 0):
                
                risk_evaluation['approved'] = True
            
            return risk_evaluation
            
        except Exception as e:
            logger.error(f"Signal risk evaluation failed: {e}")
            return {
                'approved': False,
                'risk_score': 1.0,
                'recommended_position_size': 0.0,
                'warnings': [f'Risk evaluation error: {str(e)}'],
                'risk_factors': {}
            }
    
    async def calculate_portfolio_risk(self, positions: Dict[str, Position], 
                                     portfolio_value: float, 
                                     market_data: Dict[str, Any] = None) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            # Portfolio returns for VaR calculation
            if market_data and 'portfolio_returns' in market_data:
                portfolio_returns = market_data['portfolio_returns']
            else:
                # Generate synthetic returns if not available
                portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns
            
            # Calculate VaR
            var_1d_95 = self.var_calculator.calculate_historical_var(
                portfolio_returns, confidence_level=0.95, portfolio_value=portfolio_value
            )
            var_5d_95 = self.var_calculator.calculate_historical_var(
                portfolio_returns, confidence_level=0.99, portfolio_value=portfolio_value
            )
            
            # Expected Shortfall
            expected_shortfall = self.var_calculator.calculate_expected_shortfall(
                portfolio_returns, confidence_level=0.95, portfolio_value=portfolio_value
            )
            
            # Drawdown metrics
            if market_data and 'equity_curve' in market_data:
                equity_curve = market_data['equity_curve']
                drawdown_metrics = self.drawdown_analyzer.calculate_maximum_drawdown(equity_curve)
            else:
                drawdown_metrics = {'max_drawdown': 0.05, 'current_drawdown': 0.02}
            
            # Portfolio weights
            current_weights = self._calculate_portfolio_weights(positions, portfolio_value)
            
            # Concentration risk
            concentration_metrics = self.correlation_analyzer.calculate_concentration_risk(current_weights)
            
            # Correlation risk
            correlation_risk = 0.0
            if market_data and 'correlation_matrix' in market_data:
                correlation_risk = self.correlation_analyzer.calculate_correlation_risk(
                    market_data['correlation_matrix'], current_weights
                )
            
            # Performance ratios
            if len(portfolio_returns) > 30:
                annual_return = portfolio_returns.mean() * 252
                annual_volatility = portfolio_returns.std() * np.sqrt(252)
                
                sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
                
                # Sortino ratio (downside deviation)
                negative_returns = portfolio_returns[portfolio_returns < 0]
                downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else annual_volatility
                sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
            else:
                sharpe_ratio = 0.0
                sortino_ratio = 0.0
            
            # Beta calculation (vs market if available)
            beta = market_data.get('portfolio_beta', 1.0) if market_data else 1.0
            
            # Total exposure
            total_exposure = sum(abs(pos.size * pos.current_price) for pos in positions.values())
            
            # Liquidity risk (simplified)
            liquidity_risk = market_data.get('liquidity_risk', 0.1) if market_data else 0.1
            
            risk_metrics = RiskMetrics(
                portfolio_value=portfolio_value,
                total_exposure=total_exposure,
                var_1d=var_1d_95,
                var_5d=var_5d_95,
                expected_shortfall=expected_shortfall,
                maximum_drawdown=drawdown_metrics.get('max_drawdown', 0),
                current_drawdown=drawdown_metrics.get('current_drawdown', 0),
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                beta=beta,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_metrics.get('herfindahl_index', 0),
                liquidity_risk=liquidity_risk,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.risk_history.append(risk_metrics)
            
            # Check for risk alerts
            await self._check_risk_alerts(risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Portfolio risk calculation failed: {e}")
            return RiskMetrics(
                portfolio_value=portfolio_value,
                total_exposure=0,
                var_1d=portfolio_value * 0.05,
                var_5d=portfolio_value * 0.10,
                expected_shortfall=portfolio_value * 0.07,
                maximum_drawdown=0.05,
                current_drawdown=0.02,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                beta=1.0,
                correlation_risk=0.5,
                concentration_risk=0.3,
                liquidity_risk=0.1,
                timestamp=datetime.now()
            )
    
    def _calculate_portfolio_weights(self, positions: Dict[str, Position], 
                                   portfolio_value: float) -> Dict[str, float]:
        """Calculate current portfolio weights"""
        weights = {}
        
        for symbol, position in positions.items():
            position_value = abs(position.size * position.current_price)
            weight = position_value / portfolio_value if portfolio_value > 0 else 0
            weights[symbol] = weight
        
        return weights
    
    async def _check_risk_alerts(self, risk_metrics: RiskMetrics):
        """Check for risk limit breaches and generate alerts"""
        alerts = []
        
        # VaR breach check
        var_ratio = risk_metrics.var_1d / risk_metrics.portfolio_value
        if var_ratio > self.max_portfolio_risk:
            alert = RiskAlert(
                alert_type=AlertType.VAR_BREACH,
                risk_level=RiskLevel.HIGH if var_ratio > self.max_portfolio_risk * 1.5 else RiskLevel.MEDIUM,
                message=f"Daily VaR ({var_ratio:.2%}) exceeds limit ({self.max_portfolio_risk:.2%})",
                timestamp=datetime.now(),
                affected_symbols=['PORTFOLIO'],
                recommended_action="Reduce position sizes or hedge portfolio",
                metadata={'var_ratio': var_ratio, 'limit': self.max_portfolio_risk}
            )
            alerts.append(alert)
        
        # Drawdown check
        if risk_metrics.current_drawdown > self.max_drawdown:
            alert = RiskAlert(
                alert_type=AlertType.DRAWDOWN,
                risk_level=RiskLevel.HIGH if risk_metrics.current_drawdown > self.max_drawdown * 1.2 else RiskLevel.MEDIUM,
                message=f"Current drawdown ({risk_metrics.current_drawdown:.2%}) exceeds limit ({self.max_drawdown:.2%})",
                timestamp=datetime.now(),
                affected_symbols=['PORTFOLIO'],
                recommended_action="Consider reducing exposure or implementing stop losses",
                metadata={'current_drawdown': risk_metrics.current_drawdown, 'limit': self.max_drawdown}
            )
            alerts.append(alert)
        
        # Concentration risk check
        if risk_metrics.concentration_risk > 0.5:  # High concentration
            alert = RiskAlert(
                alert_type=AlertType.PORTFOLIO_CONCENTRATION,
                risk_level=RiskLevel.MEDIUM,
                message=f"High portfolio concentration (HHI: {risk_metrics.concentration_risk:.3f})",
                timestamp=datetime.now(),
                affected_symbols=['PORTFOLIO'],
                recommended_action="Diversify portfolio across more positions",
                metadata={'concentration_risk': risk_metrics.concentration_risk}
            )
            alerts.append(alert)
        
        # Correlation risk check
        if risk_metrics.correlation_risk > self.max_correlation:
            alert = RiskAlert(
                alert_type=AlertType.CORRELATION,
                risk_level=RiskLevel.MEDIUM,
                message=f"High portfolio correlation ({risk_metrics.correlation_risk:.2%}) exceeds limit ({self.max_correlation:.2%})",
                timestamp=datetime.now(),
                affected_symbols=['PORTFOLIO'],
                recommended_action="Reduce correlated positions or add uncorrelated assets",
                metadata={'correlation_risk': risk_metrics.correlation_risk, 'limit': self.max_correlation}
            )
            alerts.append(alert)
        
        # Add new alerts to the list
        self.risk_alerts.extend(alerts)
        
        # Log critical alerts
        for alert in alerts:
            if alert.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                logger.warning(f"Risk Alert: {alert.message}")
    
    def adjust_position_sizes(self, signals: List[TradingSignal], 
                            current_risk_metrics: RiskMetrics) -> List[TradingSignal]:
        """Dynamically adjust position sizes based on current risk"""
        adjusted_signals = []
        
        # Determine risk adjustment factor
        risk_factors = [
            current_risk_metrics.current_drawdown / self.max_drawdown,
            current_risk_metrics.var_1d / (current_risk_metrics.portfolio_value * self.max_portfolio_risk),
            current_risk_metrics.correlation_risk / self.max_correlation,
            current_risk_metrics.concentration_risk / 0.5  # Target concentration
        ]
        
        avg_risk_factor = np.mean(risk_factors)
        
        # Risk adjustment multiplier (reduce size when risk is high)
        if avg_risk_factor > 1.0:
            risk_multiplier = 1.0 / avg_risk_factor
        else:
            risk_multiplier = min(1.2, 1.0 + (1.0 - avg_risk_factor) * 0.5)  # Increase up to 20% when risk is low
        
        self.risk_multiplier = risk_multiplier
        
        for signal in signals:
            adjusted_signal = signal
            
            if signal.position_size:
                # Apply risk adjustment
                adjusted_size = signal.position_size * risk_multiplier
                
                # Additional adjustments based on signal characteristics
                if signal.strength < 0.6:  # Low confidence signals get smaller size
                    adjusted_size *= 0.7
                
                # Ensure minimum position size
                min_size = current_risk_metrics.portfolio_value * 0.005 / signal.price  # 0.5% minimum
                adjusted_size = max(adjusted_size, min_size)
                
                # Update signal
                adjusted_signal.position_size = adjusted_size
                
                # Add risk adjustment metadata
                if not adjusted_signal.metadata:
                    adjusted_signal.metadata = {}
                
                adjusted_signal.metadata.update({
                    'risk_multiplier': risk_multiplier,
                    'original_size': signal.position_size,
                    'risk_adjustment_factors': risk_factors
                })
            
            adjusted_signals.append(adjusted_signal)
        
        logger.info(f"Applied risk multiplier of {risk_multiplier:.3f} to position sizes")
        
        return adjusted_signals
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        if not self.risk_history:
            return {'error': 'No risk data available'}
        
        latest_metrics = self.risk_history[-1]
        
        # Recent alerts (last 24 hours)
        recent_alerts = [
            alert for alert in self.risk_alerts 
            if alert.timestamp > datetime.now() - timedelta(days=1)
        ]
        
        # Alert counts by level
        alert_counts = {
            level.value: len([a for a in recent_alerts if a.risk_level == level])
            for level in RiskLevel
        }
        
        # Risk level assessment
        current_risk_level = self._assess_current_risk_level(latest_metrics)
        
        risk_summary = {
            'current_risk_level': current_risk_level.value,
            'portfolio_value': latest_metrics.portfolio_value,
            'total_exposure': latest_metrics.total_exposure,
            'key_metrics': {
                'var_1d_pct': (latest_metrics.var_1d / latest_metrics.portfolio_value) * 100,
                'current_drawdown_pct': latest_metrics.current_drawdown * 100,
                'sharpe_ratio': latest_metrics.sharpe_ratio,
                'correlation_risk': latest_metrics.correlation_risk,
                'concentration_risk': latest_metrics.concentration_risk
            },
            'risk_limits': {
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_position_size': self.max_position_size,
                'max_drawdown': self.max_drawdown,
                'max_correlation': self.max_correlation
            },
            'recent_alerts': {
                'count_by_level': alert_counts,
                'total_alerts': len(recent_alerts),
                'latest_alerts': [
                    {
                        'type': alert.alert_type.value,
                        'level': alert.risk_level.value,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in recent_alerts[-5:]  # Last 5 alerts
                ]
            },
            'risk_multiplier': self.risk_multiplier,
            'timestamp': latest_metrics.timestamp.isoformat()
        }
        
        return risk_summary
    
    def _assess_current_risk_level(self, risk_metrics: RiskMetrics) -> RiskLevel:
        """Assess current overall risk level"""
        risk_factors = []
        
        # VaR risk
        var_ratio = risk_metrics.var_1d / risk_metrics.portfolio_value
        if var_ratio > self.max_portfolio_risk * 1.5:
            risk_factors.append(RiskLevel.CRITICAL)
        elif var_ratio > self.max_portfolio_risk:
            risk_factors.append(RiskLevel.HIGH)
        elif var_ratio > self.max_portfolio_risk * 0.5:
            risk_factors.append(RiskLevel.MEDIUM)
        else:
            risk_factors.append(RiskLevel.LOW)
        
        # Drawdown risk
        if risk_metrics.current_drawdown > self.max_drawdown * 1.2:
            risk_factors.append(RiskLevel.CRITICAL)
        elif risk_metrics.current_drawdown > self.max_drawdown:
            risk_factors.append(RiskLevel.HIGH)
        elif risk_metrics.current_drawdown > self.max_drawdown * 0.5:
            risk_factors.append(RiskLevel.MEDIUM)
        else:
            risk_factors.append(RiskLevel.LOW)
        
        # Correlation risk
        if risk_metrics.correlation_risk > self.max_correlation:
            risk_factors.append(RiskLevel.HIGH)
        elif risk_metrics.correlation_risk > self.max_correlation * 0.7:
            risk_factors.append(RiskLevel.MEDIUM)
        else:
            risk_factors.append(RiskLevel.LOW)
        
        # Overall assessment (take maximum risk level)
        risk_levels = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        max_risk_index = max(risk_levels.index(rf) for rf in risk_factors)
        
        return risk_levels[max_risk_index]
    
    def save_risk_state(self, filepath: str) -> bool:
        """Save risk management state"""
        try:
            risk_state = {
                'config': self.config,
                'risk_alerts': [asdict(alert) for alert in self.risk_alerts[-100:]],  # Last 100 alerts
                'risk_history': [asdict(metrics) for metrics in self.risk_history[-100:]],  # Last 100 metrics
                'current_risk_level': self.current_risk_level.value,
                'risk_multiplier': self.risk_multiplier
            }
            
            import joblib
            joblib.dump(risk_state, filepath)
            logger.info(f"Risk state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save risk state: {e}")
            return False
    
    def load_risk_state(self, filepath: str) -> bool:
        """Load risk management state"""
        try:
            import joblib
            risk_state = joblib.load(filepath)
            
            self.config = risk_state.get('config', {})
            self.current_risk_level = RiskLevel(risk_state.get('current_risk_level', 'medium'))
            self.risk_multiplier = risk_state.get('risk_multiplier', 1.0)
            
            # Reconstruct alerts
            alerts_data = risk_state.get('risk_alerts', [])
            self.risk_alerts = []
            for alert_dict in alerts_data:
                alert = RiskAlert(
                    alert_type=AlertType(alert_dict['alert_type']),
                    risk_level=RiskLevel(alert_dict['risk_level']),
                    message=alert_dict['message'],
                    timestamp=datetime.fromisoformat(alert_dict['timestamp']),
                    affected_symbols=alert_dict['affected_symbols'],
                    recommended_action=alert_dict['recommended_action'],
                    metadata=alert_dict['metadata']
                )
                self.risk_alerts.append(alert)
            
            # Reconstruct risk history
            metrics_data = risk_state.get('risk_history', [])
            self.risk_history = []
            for metrics_dict in metrics_data:
                metrics = RiskMetrics(
                    portfolio_value=metrics_dict['portfolio_value'],
                    total_exposure=metrics_dict['total_exposure'],
                    var_1d=metrics_dict['var_1d'],
                    var_5d=metrics_dict['var_5d'],
                    expected_shortfall=metrics_dict['expected_shortfall'],
                    maximum_drawdown=metrics_dict['maximum_drawdown'],
                    current_drawdown=metrics_dict['current_drawdown'],
                    sharpe_ratio=metrics_dict['sharpe_ratio'],
                    sortino_ratio=metrics_dict['sortino_ratio'],
                    beta=metrics_dict['beta'],
                    correlation_risk=metrics_dict['correlation_risk'],
                    concentration_risk=metrics_dict['concentration_risk'],
                    liquidity_risk=metrics_dict['liquidity_risk'],
                    timestamp=datetime.fromisoformat(metrics_dict['timestamp'])
                )
                self.risk_history.append(metrics)
            
            logger.info(f"Risk state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load risk state: {e}")
            return False