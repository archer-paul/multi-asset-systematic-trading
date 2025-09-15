"""
Portfolio Analysis Module for Trading Bot
Provides comprehensive portfolio analytics and performance tracking
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import yfinance as yf

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

@dataclass
class Attribution:
    """Performance attribution data"""
    security_selection: float
    asset_allocation: float
    interaction: float
    total_active_return: float

class PerformanceAnalyzer:
    """Comprehensive portfolio analysis system"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.risk_free_rate = 0.02  # 2% risk-free rate
        self.benchmark_symbol = 'SPY'  # S&P 500 ETF as benchmark

    async def calculate_performance_metrics(self, portfolio_history: pd.DataFrame,
                                          benchmark_history: pd.DataFrame = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            if portfolio_history.empty:
                return self._get_default_metrics()

            portfolio_returns = portfolio_history['value'].pct_change().dropna()
            total_return = (portfolio_history['value'].iloc[-1] / portfolio_history['value'].iloc[0] - 1) * 100
            years = len(portfolio_returns) / 252
            annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
            volatility = portfolio_returns.std() * np.sqrt(252) * 100
            excess_return = annualized_return - self.risk_free_rate * 100
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            max_drawdown = await self._calculate_max_drawdown(portfolio_history['value'])
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            positive_returns = portfolio_returns[portfolio_returns > 0]
            negative_returns = portfolio_returns[portfolio_returns < 0]
            win_rate = len(positive_returns) / len(portfolio_returns) * 100
            avg_win = positive_returns.mean() * 100 if len(positive_returns) > 0 else 0
            avg_loss = negative_returns.mean() * 100 if len(negative_returns) > 0 else 0
            profit_factor = abs(avg_win * len(positive_returns)) / abs(avg_loss * len(negative_returns)) if len(negative_returns) > 0 and avg_loss != 0 else 0

            return PerformanceMetrics(
                total_return=total_return, annualized_return=annualized_return, volatility=volatility,
                sharpe_ratio=sharpe_ratio, max_drawdown=max_drawdown, calmar_ratio=calmar_ratio,
                win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss, profit_factor=profit_factor
            )
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            return self._get_default_metrics()

    async def compare_to_benchmark(self, portfolio_history: pd.DataFrame, benchmark_symbol: str = 'SPY') -> Dict[str, Any]:
        try:
            if portfolio_history.empty:
                return await self._get_mock_benchmark_comparison()

            start_date = portfolio_history.index[0]
            end_date = portfolio_history.index[-1]

            benchmark_data = await asyncio.to_thread(yf.download, benchmark_symbol, start=start_date, end=end_date)

            if benchmark_data.empty:
                return await self._get_mock_benchmark_comparison()

            portfolio_values = portfolio_history.reindex(benchmark_data.index, method='ffill')
            benchmark_values = benchmark_data['Close']
            portfolio_normalized = portfolio_values['value'] / portfolio_values['value'].iloc[0]
            benchmark_normalized = benchmark_values / benchmark_values.iloc[0]
            portfolio_return = (portfolio_normalized.iloc[-1] - 1) * 100
            benchmark_return = (benchmark_normalized.iloc[-1] - 1) * 100
            outperformance = portfolio_return - benchmark_return

            comparison_data = []
            for date in portfolio_normalized.index:
                if date in benchmark_normalized.index:
                    comparison_data.append({
                        'date': date.strftime('%Y-%m-%d'), 'portfolio': portfolio_normalized[date], 'benchmark': benchmark_normalized[date],
                        'portfolio_return': (portfolio_normalized[date] - 1) * 100, 'benchmark_return': (benchmark_normalized[date] - 1) * 100
                    })

            return {
                'data': comparison_data, 'portfolio_total_return': portfolio_return, 'benchmark_total_return': benchmark_return,
                'outperformance': outperformance, 'benchmark_symbol': benchmark_symbol,
                'tracking_error': self._calculate_tracking_error(portfolio_normalized, benchmark_normalized),
                'information_ratio': outperformance / self._calculate_tracking_error(portfolio_normalized, benchmark_normalized) if self._calculate_tracking_error(portfolio_normalized, benchmark_normalized) != 0 else 0
            }
        except Exception as e:
            logger.error(f"Failed to compare to benchmark: {e}")
            return await self._get_mock_benchmark_comparison()

    async def calculate_attribution(self, portfolio_positions: List[Dict], benchmark_weights: Dict[str, float],
                                   portfolio_returns: Dict[str, float], benchmark_returns: Dict[str, float]) -> Attribution:
        # ... (implementation unchanged)
        pass

    async def analyze_sector_performance(self, positions: List[Dict]) -> List[Dict[str, Any]]:
        # ... (implementation unchanged)
        pass

    async def calculate_risk_adjusted_returns(self, portfolio_history: pd.DataFrame, benchmark_history: pd.DataFrame = None) -> Dict[str, float]:
        # ... (implementation unchanged)
        pass

    async def get_portfolio_statistics(self, positions: List[Dict]) -> Dict[str, Any]:
        # ... (implementation unchanged)
        pass

    async def _calculate_max_drawdown(self, values: pd.Series) -> float:
        # ... (implementation unchanged)
        pass

    def _calculate_tracking_error(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        # ... (implementation unchanged)
        pass

    async def _estimate_beta(self, portfolio_returns: pd.Series) -> float:
        # ... (implementation unchanged)
        pass

    def _get_sector(self, symbol: str) -> str:
        # ... (implementation unchanged)
        pass

    def _get_default_metrics(self) -> PerformanceMetrics:
        # ... (implementation unchanged)
        pass

    async def _get_mock_benchmark_comparison(self) -> Dict[str, Any]:
        # ... (implementation unchanged)
        pass

    def _get_mock_portfolio_statistics(self) -> Dict[str, Any]:
        # ... (implementation unchanged)
        pass
