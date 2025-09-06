"""
Advanced Portfolio Management Module for Trading Bot
Comprehensive portfolio tracking, execution, optimization, and performance monitoring
with sophisticated order management, slippage modeling, and transaction cost analysis
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from abc import ABC, abstractmethod
import uuid
import json

from trading.strategy import TradingSignal, Position, PositionType, SignalType
from trading.risk_manager import RiskManager, RiskMetrics

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types for execution"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class ExecutionVenue(Enum):
    """Execution venues/exchanges"""
    SIMULATED = "simulated"
    ALPACA = "alpaca"
    INTERACTIVE_BROKERS = "interactive_brokers"
    BINANCE = "binance"
    COINBASE = "coinbase"

@dataclass
class Order:
    """Order data structure"""
    order_id: str
    symbol: str
    side: str  # buy/sell
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    time_in_force: str  # DAY, GTC, IOC, FOK
    created_time: datetime
    status: OrderStatus
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    venue: ExecutionVenue = ExecutionVenue.SIMULATED
    parent_signal_id: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class Trade:
    """Individual trade/execution record"""
    trade_id: str
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    venue: ExecutionVenue
    market_impact: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = None

@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time"""
    timestamp: datetime
    total_value: float
    cash_balance: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    day_pnl: float
    positions: Dict[str, Position]
    performance_metrics: Dict[str, Any]

class ExecutionEngine(ABC):
    """Abstract base class for execution engines"""
    
    @abstractmethod
    async def submit_order(self, order: Order) -> Dict[str, Any]:
        """Submit order to execution venue"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel pending order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get current order status"""
        pass

class SimulatedExecutionEngine(ExecutionEngine):
    """Simulated execution engine for backtesting and paper trading"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.slippage_model = SlippageModel(self.config.get('slippage', {}))
        self.commission_model = CommissionModel(self.config.get('commission', {}))
        self.market_impact_model = MarketImpactModel(self.config.get('market_impact', {}))
        
        # Market data simulation
        self.market_data = {}
        self.order_book_depth = self.config.get('order_book_depth', 0.001)  # 0.1% spread
        
        # Execution parameters
        self.fill_probability = self.config.get('fill_probability', 0.95)
        self.partial_fill_probability = self.config.get('partial_fill_probability', 0.1)
        self.execution_delay_ms = self.config.get('execution_delay_ms', 100)
    
    async def submit_order(self, order: Order) -> Dict[str, Any]:
        """Simulate order submission"""
        try:
            # Simulate network delay
            await asyncio.sleep(self.execution_delay_ms / 1000.0)
            
            # Get market data for the symbol
            market_price = self._get_market_price(order.symbol)
            if market_price is None:
                return {
                    'success': False,
                    'error': f'No market data available for {order.symbol}',
                    'order_id': order.order_id
                }
            
            # Determine if order should fill
            if np.random.random() > self.fill_probability:
                order.status = OrderStatus.REJECTED
                return {
                    'success': False,
                    'error': 'Order rejected by exchange',
                    'order_id': order.order_id
                }
            
            # Calculate execution price with slippage
            execution_price = self.slippage_model.apply_slippage(
                order, market_price, self._get_market_volatility(order.symbol)
            )
            
            # Calculate market impact
            market_impact = self.market_impact_model.calculate_impact(
                order.quantity, self._get_average_volume(order.symbol), market_price
            )
            
            execution_price += market_impact
            
            # Determine fill quantity (full or partial)
            if order.order_type == OrderType.MARKET or np.random.random() > self.partial_fill_probability:
                # Full fill
                fill_quantity = order.quantity
                order.status = OrderStatus.FILLED
            else:
                # Partial fill
                fill_quantity = order.quantity * np.random.uniform(0.3, 0.9)
                order.status = OrderStatus.PARTIAL_FILLED
            
            # Calculate commission
            commission = self.commission_model.calculate_commission(
                order.symbol, fill_quantity, execution_price
            )
            
            # Update order
            order.filled_quantity += fill_quantity
            order.average_fill_price = execution_price
            order.commission += commission
            
            # Create trade record
            trade = Trade(
                trade_id=str(uuid.uuid4()),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=fill_quantity,
                price=execution_price,
                timestamp=datetime.now(),
                commission=commission,
                venue=ExecutionVenue.SIMULATED,
                market_impact=market_impact,
                slippage=execution_price - market_price,
                metadata={'market_price': market_price}
            )
            
            return {
                'success': True,
                'order_id': order.order_id,
                'trade': trade,
                'execution_price': execution_price,
                'fill_quantity': fill_quantity,
                'commission': commission
            }
            
        except Exception as e:
            logger.error(f"Simulated order execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'order_id': order.order_id
            }
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Simulate order cancellation"""
        # Simulate small delay
        await asyncio.sleep(0.05)
        
        return {
            'success': True,
            'order_id': order_id,
            'status': OrderStatus.CANCELLED.value
        }
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get simulated order status"""
        return {
            'success': True,
            'order_id': order_id,
            'status': 'unknown'  # Would be tracked in real implementation
        }
    
    def _get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price (simulated)"""
        # In a real implementation, this would fetch from market data
        return self.market_data.get(symbol, {}).get('price', 100.0)  # Default price
    
    def _get_market_volatility(self, symbol: str) -> float:
        """Get market volatility for slippage calculation"""
        return self.market_data.get(symbol, {}).get('volatility', 0.02)  # 2% default volatility
    
    def _get_average_volume(self, symbol: str) -> float:
        """Get average trading volume"""
        return self.market_data.get(symbol, {}).get('avg_volume', 1000000)  # Default volume
    
    def update_market_data(self, symbol: str, price: float, volume: float = None, volatility: float = None):
        """Update market data for simulation"""
        if symbol not in self.market_data:
            self.market_data[symbol] = {}
        
        self.market_data[symbol]['price'] = price
        if volume:
            self.market_data[symbol]['avg_volume'] = volume
        if volatility:
            self.market_data[symbol]['volatility'] = volatility

class SlippageModel:
    """Model for calculating trading slippage"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.base_slippage_bps = self.config.get('base_slippage_bps', 2.0)  # 2 basis points
        self.volatility_multiplier = self.config.get('volatility_multiplier', 1.5)
        self.size_impact_factor = self.config.get('size_impact_factor', 0.1)
    
    def apply_slippage(self, order: Order, market_price: float, volatility: float) -> float:
        """Apply slippage to order execution price"""
        try:
            # Base slippage
            base_slippage = market_price * (self.base_slippage_bps / 10000.0)
            
            # Volatility-based slippage
            volatility_slippage = market_price * volatility * self.volatility_multiplier / 100.0
            
            # Order type adjustments
            if order.order_type == OrderType.MARKET:
                slippage_multiplier = 1.0
            elif order.order_type == OrderType.LIMIT:
                slippage_multiplier = 0.3  # Limit orders have less slippage
            else:
                slippage_multiplier = 0.8
            
            # Direction adjustment (buy orders slip up, sell orders slip down)
            direction = 1 if order.side.lower() == 'buy' else -1
            
            total_slippage = (base_slippage + volatility_slippage) * slippage_multiplier * direction
            
            return market_price + total_slippage
            
        except Exception as e:
            logger.error(f"Slippage calculation failed: {e}")
            return market_price  # Return market price if calculation fails

class CommissionModel:
    """Model for calculating trading commissions"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.fixed_commission = self.config.get('fixed_commission', 1.0)  # $1 per trade
        self.rate_per_share = self.config.get('rate_per_share', 0.005)  # $0.005 per share
        self.percentage_rate = self.config.get('percentage_rate', 0.0)  # 0% of trade value
        self.min_commission = self.config.get('min_commission', 1.0)
        self.max_commission = self.config.get('max_commission', 100.0)
    
    def calculate_commission(self, symbol: str, quantity: float, price: float) -> float:
        """Calculate commission for a trade"""
        try:
            trade_value = quantity * price
            
            # Calculate different commission components
            fixed_cost = self.fixed_commission
            share_cost = quantity * self.rate_per_share
            percentage_cost = trade_value * (self.percentage_rate / 100.0)
            
            total_commission = fixed_cost + share_cost + percentage_cost
            
            # Apply min/max limits
            total_commission = max(self.min_commission, min(self.max_commission, total_commission))
            
            return total_commission
            
        except Exception as e:
            logger.error(f"Commission calculation failed: {e}")
            return self.fixed_commission

class MarketImpactModel:
    """Model for calculating market impact of trades"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.impact_coefficient = self.config.get('impact_coefficient', 0.1)
        self.volume_participation_threshold = self.config.get('volume_participation_threshold', 0.05)
    
    def calculate_impact(self, order_quantity: float, average_volume: float, price: float) -> float:
        """Calculate market impact in price terms"""
        try:
            if average_volume <= 0:
                return 0.0
            
            # Participation rate in daily volume
            participation_rate = order_quantity / average_volume
            
            # Market impact is non-linear in participation rate
            if participation_rate < self.volume_participation_threshold:
                impact_bps = self.impact_coefficient * (participation_rate ** 0.5) * 10000
            else:
                # Higher impact for large orders
                impact_bps = self.impact_coefficient * (participation_rate ** 0.8) * 10000
            
            return price * (impact_bps / 10000.0)
            
        except Exception as e:
            logger.error(f"Market impact calculation failed: {e}")
            return 0.0

class PerformanceAnalyzer:
    """Comprehensive portfolio performance analysis"""
    
    def __init__(self):
        self.portfolio_history = []
        self.benchmark_returns = []
    
    def add_snapshot(self, snapshot: PortfolioSnapshot):
        """Add portfolio snapshot for performance tracking"""
        self.portfolio_history.append(snapshot)
    
    def calculate_returns(self) -> pd.Series:
        """Calculate portfolio returns"""
        if len(self.portfolio_history) < 2:
            return pd.Series()
        
        values = [snapshot.total_value for snapshot in self.portfolio_history]
        timestamps = [snapshot.timestamp for snapshot in self.portfolio_history]
        
        value_series = pd.Series(values, index=timestamps)
        returns = value_series.pct_change().dropna()
        
        return returns
    
    def calculate_comprehensive_metrics(self, benchmark_returns: pd.Series = None) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        try:
            returns = self.calculate_returns()
            
            if len(returns) < 10:
                return {'error': 'Insufficient data for analysis'}
            
            # Basic metrics
            total_return = (self.portfolio_history[-1].total_value / self.portfolio_history[0].total_value) - 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            annualized_volatility = returns.std() * np.sqrt(252)
            
            # Risk metrics
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
            
            # Downside metrics
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else annualized_volatility
            sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
            
            # Drawdown analysis
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Win/Loss metrics
            winning_trades = len(returns[returns > 0])
            losing_trades = len(returns[returns < 0])
            win_rate = winning_trades / len(returns) if len(returns) > 0 else 0
            
            avg_win = returns[returns > 0].mean() if winning_trades > 0 else 0
            avg_loss = returns[returns < 0].mean() if losing_trades > 0 else 0
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else float('inf')
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            metrics = {
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'annualized_return': annualized_return,
                'annualized_return_pct': annualized_return * 100,
                'annualized_volatility': annualized_volatility,
                'annualized_volatility_pct': annualized_volatility * 100,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100,
                'win_rate': win_rate,
                'win_rate_pct': win_rate * 100,
                'profit_factor': profit_factor,
                'avg_win_pct': avg_win * 100,
                'avg_loss_pct': avg_loss * 100,
                'total_trades': len(returns),
                'winning_trades': winning_trades,
                'losing_trades': losing_trades
            }
            
            # Benchmark comparison if provided
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                # Align returns with benchmark
                aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
                
                if len(aligned_returns) > 10:
                    # Beta calculation
                    covariance = aligned_returns.cov(aligned_benchmark)
                    benchmark_variance = aligned_benchmark.var()
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
                    
                    # Alpha calculation
                    benchmark_annualized = (1 + aligned_benchmark.mean()) ** 252 - 1
                    alpha = annualized_return - beta * benchmark_annualized
                    
                    # Information ratio
                    excess_returns = aligned_returns - aligned_benchmark
                    tracking_error = excess_returns.std() * np.sqrt(252)
                    information_ratio = excess_returns.mean() * np.sqrt(252) / tracking_error if tracking_error > 0 else 0
                    
                    metrics.update({
                        'beta': beta,
                        'alpha': alpha,
                        'alpha_pct': alpha * 100,
                        'information_ratio': information_ratio,
                        'tracking_error': tracking_error,
                        'tracking_error_pct': tracking_error * 100
                    })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {'error': str(e)}

class AdvancedPortfolioManager:
    """Advanced Portfolio Management System"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Portfolio initialization
        self.initial_capital = self.config.get('initial_capital', 100000.0)
        self.cash_balance = self.initial_capital
        self.positions = {}
        self.pending_orders = {}
        self.order_history = []
        self.trade_history = []
        
        # Execution engine
        execution_type = self.config.get('execution_type', 'simulated')
        if execution_type == 'simulated':
            self.execution_engine = SimulatedExecutionEngine(self.config.get('execution', {}))
        else:
            raise NotImplementedError(f"Execution type {execution_type} not implemented")
        
        # Risk management
        self.risk_manager = RiskManager(self.config.get('risk_management', {}))
        
        # Performance tracking
        self.performance_analyzer = PerformanceAnalyzer()
        self.portfolio_snapshots = []
        
        # Configuration parameters
        self.max_positions = self.config.get('max_positions', 10)
        self.position_sizing_method = self.config.get('position_sizing_method', 'risk_based')
        self.rebalancing_frequency = self.config.get('rebalancing_frequency', 'daily')
        
        # Portfolio optimization
        self.enable_portfolio_optimization = self.config.get('enable_portfolio_optimization', True)
        self.optimization_frequency = self.config.get('optimization_frequency', 'weekly')
        
        # Performance tracking
        self.last_snapshot_time = datetime.now()
        self.snapshot_frequency = timedelta(hours=1)  # Hourly snapshots
    
    async def process_signals(self, signals: List[TradingSignal], market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process trading signals with comprehensive risk management"""
        try:
            logger.info(f"Processing {len(signals)} trading signals")
            
            # Update market data in execution engine
            if market_data:
                for symbol, data in market_data.items():
                    if isinstance(self.execution_engine, SimulatedExecutionEngine):
                        price = data.get('price', data.get('Close', 100.0))
                        volume = data.get('volume', data.get('Volume'))
                        volatility = data.get('volatility', 0.02)
                        
                        self.execution_engine.update_market_data(symbol, price, volume, volatility)
            
            # Calculate current portfolio risk metrics
            portfolio_value = self.get_portfolio_value()
            current_risk_metrics = await self.risk_manager.calculate_portfolio_risk(
                self.positions, portfolio_value, market_data
            )
            
            # Risk evaluation for each signal
            processed_signals = []
            rejected_signals = []
            
            for signal in signals:
                risk_evaluation = await self.risk_manager.evaluate_signal_risk(
                    signal, self.positions, portfolio_value, market_data
                )
                
                if risk_evaluation['approved']:
                    # Update signal with risk-adjusted position size
                    signal.position_size = risk_evaluation['recommended_position_size']
                    processed_signals.append(signal)
                else:
                    rejected_signals.append({
                        'signal': signal,
                        'rejection_reason': risk_evaluation['warnings']
                    })
                    logger.warning(f"Signal rejected for {signal.symbol}: {risk_evaluation['warnings']}")
            
            # Apply portfolio-level risk adjustments
            if processed_signals:
                processed_signals = self.risk_manager.adjust_position_sizes(processed_signals, current_risk_metrics)
            
            # Execute approved signals
            execution_results = await self._execute_signals(processed_signals)
            
            # Update portfolio snapshot
            await self._update_portfolio_snapshot(market_data)
            
            # Generate comprehensive results
            results = {
                'success': True,
                'processed_signals': len(processed_signals),
                'rejected_signals': len(rejected_signals),
                'execution_results': execution_results,
                'portfolio_metrics': current_risk_metrics,
                'portfolio_value': portfolio_value,
                'cash_balance': self.cash_balance,
                'active_positions': len(self.positions),
                'risk_alerts': self.risk_manager.risk_alerts[-5:] if self.risk_manager.risk_alerts else [],
                'timestamp': datetime.now()
            }
            
            if rejected_signals:
                results['rejected_signals'] = [
                    {
                        'symbol': rs['signal'].symbol,
                        'signal_type': rs['signal'].signal_type.value,
                        'rejection_reasons': rs['rejection_reason']
                    }
                    for rs in rejected_signals
                ]
            
            return results
            
        except Exception as e:
            logger.error(f"Signal processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processed_signals': 0,
                'rejected_signals': len(signals),
                'timestamp': datetime.now()
            }
    
    async def _execute_signals(self, signals: List[TradingSignal]) -> List[Dict[str, Any]]:
        """Execute trading signals through execution engine"""
        execution_results = []
        
        for signal in signals:
            try:
                # Create order from signal
                order = self._create_order_from_signal(signal)
                
                # Submit order for execution
                execution_result = await self.execution_engine.submit_order(order)
                
                if execution_result['success']:
                    # Update portfolio state
                    await self._process_successful_execution(order, execution_result)
                    
                    execution_results.append({
                        'symbol': signal.symbol,
                        'signal_type': signal.signal_type.value,
                        'success': True,
                        'execution_price': execution_result['execution_price'],
                        'fill_quantity': execution_result['fill_quantity'],
                        'commission': execution_result['commission'],
                        'order_id': order.order_id
                    })
                else:
                    execution_results.append({
                        'symbol': signal.symbol,
                        'signal_type': signal.signal_type.value,
                        'success': False,
                        'error': execution_result['error'],
                        'order_id': order.order_id
                    })
                    
                # Store order in history
                self.order_history.append(order)
                
            except Exception as e:
                logger.error(f"Execution failed for {signal.symbol}: {e}")
                execution_results.append({
                    'symbol': signal.symbol,
                    'signal_type': signal.signal_type.value,
                    'success': False,
                    'error': str(e)
                })
        
        return execution_results
    
    def _create_order_from_signal(self, signal: TradingSignal) -> Order:
        """Create order from trading signal"""
        # Determine order side
        if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            side = 'buy'
        elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            side = 'sell'
        else:
            raise ValueError(f"Invalid signal type for order creation: {signal.signal_type}")
        
        # Determine order type (default to market for simplicity)
        order_type = OrderType.MARKET
        
        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=signal.symbol,
            side=side,
            order_type=order_type,
            quantity=signal.position_size or 0,
            price=signal.price if order_type == OrderType.LIMIT else None,
            stop_price=signal.stop_loss,
            time_in_force='DAY',
            created_time=datetime.now(),
            status=OrderStatus.PENDING,
            venue=ExecutionVenue.SIMULATED,
            parent_signal_id=str(id(signal)),
            metadata={
                'signal_strength': signal.strength,
                'strategy': signal.strategy,
                'target_price': signal.target_price,
                'signal_metadata': signal.metadata
            }
        )
        
        return order
    
    async def _process_successful_execution(self, order: Order, execution_result: Dict[str, Any]):
        """Process successful order execution"""
        trade = execution_result['trade']
        
        # Update cash balance
        trade_value = trade.quantity * trade.price
        if order.side == 'buy':
            self.cash_balance -= (trade_value + trade.commission)
        else:
            self.cash_balance += (trade_value - trade.commission)
        
        # Update or create position
        if order.side == 'buy':
            await self._update_long_position(order, trade)
        else:
            await self._update_short_position(order, trade)
        
        # Store trade in history
        self.trade_history.append(trade)
        
        logger.info(f"Executed {order.side} order for {order.symbol}: {trade.quantity} shares at ${trade.price:.2f}")
    
    async def _update_long_position(self, order: Order, trade: Trade):
        """Update long position"""
        if order.symbol in self.positions:
            # Add to existing position
            position = self.positions[order.symbol]
            
            # Calculate new average price
            total_value = (position.size * position.entry_price) + (trade.quantity * trade.price)
            total_quantity = position.size + trade.quantity
            
            position.size = total_quantity
            position.entry_price = total_value / total_quantity if total_quantity > 0 else trade.price
            position.current_price = trade.price
            
            # Update stop loss and take profit if specified
            if order.stop_price:
                position.stop_loss = order.stop_price
            if order.metadata and order.metadata.get('target_price'):
                position.take_profit = order.metadata['target_price']
                
        else:
            # Create new position
            position = Position(
                symbol=order.symbol,
                position_type=PositionType.LONG,
                size=trade.quantity,
                entry_price=trade.price,
                current_price=trade.price,
                entry_time=trade.timestamp,
                stop_loss=order.stop_price,
                take_profit=order.metadata.get('target_price') if order.metadata else None,
                metadata={
                    'entry_trade_id': trade.trade_id,
                    'entry_order_id': order.order_id,
                    'initial_signal_strength': order.metadata.get('signal_strength') if order.metadata else None
                }
            )
            
            self.positions[order.symbol] = position
    
    async def _update_short_position(self, order: Order, trade: Trade):
        """Update position for sell order (could be closing long or opening short)"""
        if order.symbol in self.positions:
            position = self.positions[order.symbol]
            
            if position.position_type == PositionType.LONG:
                # Closing long position (full or partial)
                if trade.quantity >= position.size:
                    # Close entire position
                    realized_pnl = (trade.price - position.entry_price) * position.size
                    del self.positions[order.symbol]
                    
                    logger.info(f"Closed long position in {order.symbol}: Realized P&L = ${realized_pnl:.2f}")
                else:
                    # Partial close
                    realized_pnl = (trade.price - position.entry_price) * trade.quantity
                    position.size -= trade.quantity
                    position.current_price = trade.price
                    
                    logger.info(f"Partially closed long position in {order.symbol}: Realized P&L = ${realized_pnl:.2f}")
            else:
                # Adding to short position (not implemented in this basic version)
                logger.warning(f"Short position handling not fully implemented for {order.symbol}")
        else:
            # This would be a short sale (not implemented in this basic version)
            logger.warning(f"Short selling not implemented for {order.symbol}")
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(
            position.size * position.current_price 
            for position in self.positions.values()
        )
        
        return self.cash_balance + positions_value
    
    def get_positions_summary(self) -> Dict[str, Any]:
        """Get summary of current positions"""
        if not self.positions:
            return {
                'total_positions': 0,
                'total_value': 0,
                'unrealized_pnl': 0,
                'positions': []
            }
        
        positions_summary = []
        total_unrealized_pnl = 0
        total_positions_value = 0
        
        for symbol, position in self.positions.items():
            market_value = position.size * position.current_price
            unrealized_pnl = (position.current_price - position.entry_price) * position.size
            unrealized_pnl_pct = (unrealized_pnl / (position.entry_price * position.size)) * 100
            
            total_unrealized_pnl += unrealized_pnl
            total_positions_value += market_value
            
            positions_summary.append({
                'symbol': symbol,
                'position_type': position.position_type.value,
                'size': position.size,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'market_value': market_value,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pnl_pct,
                'entry_time': position.entry_time.isoformat(),
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit
            })
        
        return {
            'total_positions': len(self.positions),
            'total_value': total_positions_value,
            'unrealized_pnl': total_unrealized_pnl,
            'unrealized_pnl_pct': (total_unrealized_pnl / total_positions_value) * 100 if total_positions_value > 0 else 0,
            'positions': positions_summary
        }
    
    async def _update_portfolio_snapshot(self, market_data: Dict[str, Any] = None):
        """Update portfolio snapshot for performance tracking"""
        # Update current prices if market data is available
        if market_data:
            for symbol, position in self.positions.items():
                if symbol in market_data:
                    if isinstance(market_data[symbol], dict):
                        position.current_price = market_data[symbol].get('price', market_data[symbol].get('Close', position.current_price))
                    else:
                        position.current_price = market_data[symbol]
        
        # Calculate unrealized P&L
        unrealized_pnl = sum(
            (position.current_price - position.entry_price) * position.size
            for position in self.positions.values()
        )
        
        # Calculate day P&L (simplified - would need previous day's values)
        day_pnl = unrealized_pnl  # Simplified calculation
        
        # Calculate realized P&L from trades
        realized_pnl = sum(
            (trade.price - self._get_position_entry_price(trade.symbol, trade.timestamp)) * trade.quantity
            for trade in self.trade_history
            if trade.side == 'sell'
        )
        
        # Create portfolio snapshot
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            total_value=self.get_portfolio_value(),
            cash_balance=self.cash_balance,
            positions_value=sum(position.size * position.current_price for position in self.positions.values()),
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            day_pnl=day_pnl,
            positions=self.positions.copy(),
            performance_metrics={}  # Could add additional metrics here
        )
        
        # Store snapshot
        self.portfolio_snapshots.append(snapshot)
        self.performance_analyzer.add_snapshot(snapshot)
        
        # Keep only recent snapshots (e.g., last 1000)
        if len(self.portfolio_snapshots) > 1000:
            self.portfolio_snapshots = self.portfolio_snapshots[-1000:]
    
    def _get_position_entry_price(self, symbol: str, trade_time: datetime) -> float:
        """Get position entry price at specific time (simplified)"""
        # This is a simplified implementation
        # In practice, you'd track entry prices over time
        if symbol in self.positions:
            return self.positions[symbol].entry_price
        return 0.0
    
    async def check_exit_conditions(self) -> List[TradingSignal]:
        """Check for positions that should be closed"""
        exit_signals = []
        
        for symbol, position in self.positions.items():
            # Stop loss check
            if (position.stop_loss and 
                position.position_type == PositionType.LONG and 
                position.current_price <= position.stop_loss):
                
                exit_signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=1.0,
                    price=position.current_price,
                    timestamp=datetime.now(),
                    strategy="StopLoss",
                    position_size=position.size,
                    metadata={'exit_reason': 'stop_loss', 'stop_price': position.stop_loss}
                )
                exit_signals.append(exit_signal)
            
            # Take profit check
            elif (position.take_profit and 
                  position.position_type == PositionType.LONG and 
                  position.current_price >= position.take_profit):
                
                exit_signal = TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=1.0,
                    price=position.current_price,
                    timestamp=datetime.now(),
                    strategy="TakeProfit",
                    position_size=position.size,
                    metadata={'exit_reason': 'take_profit', 'target_price': position.take_profit}
                )
                exit_signals.append(exit_signal)
        
        return exit_signals
    
    def get_comprehensive_performance_metrics(self, benchmark_returns: pd.Series = None) -> Dict[str, Any]:
        """Get comprehensive portfolio performance metrics"""
        try:
            # Basic portfolio information
            portfolio_info = {
                'initial_capital': self.initial_capital,
                'current_value': self.get_portfolio_value(),
                'cash_balance': self.cash_balance,
                'positions_count': len(self.positions),
                'total_trades': len(self.trade_history)
            }
            
            # Performance analysis
            performance_metrics = self.performance_analyzer.calculate_comprehensive_metrics(benchmark_returns)
            
            # Risk metrics (latest)
            risk_summary = self.risk_manager.get_risk_summary()
            
            # Position summary
            positions_summary = self.get_positions_summary()
            
            # Trading activity metrics
            if self.trade_history:
                total_commission = sum(trade.commission for trade in self.trade_history)
                total_volume = sum(trade.quantity * trade.price for trade in self.trade_history)
                avg_trade_size = total_volume / len(self.trade_history)
                
                trading_metrics = {
                    'total_commission': total_commission,
                    'total_volume': total_volume,
                    'average_trade_size': avg_trade_size,
                    'commission_as_pct_of_volume': (total_commission / total_volume) * 100 if total_volume > 0 else 0
                }
            else:
                trading_metrics = {
                    'total_commission': 0,
                    'total_volume': 0,
                    'average_trade_size': 0,
                    'commission_as_pct_of_volume': 0
                }
            
            # Combine all metrics
            comprehensive_metrics = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_info': portfolio_info,
                'performance_metrics': performance_metrics,
                'risk_metrics': risk_summary,
                'positions_summary': positions_summary,
                'trading_metrics': trading_metrics
            }
            
            return comprehensive_metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate comprehensive metrics: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def save_portfolio_state(self, filepath: str) -> bool:
        """Save complete portfolio state"""
        try:
            # Prepare portfolio state for serialization
            portfolio_state = {
                'config': self.config,
                'initial_capital': self.initial_capital,
                'cash_balance': self.cash_balance,
                'positions': {
                    symbol: {
                        'symbol': pos.symbol,
                        'position_type': pos.position_type.value,
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'entry_time': pos.entry_time.isoformat(),
                        'stop_loss': pos.stop_loss,
                        'take_profit': pos.take_profit,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'metadata': pos.metadata
                    }
                    for symbol, pos in self.positions.items()
                },
                'order_history': [asdict(order) for order in self.order_history[-1000:]],  # Last 1000 orders
                'trade_history': [asdict(trade) for trade in self.trade_history[-1000:]],   # Last 1000 trades
                'portfolio_snapshots': [asdict(snapshot) for snapshot in self.portfolio_snapshots[-100:]],  # Last 100 snapshots
                'timestamp': datetime.now().isoformat()
            }
            
            # Handle datetime serialization in nested objects
            def serialize_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: serialize_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [serialize_datetime(item) for item in obj]
                else:
                    return obj
            
            portfolio_state = serialize_datetime(portfolio_state)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(portfolio_state, f, indent=2, default=str)
            
            # Save risk manager state
            risk_filepath = filepath.replace('.json', '_risk.pkl')
            self.risk_manager.save_risk_state(risk_filepath)
            
            logger.info(f"Portfolio state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save portfolio state: {e}")
            return False
    
    def load_portfolio_state(self, filepath: str) -> bool:
        """Load complete portfolio state"""
        try:
            # Load portfolio state
            with open(filepath, 'r') as f:
                portfolio_state = json.load(f)
            
            self.config = portfolio_state.get('config', {})
            self.initial_capital = portfolio_state.get('initial_capital', 100000.0)
            self.cash_balance = portfolio_state.get('cash_balance', self.initial_capital)
            
            # Reconstruct positions
            positions_data = portfolio_state.get('positions', {})
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
            
            # Load order history (simplified - would need full reconstruction in production)
            self.order_history = []  # Simplified loading
            self.trade_history = []  # Simplified loading
            
            # Load risk manager state
            risk_filepath = filepath.replace('.json', '_risk.pkl')
            try:
                self.risk_manager.load_risk_state(risk_filepath)
            except Exception as e:
                logger.warning(f"Could not load risk manager state: {e}")
            
            logger.info(f"Portfolio state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load portfolio state: {e}")
            return False
    
    async def initialize(self):
        """Initialize the portfolio manager - required by bot orchestrator"""
        logger.info("Advanced Portfolio Manager initialized")
        
        # Initialize execution engine if needed
        if hasattr(self.execution_engine, 'initialize'):
            await self.execution_engine.initialize()
        
        # Initialize risk manager if needed
        if hasattr(self.risk_manager, 'initialize'):
            await self.risk_manager.initialize()
        
        # Log initial state
        logger.info(f"Portfolio initialized with capital: ${self.initial_capital:,.2f}")
        logger.info(f"Max positions: {self.max_positions}")
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        total_value = self.cash_balance
        total_invested = 0
        position_count = len(self.positions)
        
        # Calculate total portfolio value
        for symbol, position in self.positions.items():
            position_value = position.get('quantity', 0) * position.get('current_price', position.get('price', 0))
            total_value += position_value
            total_invested += position.get('quantity', 0) * position.get('price', 0)
        
        # Calculate returns
        total_return = total_value - self.initial_capital if self.initial_capital > 0 else 0
        total_return_pct = (total_return / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        
        return {
            'timestamp': datetime.now(),
            'initial_capital': self.initial_capital,
            'cash_balance': self.cash_balance,
            'total_invested': total_invested,
            'total_value': total_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'position_count': position_count,
            'positions': self.positions,
            'max_positions': self.max_positions,
            'available_cash_pct': (self.cash_balance / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        }

# Alias for backward compatibility
PortfolioManager = AdvancedPortfolioManager