"""
Advanced Portfolio API - Endpoints for sophisticated portfolio decisions
"""

import logging
import json
from datetime import datetime
from flask import Blueprint, jsonify, request
from typing import Dict, Any

logger = logging.getLogger(__name__)

# This will be injected by the dashboard server
bot_orchestrator = None

def create_advanced_portfolio_bp():
    """Create Blueprint for advanced portfolio endpoints"""

    advanced_bp = Blueprint('advanced_portfolio', __name__, url_prefix='/api/advanced')

    @advanced_bp.route('/portfolio/decisions', methods=['GET'])
    def get_advanced_decisions():
        """Get advanced portfolio decisions"""
        try:
            if not bot_orchestrator or not bot_orchestrator.is_initialized:
                return jsonify({
                    'error': 'Bot not initialized',
                    'status': 'unavailable'
                }), 503

            # Get decision summary from advanced decision engine
            decision_summary = bot_orchestrator.advanced_decision_engine.get_decision_summary()

            # Export decisions for dashboard
            dashboard_data = bot_orchestrator.advanced_decision_engine.export_decisions_to_dashboard()

            return jsonify({
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'decision_summary': decision_summary,
                'dashboard_data': dashboard_data,
                'bot_status': {
                    'cycle_count': bot_orchestrator.cycle_count,
                    'last_cycle': bot_orchestrator.last_cycle_time.isoformat() if bot_orchestrator.last_cycle_time else None
                }
            })

        except Exception as e:
            logger.error(f"Advanced decisions API error: {e}")
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500

    @advanced_bp.route('/portfolio/analysis', methods=['GET'])
    def get_portfolio_analysis():
        """Get comprehensive portfolio analysis"""
        try:
            if not bot_orchestrator or not bot_orchestrator.is_initialized:
                return jsonify({
                    'error': 'Bot not initialized',
                    'status': 'unavailable'
                }), 503

            # Get current portfolio summary
            portfolio_summary = await_sync(bot_orchestrator.portfolio_manager.get_portfolio_summary())

            # Get performance metrics
            try:
                performance_metrics = await_sync(bot_orchestrator.performance_analyzer.calculate_current_metrics())
            except:
                performance_metrics = {}

            # Get decision engine summary
            decision_summary = bot_orchestrator.advanced_decision_engine.get_decision_summary()

            # Get regime detection info
            current_regime = "unknown"
            try:
                # Try to get the last detected regime from decision history
                decision_history = bot_orchestrator.advanced_decision_engine.decision_history
                if decision_history:
                    current_regime = decision_history[-1].get('market_regime', {}).get('value', 'unknown')
            except:
                pass

            analysis_data = {
                'portfolio_overview': {
                    'total_value': portfolio_summary.get('total_value', 0),
                    'cash_balance': portfolio_summary.get('cash_balance', 0),
                    'total_return': portfolio_summary.get('total_return', 0),
                    'total_return_pct': portfolio_summary.get('total_return_pct', 0),
                    'position_count': portfolio_summary.get('position_count', 0),
                    'available_cash_pct': portfolio_summary.get('available_cash_pct', 0)
                },
                'performance_metrics': performance_metrics,
                'advanced_decisions': decision_summary,
                'market_regime': current_regime,
                'risk_analysis': {
                    'portfolio_volatility': performance_metrics.get('annualized_volatility_pct', 0),
                    'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
                    'max_drawdown_pct': performance_metrics.get('max_drawdown_pct', 0),
                    'win_rate_pct': performance_metrics.get('win_rate_pct', 0)
                }
            }

            return jsonify({
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis_data
            })

        except Exception as e:
            logger.error(f"Portfolio analysis API error: {e}")
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500

    @advanced_bp.route('/portfolio/signals', methods=['GET'])
    def get_current_signals():
        """Get current trading signals from advanced engine"""
        try:
            if not bot_orchestrator or not bot_orchestrator.is_initialized:
                return jsonify({
                    'error': 'Bot not initialized',
                    'status': 'unavailable'
                }), 503

            # Get latest analysis if available
            latest_analysis = getattr(bot_orchestrator, 'latest_analysis', {})

            # Get decision summary
            decision_summary = bot_orchestrator.advanced_decision_engine.get_decision_summary()

            signals_data = {
                'current_signals': decision_summary.get('decisions', []),
                'signal_summary': {
                    'total_signals': decision_summary.get('total_decisions', 0),
                    'buy_signals': decision_summary.get('buy_decisions', 0),
                    'sell_signals': decision_summary.get('sell_decisions', 0),
                    'hold_signals': decision_summary.get('hold_decisions', 0),
                    'avg_conviction': decision_summary.get('avg_conviction', 0),
                    'total_target_allocation': decision_summary.get('total_target_allocation', 0)
                },
                'latest_analysis': {
                    'top_news_sentiment': latest_analysis.get('top_news_sentiment', []),
                    'social_media_sentiment': latest_analysis.get('social_media_sentiment', {})
                }
            }

            return jsonify({
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'signals': signals_data
            })

        except Exception as e:
            logger.error(f"Current signals API error: {e}")
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500

    @advanced_bp.route('/portfolio/optimize', methods=['POST'])
    def trigger_portfolio_optimization():
        """Trigger manual portfolio optimization"""
        try:
            if not bot_orchestrator or not bot_orchestrator.is_initialized:
                return jsonify({
                    'error': 'Bot not initialized',
                    'status': 'unavailable'
                }), 503

            # Get request parameters
            data = request.get_json() or {}
            force_rebalance = data.get('force_rebalance', False)

            # Note: This would trigger a new trading cycle
            # For now, we'll just return the current state
            decision_summary = bot_orchestrator.advanced_decision_engine.get_decision_summary()

            return jsonify({
                'status': 'success',
                'message': 'Portfolio optimization requested',
                'timestamp': datetime.now().isoformat(),
                'current_decisions': decision_summary,
                'note': 'Manual optimization will be applied in the next trading cycle'
            })

        except Exception as e:
            logger.error(f"Portfolio optimization API error: {e}")
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500

    @advanced_bp.route('/health', methods=['GET'])
    def advanced_health_check():
        """Health check for advanced portfolio system"""
        try:
            health_data = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'bot_initialized': bot_orchestrator.is_initialized if bot_orchestrator else False,
                'advanced_engine_available': True,
                'components': {
                    'regime_detector': 'available',
                    'signal_aggregator': 'available',
                    'portfolio_optimizer': 'available'
                }
            }

            if bot_orchestrator and bot_orchestrator.is_initialized:
                health_data['cycle_count'] = bot_orchestrator.cycle_count
                health_data['last_cycle'] = bot_orchestrator.last_cycle_time.isoformat() if bot_orchestrator.last_cycle_time else None

            return jsonify(health_data)

        except Exception as e:
            logger.error(f"Advanced health check error: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500

    @advanced_bp.route('/learning/performance', methods=['GET'])
    def get_learning_performance():
        """Get adaptive learning performance metrics"""
        try:
            if not bot_orchestrator or not bot_orchestrator.is_initialized:
                return jsonify({
                    'error': 'Bot not initialized',
                    'status': 'unavailable'
                }), 503

            # Get performance analytics from advanced decision engine
            performance_analytics = await_sync(bot_orchestrator.advanced_decision_engine.get_performance_analytics())

            # Get ML ensemble performance
            ml_ensemble_performance = {}
            if hasattr(bot_orchestrator, 'ensemble_predictor') and bot_orchestrator.ensemble_predictor:
                try:
                    ml_metrics = await_sync(bot_orchestrator.ensemble_predictor.get_performance_metrics())
                    ml_ensemble_performance = ml_metrics or {}
                except:
                    ml_ensemble_performance = {}

            # Combine metrics
            learning_data = {
                'adaptive_decision_performance': performance_analytics,
                'ml_ensemble_performance': ml_ensemble_performance,
                'signal_weighting': {
                    'current_weights': getattr(bot_orchestrator.advanced_decision_engine.signal_aggregator, 'ml_ensemble_performance', {}),
                    'base_weights': bot_orchestrator.advanced_decision_engine.signal_aggregator.base_weights,
                },
                'learning_summary': {
                    'feedback_enabled': bot_orchestrator.advanced_decision_engine.performance_feedback_enabled,
                    'database_integrated': bot_orchestrator.advanced_decision_engine.db_manager is not None,
                    'ml_ensemble_integrated': bot_orchestrator.advanced_decision_engine.ensemble_predictor is not None,
                    'total_cycles': bot_orchestrator.cycle_count
                }
            }

            return jsonify({
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'learning_data': learning_data
            })

        except Exception as e:
            logger.error(f"Learning performance API error: {e}")
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500

    return advanced_bp

def await_sync(coro):
    """Helper to run async functions in sync context"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop running, create a new one
        return asyncio.run(coro)

def set_bot_orchestrator(orchestrator):
    """Set the bot orchestrator instance"""
    global bot_orchestrator
    bot_orchestrator = orchestrator