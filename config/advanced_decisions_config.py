"""
Configuration for Advanced Portfolio Decision Engine
"""

# Base configuration for advanced decisions
ADVANCED_DECISIONS_CONFIG = {
    # Regime Detection Settings
    'regime_detection': {
        'regime_lookback_days': 60,
        'volatility_threshold': 0.02,  # 2% daily volatility threshold
        'trend_threshold': 0.05,       # 5% trend strength threshold
    },

    # Signal Aggregation Settings
    'signal_aggregation': {
        # Base weights for different signal types
        'base_weights': {
            'technical': 0.25,
            'ml': 0.30,
            'sentiment': 0.15,
            'fundamental': 0.20,
            'momentum': 0.10
        },

        # Regime-specific adjustments (multipliers applied to base weights)
        'regime_adjustments': {
            'bull_trend': {
                'momentum': 1.3,
                'sentiment': 1.2,
                'technical': 0.9
            },
            'bear_trend': {
                'fundamental': 1.4,
                'risk': 1.5,
                'momentum': 0.8
            },
            'high_volatility': {
                'risk': 1.6,
                'technical': 1.2,
                'sentiment': 0.8
            },
            'low_volatility': {
                'momentum': 1.2,
                'ml': 1.1,
                'fundamental': 1.1
            },
            'sideways': {
                'mean_reversion': 1.3,
                'technical': 1.1,
                'momentum': 0.9
            },
            'crisis': {
                'fundamental': 1.5,
                'risk': 2.0,
                'sentiment': 0.6
            }
        }
    },

    # Portfolio Optimization Settings
    'portfolio_optimization': {
        'target_return': 0.12,         # 12% annual target return
        'max_weight': 0.15,           # Maximum 15% per position
        'min_weight': 0.02,           # Minimum 2% per position
        'risk_free_rate': 0.03,       # 3% risk-free rate
        'risk_aversion': 3.0,         # Risk aversion parameter for optimization
    },

    # Decision Filtering
    'min_conviction': 0.3,            # Minimum conviction threshold
    'rebalance_threshold': 0.05,      # Minimum weight change to trigger rebalance
    'max_portfolio_concentration': 0.6,  # Maximum allocation to single sector

    # Risk Management
    'risk_constraints': {
        'max_portfolio_volatility': 0.25,  # 25% maximum portfolio volatility
        'max_individual_volatility': 0.35, # 35% maximum individual stock volatility
        'max_correlation': 0.7,            # Maximum correlation between positions
        'liquidity_threshold': 0.5,        # Minimum liquidity score
    },

    # Performance Tracking
    'performance_tracking': {
        'lookback_periods': [30, 90, 252],  # Days for performance evaluation
        'rebalance_frequency': 5,           # Days between rebalancing checks
        'adaptive_learning_rate': 0.1,     # Learning rate for strategy adaptation
    }
}

def get_advanced_decisions_config():
    """Get the advanced decisions configuration"""
    return ADVANCED_DECISIONS_CONFIG.copy()

def update_config_for_regime(config, regime):
    """Update configuration based on detected market regime"""
    regime_configs = {
        'bull_trend': {
            'min_conviction': 0.25,  # Lower threshold in bull markets
            'max_weight': 0.18,      # Allow larger positions
            'target_return': 0.15,   # Higher target return
        },
        'bear_trend': {
            'min_conviction': 0.4,   # Higher threshold in bear markets
            'max_weight': 0.12,      # Smaller positions
            'target_return': 0.08,   # Lower target return
        },
        'high_volatility': {
            'min_conviction': 0.45,  # Very high threshold in volatile markets
            'max_weight': 0.10,      # Much smaller positions
            'rebalance_threshold': 0.03,  # More frequent rebalancing
        },
        'crisis': {
            'min_conviction': 0.6,   # Extremely high threshold in crisis
            'max_weight': 0.08,      # Very small positions
            'target_return': 0.05,   # Capital preservation mode
        }
    }

    regime_specific = regime_configs.get(regime, {})

    # Update config with regime-specific settings
    updated_config = config.copy()
    for key, value in regime_specific.items():
        if key in updated_config:
            updated_config[key] = value

    return updated_config