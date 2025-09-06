# Advanced News-Based Trading Bot

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-green.svg)]()

[🇫🇷 Version Française](README_FR.md) | [🇬🇧 English Version](README.md)

An advanced algorithmic trading bot that combines **dual machine learning approaches** with **multi-source sentiment analysis** to make informed trading decisions on US and European stocks.

## Key Features

### Dual Machine Learning Architecture
- **Traditional ML**: XGBoost + Random Forest with 50+ technical indicators
- **Transformer ML**: Custom financial Transformer for sequence modeling
- **Ensemble Strategy**: Intelligent combination of both approaches (60% Transformer, 40% Traditional)

### Multi-Source Sentiment Analysis
- **Reliable Sources**: NewsAPI, Alpha Vantage, Finnhub, Reuters, Bloomberg
- **Social Media**: Twitter/X, Reddit WallStreetBets (modular integration)
- **AI-Powered**: Gemini AI for advanced financial sentiment analysis
- **Regional Context**: US vs EU market sentiment differentiation

### Multi-Region Market Support
- **US Stocks**: AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA, JPM, V, JNJ
- **European Stocks**: ASML.AS, SAP, NESN.SW, MC.PA, OR.PA, RMS.PA, ADYEN.AS
- **Currency Normalization**: All prices normalized to EUR for consistency

### Advanced Risk Management
- **Position Sizing**: Maximum 15% per position, 2% portfolio risk
- **Correlation Checks**: Avoid over-concentration in similar assets
- **Volatility Adjustment**: Dynamic position sizing based on volatility
- **Transaction Costs**: Realistic 0.1% per trade simulation

## Architecture Overview

```
trading_bot/
├── main.py                 # Main execution script
├── core/
│   ├── config.py          # Configuration management
│   ├── database.py        # Database models and connections
│   └── utils.py           # Utility functions
├── data/
│   ├── collectors.py      # Multi-source data collection
│   ├── news_sources.py    # News API integrations
│   └── market_data.py     # Market data handling
├── analysis/
│   ├── sentiment.py       # Sentiment analysis engines
│   ├── social_media.py    # Social media sentiment (modular)
│   └── technical.py       # Technical analysis indicators
├── ml/
│   ├── traditional.py     # XGBoost/Random Forest models
│   ├── transformer.py     # Transformer architecture
│   └── ensemble.py        # Ensemble strategy
├── trading/
│   ├── strategy.py        # Trading strategy implementation
│   ├── risk_management.py # Risk management rules
│   └── execution.py       # Trade execution engine
└── analytics/
    ├── performance.py     # Performance analytics
    └── reporting.py       # Report generation
```

## Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL (optional, for data persistence)
- Redis (optional, for caching)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
export NEWS_API_KEY="your_news_api_key"
export GEMINI_API_KEY="your_gemini_api_key"
export ALPHA_VANTAGE_KEY="your_alpha_vantage_key"
export FINNHUB_KEY="your_finnhub_key"
# Optional for social media sentiment
export TWITTER_BEARER_TOKEN="your_twitter_token"
export REDDIT_CLIENT_ID="your_reddit_client_id"
export REDDIT_CLIENT_SECRET="your_reddit_secret"
```

4. **Configure the bot**
```python
# Edit core/config.py
INITIAL_CAPITAL = 10000.0  # €10,000 virtual budget
ENABLE_TRADITIONAL_ML = True
ENABLE_TRANSFORMER_ML = True
ENABLE_SOCIAL_SENTIMENT = False  # Set to True for social media
```

5. **Run the bot**
```bash
python main.py
```

## Trading Strategy

### Signal Generation Process

1. **Data Collection** (Every 5 minutes)
   - Market data from Yahoo Finance
   - News from multiple sources
   - Social media sentiment (if enabled)

2. **Sentiment Analysis**
   - Gemini AI processes each news article
   - Extracts: sentiment score, market impact, urgency, confidence
   - Aggregates daily sentiment by symbol

3. **ML Predictions**
   - Traditional ML: 5-class classification (Strong Sell to Strong Buy)
   - Transformer: Sequence-based prediction with 30-day lookback
   - Ensemble: Weighted combination based on confidence

4. **Signal Composition**
   - 50% ML predictions
   - 25% News sentiment
   - 15% Technical indicators
   - 10% News urgency factor

### Classification System

| Class | Label | Expected Return | Action |
|-------|-------|----------------|--------|
| 0 | Strong Sell | < -5% | Large Short Position |
| 1 | Sell | -5% to -2% | Small Short Position |
| 2 | Hold | -2% to +2% | No Action |
| 3 | Buy | +2% to +5% | Small Long Position |
| 4 | Strong Buy | > +5% | Large Long Position |

## Configuration

### Core Settings
```python
# Budget and Risk
INITIAL_CAPITAL = 10000.0      # Starting capital in EUR
MAX_POSITION_SIZE = 0.15       # Maximum 15% per position
MAX_PORTFOLIO_RISK = 0.02      # Maximum 2% daily portfolio risk

# ML Parameters
LOOKBACK_DAYS = 60             # Historical data for training
PREDICTION_HORIZON = 5         # Prediction timeframe (days)
SEQUENCE_LENGTH = 30           # Transformer sequence length

# Trading
TRANSACTION_COSTS = 0.001      # 0.1% per trade
REFRESH_INTERVAL = 300         # 5 minutes between cycles
```

### Model Weights
```python
# Ensemble weights (can be adjusted)
TRADITIONAL_WEIGHT = 0.4       # Traditional ML weight
TRANSFORMER_WEIGHT = 0.6       # Transformer weight

# Signal composition
ML_WEIGHT = 0.5               # ML predictions
SENTIMENT_WEIGHT = 0.25       # News sentiment
TECHNICAL_WEIGHT = 0.15       # Technical analysis
URGENCY_WEIGHT = 0.1          # News urgency
```

## Performance Metrics

The bot tracks comprehensive performance metrics:

- **Return Metrics**: Total return, annualized return, Sharpe ratio
- **Risk Metrics**: Maximum drawdown, volatility, VaR
- **Trading Metrics**: Win rate, average holding period, turnover
- **Model Performance**: Accuracy, precision, recall for each ML model

### Example Output
```
=== PERFORMANCE REPORT ===
Final Portfolio Value: €11,247.83
Total Return: +12.48%
Sharpe Ratio: 1.34
Max Drawdown: -3.21%
Win Rate: 64.2%
Total Trades: 127
```

## Modular Extensions

### Adding Social Media Sentiment

1. **Enable social sentiment**
```python
# In core/config.py
ENABLE_SOCIAL_SENTIMENT = True
```

2. **Configure social sources**
```python
SOCIAL_SOURCES = {
    'twitter': True,    # Twitter/X sentiment
    'reddit': True,     # Reddit WallStreetBets
    'discord': False    # Discord trading channels
}
```

3. **Adjust weights**
```python
# Signal composition with social media
ML_WEIGHT = 0.45
NEWS_SENTIMENT_WEIGHT = 0.20
SOCIAL_SENTIMENT_WEIGHT = 0.15
TECHNICAL_WEIGHT = 0.15
URGENCY_WEIGHT = 0.05
```

### Custom Indicators

Add custom technical indicators in `analysis/technical.py`:

```python
def custom_indicator(data: pd.DataFrame) -> pd.Series:
    """Your custom technical indicator"""
    return your_calculation(data)
```

## Testing and Validation

### Backtesting
```bash
python scripts/backtest.py --start-date 2023-01-01 --end-date 2024-01-01
```

### Model Validation
```bash
python scripts/validate_models.py --symbol AAPL --test-size 0.2
```

### Paper Trading
The bot runs in virtual mode by default. For paper trading with real market data:
```python
PAPER_TRADING = True
REAL_TRADING = False  # Set to True only when ready
```

## Requirements

### Core Dependencies
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `scikit-learn>=1.1.0` - Traditional ML
- `xgboost>=1.6.0` - Gradient boosting
- `torch>=1.12.0` - Deep learning
- `yfinance>=0.1.87` - Market data
- `google-generativeai>=0.3.0` - Sentiment analysis

### Optional Dependencies
- `postgresql` - Data persistence
- `redis` - Caching
- `tweepy` - Twitter/X integration
- `praw` - Reddit integration

## Disclaimer

**This is educational software for virtual trading only.**

- The bot trades with virtual money by default
- Past performance does not guarantee future results
- All trading involves risk of loss
- Use real money only after thorough testing and understanding
- The authors are not responsible for any financial losses

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Google Gemini AI** for advanced sentiment analysis
- **Yahoo Finance** for reliable market data
- **NewsAPI** for comprehensive news coverage
- **scikit-learn** and **PyTorch** communities
- **Open source community** for amazing tools and libraries

---

** Remember: Never invest more than you can afford to lose. This bot is for educational purposes and virtual trading.**