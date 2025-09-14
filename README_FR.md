# Bot de Trading Basé sur l'Analyse de Sentiment

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Actif](https://img.shields.io/badge/Status-Actif-green.svg)]()

Version Française | [English Version](README.md)

Un bot de trading algorithmique avancé qui combine **deux approches de machine learning** avec **l'analyse de sentiment multi-sources** pour prendre des décisions de trading éclairées sur les actions américaines et européennes.

## Fonctionnalités Principales

### Architecture Double Machine Learning
- **ML Traditionnel** : XGBoost + Random Forest avec 50+ indicateurs techniques
- **ML Transformer** : Architecture Transformer personnalisée pour la modélisation séquentielle
- **Stratégie d'Ensemble** : Combinaison intelligente des deux approches (60% Transformer, 40% Traditionnel)

### Analyse de Sentiment Multi-Sources
- **Sources Fiables** : NewsAPI, Alpha Vantage, Finnhub, Reuters, Bloomberg
- **Réseaux Sociaux** : Twitter/X, Reddit WallStreetBets (intégration modulaire)
- **IA Avancée** : Gemini AI pour l'analyse de sentiment financier
- **Contexte Régional** : Différenciation sentiment marché US vs EU

### Support Multi-Régions
- **Actions US** : AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA, JPM, V, JNJ
- **Actions Européennes** : ASML.AS, SAP, NESN.SW, MC.PA, OR.PA, RMS.PA, ADYEN.AS
- **Normalisation Devise** : Tous les prix normalisés en EUR pour cohérence

### Gestion de Risque Avancée
- **Taille Position** : Maximum 15% par position, 2% risque portefeuille
- **Vérifications Corrélation** : Évite la surconcentration dans des actifs similaires
- **Ajustement Volatilité** : Taille de position dynamique basée sur la volatilité
- **Coûts Transaction** : Simulation réaliste de 0,1% par trade

## Aperçu de l'Architecture

```
trading_bot/
├── main.py                 # Script d'exécution principal
├── core/
│   ├── config.py          # Gestion de la configuration
│   ├── database.py        # Modèles et connexions base de données
│   └── utils.py           # Fonctions utilitaires
├── data/
│   ├── collectors.py      # Collecte de données multi-sources
│   ├── news_sources.py    # Intégrations API news
│   └── market_data.py     # Gestion données marché
├── analysis/
│   ├── sentiment.py       # Moteurs d'analyse sentiment
│   ├── social_media.py    # Sentiment réseaux sociaux (modulaire)
│   └── technical.py       # Indicateurs analyse technique
├── ml/
│   ├── traditional.py     # Modèles XGBoost/Random Forest
│   ├── transformer.py     # Architecture Transformer
│   └── ensemble.py        # Stratégie d'ensemble
├── trading/
│   ├── strategy.py        # Implémentation stratégie trading
│   ├── risk_management.py # Règles gestion risque
│   └── execution.py       # Moteur d'exécution trades
└── analytics/
    ├── performance.py     # Analytics de performance
    └── reporting.py       # Génération de rapports
```

## Démarrage Rapide

### Prérequis
- Python 3.8+
- PostgreSQL (optionnel, pour persistance données)
- Redis (optionnel, pour mise en cache)

### Installation

1. **Cloner le repository**
```bash
git clone https://github.com/archer-paul/trading-bot.git
cd trading-bot
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

3. **Configurer les variables d'environnement**
```bash
export NEWS_API_KEY="votre_cle_news_api"
export GEMINI_API_KEY="votre_cle_gemini"
export ALPHA_VANTAGE_KEY="votre_cle_alpha_vantage"
export FINNHUB_KEY="votre_cle_finnhub"
# Optionnel pour sentiment réseaux sociaux
export TWITTER_BEARER_TOKEN="votre_token_twitter"
export REDDIT_CLIENT_ID="votre_client_id_reddit"
export REDDIT_CLIENT_SECRET="votre_secret_reddit"
```

4. **Configurer le bot**
```python
# Éditer core/config.py
INITIAL_CAPITAL = 10000.0  # Budget virtuel €10,000
ENABLE_TRADITIONAL_ML = True
ENABLE_TRANSFORMER_ML = True
ENABLE_SOCIAL_SENTIMENT = False  # Mettre True pour réseaux sociaux
```

5. **Lancer le bot**
```bash
python main.py
```

## Stratégie de Trading

### Processus de Génération de Signaux

1. **Collecte de Données** (Toutes les 5 minutes)
   - Données marché depuis Yahoo Finance
   - News depuis sources multiples
   - Sentiment réseaux sociaux (si activé)

2. **Analyse de Sentiment**
   - Gemini AI traite chaque article de news
   - Extrait : score sentiment, impact marché, urgence, confiance
   - Agrège le sentiment quotidien par symbole

3. **Prédictions ML**
   - ML Traditionnel : Classification 5 classes (Strong Sell à Strong Buy)
   - Transformer : Prédiction basée séquence avec lookback 30 jours
   - Ensemble : Combinaison pondérée basée sur confiance

4. **Composition du Signal**
   - 50% Prédictions ML
   - 25% Sentiment news
   - 15% Indicateurs techniques
   - 10% Facteur urgence news

### Système de Classification

| Classe | Label | Rendement Attendu | Action |
|--------|-------|------------------|--------|
| 0 | Strong Sell | < -5% | Grande Position Short |
| 1 | Sell | -5% à -2% | Petite Position Short |
| 2 | Hold | -2% à +2% | Aucune Action |
| 3 | Buy | +2% à +5% | Petite Position Long |
| 4 | Strong Buy | > +5% | Grande Position Long |

## Configuration

### Paramètres Principaux
```python
# Budget et Risque
INITIAL_CAPITAL = 10000.0      # Capital de départ en EUR
MAX_POSITION_SIZE = 0.15       # Maximum 15% par position
MAX_PORTFOLIO_RISK = 0.02      # Maximum 2% risque quotidien portefeuille

# Paramètres ML
LOOKBACK_DAYS = 60             # Données historiques pour entraînement
PREDICTION_HORIZON = 5         # Horizon de prédiction (jours)
SEQUENCE_LENGTH = 30           # Longueur séquence Transformer

# Trading
TRANSACTION_COSTS = 0.001      # 0,1% par trade
REFRESH_INTERVAL = 300         # 5 minutes entre cycles
```

### Poids des Modèles
```python
# Poids ensemble (ajustables)
TRADITIONAL_WEIGHT = 0.4       # Poids ML traditionnel
TRANSFORMER_WEIGHT = 0.6       # Poids Transformer

# Composition signal
ML_WEIGHT = 0.5               # Prédictions ML
SENTIMENT_WEIGHT = 0.25       # Sentiment news
TECHNICAL_WEIGHT = 0.15       # Analyse technique
URGENCY_WEIGHT = 0.1          # Urgence news
```

## Métriques de Performance

Le bot suit des métriques de performance complètes :

- **Métriques Rendement** : Rendement total, rendement annualisé, ratio Sharpe
- **Métriques Risque** : Drawdown maximum, volatilité, VaR
- **Métriques Trading** : Taux de gain, période de détention moyenne, turnover
- **Performance Modèles** : Précision, rappel, F1-score pour chaque modèle ML

### Exemple de Sortie
```
=== RAPPORT DE PERFORMANCE ===
Valeur Finale Portefeuille : €11 247,83
Rendement Total : +12,48%
Ratio Sharpe : 1,34
Drawdown Maximum : -3,21%
Taux de Gain : 64,2%
Total Trades : 127
```

## Extensions Modulaires

### Ajout Sentiment Réseaux Sociaux

1. **Activer sentiment social**
```python
# Dans core/config.py
ENABLE_SOCIAL_SENTIMENT = True
```

2. **Configurer sources sociales**
```python
SOCIAL_SOURCES = {
    'twitter': True,    # Sentiment Twitter/X
    'reddit': True,     # Reddit WallStreetBets
    'discord': False    # Canaux Discord trading
}
```

3. **Ajuster les poids**
```python
# Composition signal avec réseaux sociaux
ML_WEIGHT = 0.45
NEWS_SENTIMENT_WEIGHT = 0.20
SOCIAL_SENTIMENT_WEIGHT = 0.15
TECHNICAL_WEIGHT = 0.15
URGENCY_WEIGHT = 0.05
```

### Indicateurs Personnalisés

Ajouter des indicateurs techniques personnalisés dans `analysis/technical.py` :

```python
def indicateur_personnalise(data: pd.DataFrame) -> pd.Series:
    """Votre indicateur technique personnalisé"""
    return votre_calcul(data)
```

## Tests et Validation

### Backtesting
```bash
python scripts/backtest.py --start-date 2023-01-01 --end-date 2024-01-01
```

### Validation Modèles
```bash
python scripts/validate_models.py --symbol AAPL --test-size 0.2
```

### Paper Trading
Le bot fonctionne en mode virtuel par défaut. Pour le paper trading avec données marché réelles :
```python
PAPER_TRADING = True
REAL_TRADING = False  # Mettre True seulement quand prêt
```

## Exigences

### Dépendances Principales
- `pandas>=1.5.0` - Manipulation données
- `numpy>=1.21.0` - Calcul numérique
- `scikit-learn>=1.1.0` - ML traditionnel
- `xgboost>=1.6.0` - Gradient boosting
- `torch>=1.12.0` - Deep learning
- `yfinance>=0.1.87` - Données marché
- `google-generativeai>=0.3.0` - Analyse sentiment

### Dépendances Optionnelles
- `postgresql` - Persistance données
- `redis` - Mise en cache
- `tweepy` - Intégration Twitter/X
- `praw` - Intégration Reddit

## Avertissement

**Ceci est un logiciel éducatif pour trading virtuel uniquement.**

- Le bot trade avec de l'argent virtuel par défaut
- Les performances passées ne garantissent pas les résultats futurs
- Tout trading implique un risque de perte
- Utilisez de l'argent réel seulement après tests approfondis et compréhension
- Les auteurs ne sont pas responsables des pertes financières

## Contribution

1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/fonctionnalite-geniale`)
3. Commit vos changements (`git commit -m 'Ajouter fonctionnalité géniale'`)
4. Push vers la branche (`git push origin feature/fonctionnalite-geniale`)
5. Ouvrir une Pull Request

### Configuration Développement
```bash
# Installer dépendances développement
pip install -r requirements-dev.txt

# Lancer les tests
pytest tests/

# Lancer le linting
flake8 src/
black src/
```

## Architecture Modulaire Détaillée

### Collecte de Données (`data/`)
- **collectors.py** : Orchestrateur principal collecte données
- **news_sources.py** : APIs news traditionnelles (NewsAPI, Alpha Vantage)
- **social_sources.py** : APIs réseaux sociaux (Twitter, Reddit)
- **market_data.py** : Données marché et normalisation devises

### Analyse (`analysis/`)
- **sentiment.py** : Moteur Gemini AI pour sentiment
- **social_media.py** : Analyse sentiment réseaux sociaux
- **technical.py** : Tous les indicateurs techniques (RSI, MACD, etc.)

### Machine Learning (`ml/`)
- **traditional.py** : XGBoost, Random Forest, feature engineering
- **transformer.py** : Architecture Transformer financière
- **ensemble.py** : Combinaison intelligente des modèles

### Trading (`trading/`)
- **strategy.py** : Logique génération signaux
- **risk_management.py** : Gestion risque et position sizing
- **execution.py** : Exécution trades virtuels/réels

### Analytics (`analytics/`)
- **performance.py** : Calcul métriques performance
- **reporting.py** : Génération rapports détaillés

## Flux de Données

```
News APIs → sentiment.py → 
Market Data → technical.py → ensemble.py → strategy.py → execution.py
Social APIs → social_media.py → 
```

## Dashboards et Monitoring

### Métriques en Temps Réel
- Performance portefeuille
- Précision modèles ML
- Volume sentiment par source
- Corrélations actifs

### Alertes Automatiques
- Drawdown excessif
- Positions concentrées
- Erreurs modèles ML
- Anomalies sentiment

## Roadmap

### Version 1.1 (Prochaine)
- [ ] Interface web dashboard
- [ ] Alertes Telegram/Discord
- [ ] Support cryptomonnaies
- [ ] Analyse sentiment crypto Twitter

### Version 1.2
- [ ] API REST complète
- [ ] Mode multi-portefeuilles
- [ ] Intégration brokers réels
- [ ] Backtesting avancé

### Version 2.0
- [ ] Reinforcement Learning
- [ ] Analyse on-chain (crypto)
- [ ] Sentiment analysis vidéos YouTube
- [ ] Support trading options

## Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour détails.

## Remerciements

- **Google Gemini AI** pour l'analyse de sentiment avancée
- **Yahoo Finance** pour les données marché fiables
- **NewsAPI** pour la couverture news complète
- **Communautés scikit-learn et PyTorch**
- **Communauté open source** pour les outils et librairies fantastiques

---

** Rappel : N'investissez jamais plus que ce que vous pouvez vous permettre de perdre. Ce bot est à des fins éducatives et de trading virtuel.**