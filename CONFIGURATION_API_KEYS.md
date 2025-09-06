# Guide de Configuration des Clés API - Trading Bot

## 📋 Vue d'ensemble

Ce guide détaille la configuration complète des clés API nécessaires pour votre Trading Bot. Certaines clés sont **obligatoires**, d'autres sont **recommandées** ou **optionnelles**.

---

## 🔑 Clés API Obligatoires

### 1. Gemini API Key (OBLIGATOIRE)
**Utilisation :** Analyse de sentiment avec IA générative Google

**Comment obtenir :**
1. Rendez-vous sur [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Connectez-vous avec votre compte Google
3. Cliquez sur "Create API Key"
4. Copiez la clé générée

**Configuration :**
```env
GEMINI_API_KEY=votre_clé_api_gemini_ici
```

---

## 📰 Clés API pour les Données Financières (Recommandées)

### 2. News API Key (RECOMMANDÉE)
**Utilisation :** Récupération d'actualités financières

**Comment obtenir :**
1. Allez sur [NewsAPI.org](https://newsapi.org/)
2. Cliquez sur "Get API Key"
3. Créez un compte gratuit
4. Copiez votre clé API

**Limites gratuites :** 1000 requêtes/jour
**Configuration :**
```env
NEWS_API_KEY=votre_clé_news_api_ici
```

### 3. Alpha Vantage API Key (RECOMMANDÉE)
**Utilisation :** Données financières et économiques détaillées

**Comment obtenir :**
1. Visitez [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Remplissez le formulaire gratuit
3. Vérifiez votre email
4. Copiez la clé reçue par email

**Limites gratuites :** 25 requêtes/jour
**Configuration :**
```env
ALPHA_VANTAGE_KEY=votre_clé_alpha_vantage_ici
```

### 4. Finnhub API Key (RECOMMANDÉE)
**Utilisation :** Données de marché en temps réel

**Comment obtenir :**
1. Allez sur [Finnhub.io](https://finnhub.io/)
2. Créez un compte gratuit
3. Accédez au Dashboard
4. Copiez votre API Key

**Limites gratuites :** 60 appels/minute
**Configuration :**
```env
FINNHUB_KEY=votre_clé_finnhub_ici
```

---

## 📱 Clés API Réseaux Sociaux (Optionnelles)

### 5. Twitter/X API (OPTIONNELLE)
**Utilisation :** Analyse de sentiment Twitter pour le trading

**Comment obtenir :**
1. Allez sur [Twitter Developer Portal](https://developer.twitter.com/)
2. Créez un compte développeur
3. Créez une nouvelle App
4. Générez un Bearer Token

**Configuration :**
```env
TWITTER_BEARER_TOKEN=votre_bearer_token_twitter_ici
```

### 6. Reddit API (OPTIONNELLE)
**Utilisation :** Analyse de sentiment Reddit (r/wallstreetbets, etc.)

**Comment obtenir :**
1. Allez sur [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Cliquez "Create App"
3. Choisissez "script"
4. Notez le Client ID et Client Secret

**Configuration :**
```env
REDDIT_CLIENT_ID=votre_client_id_reddit_ici
REDDIT_CLIENT_SECRET=votre_client_secret_reddit_ici
REDDIT_USER_AGENT=TradingBot/1.0
```

---

## 🗄️ Configuration Base de Données

### Redis (Recommandé pour le cache)
**Utilisation :** Cache des données pour améliorer les performances

**Configuration WSL :**
```bash
# Démarrer Redis dans WSL
sudo service redis-server start

# Vérifier le statut
redis-cli ping
```

**Configuration .env :**
```env
REDIS_URL=redis://localhost:6379
```

### PostgreSQL (Optionnel)
**Utilisation :** Stockage persistant des données historiques

**Installation WSL :**
```bash
# Installer PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Créer utilisateur et base
sudo -u postgres createuser trading_user
sudo -u postgres createdb trading_bot
```

**Configuration .env :**
```env
DATABASE_URL=postgresql://trading_user:password@localhost/trading_bot
```

---

## ⚙️ Configuration Complète du Fichier .env

Créez un fichier `.env` à la racine de votre projet avec ce contenu :

```env
# =============================================================================
# CLÉS API OBLIGATOIRES
# =============================================================================
GEMINI_API_KEY=votre_clé_api_gemini_ici

# =============================================================================
# CLÉS API DONNÉES FINANCIÈRES (Recommandées)
# =============================================================================
NEWS_API_KEY=votre_clé_news_api_ici
ALPHA_VANTAGE_KEY=votre_clé_alpha_vantage_ici
FINNHUB_KEY=votre_clé_finnhub_ici

# =============================================================================
# CLÉS API RÉSEAUX SOCIAUX (Optionnelles)
# =============================================================================
TWITTER_BEARER_TOKEN=votre_bearer_token_twitter_ici
REDDIT_CLIENT_ID=votre_client_id_reddit_ici
REDDIT_CLIENT_SECRET=votre_client_secret_reddit_ici
REDDIT_USER_AGENT=TradingBot/1.0

# =============================================================================
# CONFIGURATION BASE DE DONNÉES
# =============================================================================
DATABASE_URL=postgresql://trading_user:password@localhost/trading_bot
REDIS_URL=redis://localhost:6379

# =============================================================================
# CONFIGURATION GÉNÉRALE DU BOT
# =============================================================================
LOG_LEVEL=INFO
DEBUG_MODE=False
MAX_CYCLES=0
BACKTEST_MODE=False

# =============================================================================
# PARAMÈTRES DE TRADING
# =============================================================================
INITIAL_CAPITAL=10000.0
ENABLE_SOCIAL_SENTIMENT=False
```

---

## 🚀 Instructions de Setup Étape par Étape

### Étape 1: Copier le Template
```bash
# Copier le template fourni par setup.py
copy .env.template .env
```

### Étape 2: Configurer les Clés Obligatoires
1. Ouvrez le fichier `.env` dans votre éditeur
2. Remplacez `votre_clé_api_gemini_ici` par votre vraie clé Gemini

### Étape 3: Ajouter les Clés Recommandées
1. Configurez au moins une clé parmi News API, Alpha Vantage ou Finnhub
2. Plus vous en configurez, plus le bot aura de données

### Étape 4: Configuration Redis (Recommandé)
```bash
# Dans WSL
sudo service redis-server start

# Vérifier
redis-cli ping
# Doit retourner: PONG
```

### Étape 5: Validation
```bash
# Utiliser python3.13 au lieu de python
python3.13 setup.py validate
```

---

## 🔧 Résolution des Problèmes Courants

### Problème Redis sur Windows/WSL
**Symptôme :** `setup.py` dit que Redis n'est pas trouvé
**Solution :**
```bash
# Dans WSL, démarrer Redis
sudo service redis-server start

# Modifier setup.py pour utiliser WSL (optionnel)
# Ou ignorer l'avertissement si vous utilisez python3.13
```

### Problème Python non trouvé
**Symptôme :** `Python est introuvable`
**Solution :** Utilisez `python3.13` au lieu de `python`

### Limites des APIs Gratuites
- **News API :** 1000 requêtes/jour
- **Alpha Vantage :** 25 requêtes/jour  
- **Finnhub :** 60 requêtes/minute

**Conseil :** Activez Redis pour mettre en cache les données et réduire les appels API

---

## 🏛️ APIs Congress Trading (Transactions du Congrès)

**Pourquoi ?** Les transactions des membres du Congrès américain surperforment historiquement le marché

### 1. Finnhub.io (Insider Trading)
**Utilisation :** L'API Finnhub (déjà configurée) fournit des données sur les transactions d'initiés qui incluent certaines transactions du Congrès.

### 2. Senate Trading API 
**Spécialisé dans les données du Congrès**

#### Obtention de la clé
1. Visitez [https://api.senatetrading.com/](https://api.senatetrading.com/)
2. Créez un compte
3. Récupérez votre clé API

#### Configuration
```env
# Congress Trading Analysis (Multiple APIs)
FINNHUB_KEY=votre_cle_finnhub  # Déjà configurée
SENATE_TRADING_KEY=votre_cle_senate_trading
ENABLE_CONGRESS_TRACKING=True
CONGRESS_LOOKBACK_DAYS=90
MIN_CONGRESS_TRANSACTION=15000
```

### Fonctionnalités
- **Double source** : Finnhub + Senate Trading pour plus de données
- **Déduplication automatique** des transactions entre APIs
- **Gestion d'erreurs transparente** : continue si une API échoue
- **Filtrage intelligent** par valeur minimum (>$15k)
- **Logging détaillé** pour identifier les problèmes API

### Avantages de cette approche
- **Plus de requêtes** : Distribution sur 2 APIs
- **Fiabilité** : Continue si une API est en panne
- **Données complémentaires** : Chaque API peut avoir des données différentes
- **100% fiable** : Aucune donnée fictive, erreurs clairement identifiées

**Note :** Le bot fonctionne avec une seule API ou aucune (analyse désactivée)

---

## 📈 Configuration Recommandée pour Débuter

**Configuration minimale fonctionnelle :**
```env
GEMINI_API_KEY=votre_clé_gemini
NEWS_API_KEY=votre_clé_news_api
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
INITIAL_CAPITAL=10000.0
```

**Configuration optimale :**
```env
# APIs principales
GEMINI_API_KEY=votre_clé_gemini
NEWS_API_KEY=votre_clé_news_api

# APIs financières avancées
ALPHA_VANTAGE_KEY=votre_clé_alpha_vantage
FINNHUB_KEY=votre_clé_finnhub

# Congress trading (bonus)
QUIVER_API_KEY=votre_clé_quiver

# Réseaux sociaux
REDDIT_CLIENT_ID=votre_client_id_reddit
REDDIT_CLIENT_SECRET=votre_secret_reddit

# Infrastructure
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost/trading_bot
```

---

## 🔐 Sécurité des Clés API

1. **JAMAIS** commiter le fichier `.env` dans Git
2. Gardez vos clés privées
3. Utilisez des environnements séparés (dev/prod)
4. Renouvelez régulièrement vos clés
5. Surveillez l'usage de vos quotas

---

## ✅ Checklist Finale

- [ ] Fichier `.env` créé
- [ ] Clé Gemini configurée (OBLIGATOIRE)
- [ ] Au moins une clé de données financières configurée
- [ ] Redis démarré dans WSL
- [ ] Validation réussie avec `python3.13 setup.py validate`
- [ ] Test réussi avec `python3.13 setup.py test`

Une fois ces étapes complétées, votre Trading Bot sera prêt à fonctionner !