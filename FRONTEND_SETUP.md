# Trading Bot Frontend Setup Guide

Ce guide vous explique comment démarrer le nouveau frontend (frontend2) connecté au backend de votre trading bot.

## Architecture

- **Backend**: Python Flask/FastAPI sur le port 8080
- **Frontend**: React + Vite sur le port 5173
- **API**: Communication REST + WebSocket

## Démarrage Rapide

### Option 1: Script automatique (Recommandé)

**Windows:**
```bash
start_frontend.bat
```

**Linux/macOS:**
```bash
./start_frontend.sh
```

### Option 2: Démarrage manuel

```bash
cd frontend2
npm install
npm run dev
```

## URLs d'accès

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8080
- **API Health Check**: http://localhost:8080/health

## Fonctionnalités Implementées

### 1. Dashboard Principal
- **Layout 2/3 - 1/3**: Graphique de performance vs S&P 500 (2/3) + Camembert des holdings par secteur (1/3)
- **API Integration**: Connexion aux endpoints `/api/portfolio` et `/portfolio/holdings`
- **Temps réel**: Mise à jour automatique des données

### 2. Page ML Observatory
- **Layout haut**: Camembert des poids du meta-learner (gauche) + Graphique d'erreur de prédiction (droite)
- **Métriques ML**: Accuracy ensemble, poids des modèles, convergence
- **Visualisations**: Courbes d'erreur validation/test, statut de convergence

### 3. Connectivité Backend
- **Client API TypeScript**: Gestion complète des endpoints
- **Gestion d'erreurs**: Fallback sur données mock en cas d'erreur API
- **WebSocket**: Support pour les mises à jour temps réel
- **React Query**: Cache intelligent et refetch automatique

## Structure du Code

```
frontend2/
├── src/
│   ├── components/
│   │   ├── charts/              # Nouveaux graphiques
│   │   │   ├── PortfolioHoldingsPieChart.tsx
│   │   │   ├── MetaLearnerWeightsPieChart.tsx
│   │   │   └── ModelPredictionErrorChart.tsx
│   │   ├── dashboard/           # Composants dashboard
│   │   └── ui/                  # ShadCN UI components
│   ├── lib/
│   │   └── api.ts               # Client API TypeScript
│   ├── pages/
│   │   ├── Index.tsx            # Dashboard principal (modifié)
│   │   └── MLObservatory.tsx    # Page ML (modifiée)
│   └── ...
├── .env                         # Configuration API
└── package.json                # Dépendances (Lovable supprimé)
```

## Modifications Effectuées

### ✅ Connexion Backend
- Client API TypeScript complet
- Configuration des endpoints
- Gestion des erreurs et fallback
- Support WebSocket

### ✅ Suppression Lovable
- Références supprimées du package.json
- vite.config.ts nettoyé
- README.md réécrit
- Meta tags mis à jour

### ✅ Dashboard Principal
- Layout 2/3 - 1/3 implémenté
- Graphique performance vs S&P 500 connecté à l'API
- Camembert holdings par secteur avec interactions
- Responsive design

### ✅ Page ML Observatory
- Camembert poids meta-learner (haut gauche)
- Graphique erreur de prédiction (haut droite)
- Visualisation convergence et métriques
- Connexion aux endpoints ML

### ✅ Configuration Serveur
- Scripts de démarrage Windows/Linux
- Configuration Vite optimisée
- Variables d'environnement

## Endpoints API Utilisés

```typescript
// Portfolio
GET /api/portfolio              // Vue d'ensemble portfolio
GET /portfolio/performance      // Performance vs benchmark
GET /portfolio/holdings         // Détails des positions

// Machine Learning
GET /api/ml-dashboard          // Métriques ML
GET /ml/metrics               // Performance des modèles
GET /api/ml/cache-stats       // Statistiques cache

// Autres
GET /health                   // Santé du système
GET /api/sentiment-summary    // Analyse sentiment
GET /api/risk-management      // Gestion risques
```

## Prochaines Étapes

1. **Test de connexion**: Vérifiez que le backend est démarré sur le port 8080
2. **Démarrage frontend**: Utilisez les scripts fournis
3. **Vérification API**: Ouvrez les outils développeur pour vérifier les appels API
4. **Personnalisation**: Ajustez les couleurs, layout selon vos préférences

## Dépannage

### Erreur "API not available"
- Vérifiez que le backend fonctionne sur http://localhost:8080
- Testez l'endpoint: http://localhost:8080/health

### Erreur de build
```bash
cd frontend2
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Port déjà utilisé
- Modifiez le port dans `vite.config.ts`
- Ou arrêtez le service utilisant le port 5173

## Support

Le frontend est maintenant entièrement intégré à votre architecture de trading bot avec des visualisations modernes et une connectivité API robuste.