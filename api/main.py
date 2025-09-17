
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Créez une instance de l'application FastAPI
app = FastAPI(
    title="Trading Bot API",
    description="API pour interagir avec le bot de trading et visualiser les données.",
    version="1.0.0",
)

# Configuration du CORS (Cross-Origin Resource Sharing)
# Permet au frontend (ex: Next.js sur localhost:3000) de communiquer avec l'API (sur localhost:8080)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Attention: pour la production, restreindre à l'URL du frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Données Mock (temporaires, pour correspondre au frontend actuel) ---

model_performance_data = [
  { "name": 'XGBoost', "accuracy": 0.872, "precision": 0.845, "recall": 0.891, "f1Score": 0.867, "sharpe": 2.34 },
  { "name": 'AdaBoost', "accuracy": 0.834, "precision": 0.821, "recall": 0.847, "f1Score": 0.834, "sharpe": 1.98 },
  { "name": 'Transformer', "accuracy": 0.889, "precision": 0.876, "recall": 0.902, "f1Score": 0.889, "sharpe": 2.67 },
  { "name": 'LSTM', "accuracy": 0.801, "precision": 0.798, "recall": 0.804, "f1Score": 0.801, "sharpe": 1.76 },
  { "name": 'Random Forest', "accuracy": 0.856, "precision": 0.841, "recall": 0.871, "f1Score": 0.856, "sharpe": 2.12 },
]

ensemble_weights = [
  { "name": 'XGBoost', "value": 0.28, "color": '#3b82f6' },
  { "name": 'Transformer', "value": 0.35, "color": '#10b981' },
  { "name": 'AdaBoost', "value": 0.15, "color": '#f59e0b' },
  { "name": 'LSTM', "value": 0.12, "color": '#ef4444' },
  { "name": 'Random Forest', "value": 0.10, "color": '#8b5cf6' },
]

# --- Endpoints de l'API ---

@app.get("/")
def read_root():
    return {"status": "Trading Bot API is running"}

@app.get("/api/ml-dashboard")
async def get_ml_dashboard_data():
    """
    Endpoint pour fournir les données du tableau de bord Machine Learning.
    NOTE: Renvoie actuellement des données statiques.
    """
    # TODO: Remplacer par des appels aux vrais composants du bot
    return {
        "model_performance": model_performance_data,
        "ensemble_weights": ensemble_weights,
        "feature_importance": [
            { "feature": 'Price Momentum (5d)', "importance": 0.91 },
            { "feature": 'Volume Surge (24h)', "importance": 0.82 },
            { "feature": 'News Sentiment Score', "importance": 0.78 },
            { "feature": 'RSI (14d)', "importance": 0.65 },
            { "feature": 'Macro-Economic Trend', "importance": 0.55 },
        ],
        "model_health": {
            "data_quality": 0.94,
            "prediction_drift": 0.12,
            "model_stability": 0.87,
            "latency": 120 # ms
        }
    }

@app.get("/api/ml/cache-stats")
async def get_cache_stats():
    """
    Endpoint pour les statistiques du cache des modèles ML.
    NOTE: Renvoie actuellement des données statiques.
    """
    # TODO: Remplacer par un appel au DataCacheManager
    return {
        "hit_rates": {
            "Traditional Models": 0.85,
            "Transformer Models": 0.76,
            "Ensemble Cache": 0.92
        },
        "ttl_seconds": {
            "Traditional Models": 86400,
            "Transformer Models": 172800
        },
        "memory_usage": {
            "total_mb": 256.8,
            "avg_entry_kb": 1024.5,
            "efficiency": 0.88
        }
    }

@app.get("/api/ml/batch-results")
async def get_batch_results():
    """
    Endpoint pour les résultats de l'entraînement cross-symbol.
    NOTE: Renvoie actuellement des données statiques.
    """
    # TODO: Remplacer par les vrais résultats du BatchMLTrainer
    return {
        "total_symbols": 150,
        "training_duration_hours": 4.5,
        "avg_improvement": 0.053, # 5.3% improvement vs single-symbol models
        "cv_score": 0.78,
        "correlation_improvements": {
            "AAPL-MSFT": 0.12,
            "GOOG-AMZN": 0.09,
            "TSLA-NVDA": 0.15
        },
        "total_features": 119,
        "feature_importance_score": 0.847,
        "dimensionality_reduction": "67%"
    }

@app.get("/api/risk-metrics")
async def get_risk_metrics():
    """
    Endpoint pour les métriques de risque détaillées.
    (Placeholder)
    """
    return {"message": "Endpoint pour les métriques de risque. Non implémenté."}

@app.get("/api/kg/cascade")
async def get_kg_cascade(event: str = "test"):
    """
    Endpoint pour l'analyse de cascade du Knowledge Graph.
    (Placeholder)
    """
    return {"message": f"Analyse de cascade pour l'événement '{event}'. Non implémenté."}

# --- Lancement du serveur (pour le développement) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
