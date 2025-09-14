"""
Advanced Ensemble Predictor for Trading Bot
Combines multiple ML models using sophisticated ensemble techniques
including stacking, blending, Bayesian model averaging, and dynamic weighting
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import asyncio
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import our custom models
from ml.traditional_ml import TraditionalMLPredictor
from ml.transformer_ml import TransformerMLPredictor

logger = logging.getLogger(__name__)

class BayesianModelAveraging:
    """Bayesian Model Averaging for ensemble weighting"""
    
    def __init__(self, decay_factor: float = 0.95):
        self.decay_factor = decay_factor
        self.model_priors = {}
        self.model_likelihoods = {}
        self.model_posteriors = {}
        self.prediction_history = []
        self.actual_history = []
    
    def update_model_performance(self, model_predictions: Dict[str, float], actual: float):
        """Update model performance based on new prediction-actual pair"""
        for model_name, prediction in model_predictions.items():
            error = abs(prediction - actual)
            likelihood = np.exp(-error)  # Higher likelihood for lower error
            
            if model_name not in self.model_likelihoods:
                self.model_likelihoods[model_name] = []
                self.model_priors[model_name] = 1.0
            
            self.model_likelihoods[model_name].append(likelihood)
            
            # Apply decay to historical likelihoods
            if len(self.model_likelihoods[model_name]) > 100:
                self.model_likelihoods[model_name] = self.model_likelihoods[model_name][-100:]
        
        # Store prediction history
        self.prediction_history.append(model_predictions.copy())
        self.actual_history.append(actual)
        
        # Update posteriors
        self._update_posteriors()
    
    def _update_posteriors(self):
        """Update posterior probabilities using Bayes' theorem"""
        total_evidence = 0
        model_evidences = {}
        
        for model_name in self.model_likelihoods:
            if self.model_likelihoods[model_name]:
                # Calculate weighted likelihood (with decay)
                likelihoods = np.array(self.model_likelihoods[model_name])
                weights = np.array([self.decay_factor ** i for i in range(len(likelihoods) - 1, -1, -1)])
                weighted_likelihood = np.average(likelihoods, weights=weights)
                
                # Evidence = Prior * Likelihood
                evidence = self.model_priors[model_name] * weighted_likelihood
                model_evidences[model_name] = evidence
                total_evidence += evidence
        
        # Calculate posteriors
        if total_evidence > 0:
            self.model_posteriors = {
                model: evidence / total_evidence 
                for model, evidence in model_evidences.items()
            }
        else:
            # Equal weights if no evidence
            n_models = len(self.model_likelihoods)
            self.model_posteriors = {
                model: 1.0 / n_models 
                for model in self.model_likelihoods
            }
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights based on Bayesian posteriors"""
        return self.model_posteriors.copy()

class AdaptiveEnsembleWeighting:
    """Adaptive ensemble weighting based on recent performance"""
    
    def __init__(self, window_size: int = 50, min_weight: float = 0.05):
        self.window_size = window_size
        self.min_weight = min_weight
        self.performance_history = {}
        self.weights = {}
    
    def update_performance(self, model_predictions: Dict[str, float], actual: float):
        """Update model performance tracking"""
        for model_name, prediction in model_predictions.items():
            error = abs(prediction - actual)
            
            if model_name not in self.performance_history:
                self.performance_history[model_name] = []
            
            self.performance_history[model_name].append(error)
            
            # Keep only recent performance
            if len(self.performance_history[model_name]) > self.window_size:
                self.performance_history[model_name] = self.performance_history[model_name][-self.window_size:]
        
        # Update weights
        self._calculate_adaptive_weights()
    
    def _calculate_adaptive_weights(self):
        """Calculate adaptive weights based on recent performance"""
        model_scores = {}
        
        for model_name, errors in self.performance_history.items():
            if errors:
                # Use inverse of average error as score
                avg_error = np.mean(errors)
                model_scores[model_name] = 1.0 / (1.0 + avg_error)
        
        if model_scores:
            # Normalize scores to get weights
            total_score = sum(model_scores.values())
            
            if total_score > 0:
                self.weights = {
                    model: max(self.min_weight, score / total_score)
                    for model, score in model_scores.items()
                }
                
                # Renormalize to ensure sum = 1
                total_weight = sum(self.weights.values())
                self.weights = {
                    model: weight / total_weight 
                    for model, weight in self.weights.items()
                }
            else:
                # Equal weights if all scores are zero
                n_models = len(model_scores)
                self.weights = {model: 1.0 / n_models for model in model_scores}
    
    def get_weights(self) -> Dict[str, float]:
        """Get current adaptive weights"""
        return self.weights.copy()

class ReinforcementLearningOptimizer:
    """Reinforcement Learning optimizer for ensemble strategy selection"""
    
    def __init__(self, n_strategies: int = 6, learning_rate: float = 0.1, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995):
        self.n_strategies = n_strategies
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        # Q-table for strategy selection (state -> action values)
        self.q_table = {}
        self.strategy_names = [
            'simple_average', 'weighted_average', 'bayesian_average',
            'adaptive_weighted', 'stacked', 'best_model'
        ]
        
        # Performance tracking
        self.performance_history = []
        self.action_history = []
        self.state_history = []
        self.reward_history = []
        
    def get_state(self, market_conditions: Dict[str, Any]) -> str:
        """Convert market conditions to a state representation"""
        try:
            # Create state based on market volatility, trend, and model agreement
            volatility = market_conditions.get('volatility', 0.5)
            trend_strength = market_conditions.get('trend_strength', 0.0)
            model_agreement = market_conditions.get('model_agreement', 0.5)
            
            # Discretize continuous variables
            vol_bucket = 'low' if volatility < 0.3 else 'med' if volatility < 0.7 else 'high'
            trend_bucket = 'down' if trend_strength < -0.3 else 'flat' if trend_strength < 0.3 else 'up'
            agree_bucket = 'low' if model_agreement < 0.4 else 'med' if model_agreement < 0.7 else 'high'
            
            return f"{vol_bucket}_{trend_bucket}_{agree_bucket}"
            
        except Exception:
            return "unknown"
    
    def select_strategy(self, state: str) -> str:
        """Select strategy using epsilon-greedy policy"""
        if state not in self.q_table:
            self.q_table[state] = {strategy: 0.0 for strategy in self.strategy_names}
        
        # Exploration vs exploitation
        if np.random.random() < self.epsilon:
            # Explore: random strategy
            selected_strategy = np.random.choice(self.strategy_names)
        else:
            # Exploit: best strategy for this state
            best_strategy = max(self.q_table[state].items(), key=lambda x: x[1])[0]
            selected_strategy = best_strategy
        
        # Store action
        self.action_history.append(selected_strategy)
        self.state_history.append(state)
        
        return selected_strategy
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str = None):
        """Update Q-value using Q-learning update rule"""
        if state not in self.q_table:
            self.q_table[state] = {strategy: 0.0 for strategy in self.strategy_names}
        
        # Current Q-value
        current_q = self.q_table[state][action]
        
        # Calculate target Q-value
        if next_state and next_state in self.q_table:
            # Q-learning: Q(s,a) += α[r + γ*max(Q(s',a')) - Q(s,a)]
            next_max_q = max(self.q_table[next_state].values())
            target_q = reward + 0.9 * next_max_q  # γ = 0.9 (discount factor)
        else:
            target_q = reward
        
        # Update Q-value
        updated_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state][action] = updated_q
        
        # Store reward
        self.reward_history.append(reward)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def calculate_reward(self, predicted_return: float, actual_return: float, 
                        base_model_error: float) -> float:
        """Calculate reward based on prediction accuracy improvement"""
        try:
            ensemble_error = abs(predicted_return - actual_return)
            
            # Reward is positive if ensemble beats base models
            if ensemble_error < base_model_error:
                # Reward proportional to improvement
                improvement = (base_model_error - ensemble_error) / base_model_error
                reward = min(1.0, improvement * 2.0)  # Max reward of 1.0
            else:
                # Penalty for worse performance
                degradation = (ensemble_error - base_model_error) / base_model_error
                reward = max(-1.0, -degradation)  # Max penalty of -1.0
            
            return reward
            
        except Exception:
            return 0.0
    
    def get_strategy_preferences(self) -> Dict[str, float]:
        """Get current strategy preferences based on Q-values"""
        if not self.q_table:
            return {strategy: 1.0/len(self.strategy_names) for strategy in self.strategy_names}
        
        # Average Q-values across all states
        strategy_totals = {strategy: 0.0 for strategy in self.strategy_names}
        strategy_counts = {strategy: 0 for strategy in self.strategy_names}
        
        for state_q_values in self.q_table.values():
            for strategy, q_value in state_q_values.items():
                strategy_totals[strategy] += q_value
                strategy_counts[strategy] += 1
        
        # Calculate average Q-values
        avg_q_values = {}
        for strategy in self.strategy_names:
            if strategy_counts[strategy] > 0:
                avg_q_values[strategy] = strategy_totals[strategy] / strategy_counts[strategy]
            else:
                avg_q_values[strategy] = 0.0
        
        # Convert to probabilities using softmax
        max_q = max(avg_q_values.values())
        exp_q_values = {strategy: np.exp(q - max_q) for strategy, q in avg_q_values.items()}
        total_exp = sum(exp_q_values.values())
        
        preferences = {strategy: exp_q / total_exp for strategy, exp_q in exp_q_values.items()}
        return preferences

class MetaLearner:
    """Meta-learner for stacking ensemble"""
    
    def __init__(self, base_models: List[str], meta_model=None):
        self.base_models = base_models
        self.meta_model = meta_model or Ridge(alpha=0.1)
        self.is_trained = False
        self.feature_names = []
    
    def prepare_meta_features(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Prepare features for meta-learner"""
        meta_features = []
        self.feature_names = []
        
        for model_name in self.base_models:
            if model_name in base_predictions:
                preds = base_predictions[model_name]
                if isinstance(preds, (list, np.ndarray)) and len(preds) > 0:
                    meta_features.append(np.array(preds).reshape(-1, 1))
                    self.feature_names.append(f"{model_name}_pred")
                    
                    # Add derived features
                    if len(preds) > 1:
                        # Moving average of predictions
                        ma_preds = pd.Series(preds).rolling(min(5, len(preds))).mean().fillna(preds[0])
                        meta_features.append(ma_preds.values.reshape(-1, 1))
                        self.feature_names.append(f"{model_name}_ma_pred")
                        
                        # Prediction confidence (inverse of recent volatility)
                        pred_vol = pd.Series(preds).rolling(min(10, len(preds))).std().fillna(0)
                        confidence = 1.0 / (1.0 + pred_vol)
                        meta_features.append(confidence.values.reshape(-1, 1))
                        self.feature_names.append(f"{model_name}_confidence")
        
        if meta_features:
            return np.hstack(meta_features)
        else:
            return np.array([]).reshape(0, 0)
    
    def train(self, base_predictions: Dict[str, np.ndarray], targets: np.ndarray) -> Dict[str, Any]:
        """Train meta-learner"""
        try:
            X_meta = self.prepare_meta_features(base_predictions)
            
            if X_meta.size == 0:
                return {'success': False, 'error': 'No meta-features could be created'}
            
            # Align with targets
            min_len = min(len(X_meta), len(targets))
            X_meta = X_meta[:min_len]
            y_meta = targets[:min_len]
            
            if len(X_meta) < 10:
                return {'success': False, 'error': 'Insufficient data for meta-learner training'}
            
            # Train meta-model
            self.meta_model.fit(X_meta, y_meta)
            
            # Evaluate
            meta_predictions = self.meta_model.predict(X_meta)
            r2 = r2_score(y_meta, meta_predictions)
            mse = mean_squared_error(y_meta, meta_predictions)
            
            self.is_trained = True
            
            logger.info(f"Meta-learner trained - R2: {r2:.4f}, MSE: {mse:.6f}")
            
            return {
                'success': True,
                'r2_score': r2,
                'mse': mse,
                'n_features': X_meta.shape[1],
                'n_samples': len(X_meta),
                'feature_names': self.feature_names
            }
            
        except Exception as e:
            logger.error(f"Meta-learner training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, base_predictions: Dict[str, Union[float, List[float]]]) -> float:
        """Make meta-prediction"""
        if not self.is_trained:
            return 0.0
        
        try:
            # Convert single predictions to arrays for consistency
            pred_arrays = {}
            for model_name, preds in base_predictions.items():
                if isinstance(preds, (int, float)):
                    pred_arrays[model_name] = np.array([preds])
                else:
                    pred_arrays[model_name] = np.array(preds)
            
            X_meta = self.prepare_meta_features(pred_arrays)
            
            if X_meta.size == 0:
                return 0.0
            
            # Use only the last row for prediction
            meta_pred = self.meta_model.predict(X_meta[-1:])
            return float(meta_pred[0])
            
        except Exception as e:
            logger.error(f"Meta-prediction failed: {e}")
            return 0.0

class EnsemblePredictor:
    """Advanced ensemble predictor combining multiple ML approaches"""
    
    def __init__(self, config: Dict[str, Any] = None, traditional_ml=None, transformer_ml=None):
        self.config = config or {}
        
        # Initialize base predictors (use provided ones or create new)
        if traditional_ml is not None:
            self.traditional_ml = traditional_ml
        else:
            self.traditional_ml = TraditionalMLPredictor(config.get('traditional_ml', {}) if isinstance(config, dict) else {})
            
        if transformer_ml is not None:
            self.transformer_ml = transformer_ml
        else:
            self.transformer_ml = TransformerMLPredictor(config.get('transformer_ml', {}) if isinstance(config, dict) else {})
        
        # Ensemble methods
        self.bayesian_averaging = BayesianModelAveraging()
        self.adaptive_weighting = AdaptiveEnsembleWeighting()
        self.meta_learner = None
        self.rl_optimizer = ReinforcementLearningOptimizer()
        
        # Performance tracking
        self.ensemble_performance = {}
        self.prediction_history = []
        self.actual_history = []
        
        # Training status
        self.is_trained = False
        self.training_history = []
        
        # Ensemble strategies
        self.ensemble_strategies = {
            'simple_average': self._simple_average,
            'weighted_average': self._weighted_average,
            'bayesian_average': self._bayesian_average,
            'adaptive_weighted': self._adaptive_weighted,
            'stacked': self._stacked_ensemble,
            'best_model': self._best_model_selection
        }
        
        if isinstance(self.config, dict):
            self.active_strategies = self.config.get(
                'active_strategies', 
                ['weighted_average', 'bayesian_average', 'adaptive_weighted']
            )
        else:
            self.active_strategies = getattr(
                self.config, 'active_strategies',
                ['weighted_average', 'bayesian_average', 'adaptive_weighted']
            )
    
    async def train(self, data: pd.DataFrame, target_column: str = 'return_5d') -> Dict[str, Any]:
        """Train all base models and ensemble methods"""
        logger.info("Starting comprehensive ensemble training...")
        
        training_results = {
            'base_models': {},
            'ensemble_methods': {},
            'success': False
        }
        
        try:
            # Train base models
            logger.info("Training Traditional ML models...")
            trad_results = await self.traditional_ml.train(data, target_column)
            training_results['base_models']['traditional_ml'] = trad_results
            
            logger.info("Training Transformer ML models...")
            trans_results = await self.transformer_ml.train(data, target_column)
            training_results['base_models']['transformer_ml'] = trans_results
            
            # Check if at least one model trained successfully
            if not (trad_results.get('success', False) or trans_results.get('success', False)):
                return {
                    'success': False, 
                    'error': 'No base models trained successfully',
                    'results': training_results
                }
            
            # Prepare data for meta-learning
            if 'Close' in data.columns:
                # Create validation set for ensemble training
                val_data = data.tail(min(len(data) // 4, 200))  # Use last 25% or max 200 points
                
                base_predictions = {}
                targets = []
                
                # Get predictions from base models on validation data
                for idx in range(len(val_data) - 10):
                    sample_data = val_data.iloc[:idx + 10]
                    
                    sample_predictions = {}
                    
                    # Traditional ML predictions
                    if trad_results.get('success', False):
                        trad_pred = await self.traditional_ml.predict(sample_data)
                        if trad_pred.get('success', False):
                            sample_predictions['traditional_ml'] = trad_pred['ensemble_prediction']
                    
                    # Transformer ML predictions
                    if trans_results.get('success', False):
                        trans_pred = await self.transformer_ml.predict(sample_data)
                        if trans_pred.get('success', False):
                            sample_predictions['transformer_ml'] = trans_pred['ensemble_prediction']
                    
                    # Store predictions and actual target
                    if sample_predictions:
                        for model_name, pred in sample_predictions.items():
                            if model_name not in base_predictions:
                                base_predictions[model_name] = []
                            base_predictions[model_name].append(pred)
                        
                        # Get actual future return
                        if target_column in val_data.columns:
                            actual = val_data.iloc[idx + 10][target_column]
                        else:
                            actual = (val_data.iloc[idx + 15]['Close'] / val_data.iloc[idx + 10]['Close'] - 1) if idx + 15 < len(val_data) else 0
                        
                        targets.append(actual)
                
                # Train meta-learner if we have enough data
                if base_predictions and len(targets) > 10:
                    model_names = list(base_predictions.keys())
                    self.meta_learner = MetaLearner(model_names)
                    
                    meta_results = self.meta_learner.train(base_predictions, np.array(targets))
                    training_results['ensemble_methods']['meta_learner'] = meta_results
                    
                    # Initialize ensemble methods with historical data
                    for i, actual in enumerate(targets):
                        sample_preds = {model: preds[i] for model, preds in base_predictions.items() if i < len(preds)}
                        if sample_preds:
                            self.bayesian_averaging.update_model_performance(sample_preds, actual)
                            self.adaptive_weighting.update_performance(sample_preds, actual)
            
            self.is_trained = True
            
            # Store training information
            training_info = {
                'timestamp': datetime.now(),
                'target_column': target_column,
                'base_models_trained': [
                    name for name, results in training_results['base_models'].items() 
                    if results.get('success', False)
                ],
                'ensemble_methods': list(training_results['ensemble_methods'].keys()),
                'active_strategies': self.active_strategies
            }
            self.training_history.append(training_info)
            
            training_results['success'] = True
            training_results['training_info'] = training_info
            
            logger.info("Ensemble training completed successfully!")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            return {'success': False, 'error': str(e), 'results': training_results}
    
    async def predict(self, symbol: str = None, market_data: pd.DataFrame = None, 
                    news_data: List[Dict] = None, social_data: Dict = None,
                    multi_timeframe_data: Dict = None, region: str = None, 
                    data: pd.DataFrame = None) -> Dict[str, Any]:
        """Make comprehensive ensemble predictions"""
        if not self.is_trained:
            return {'error': 'Ensemble not trained yet'}
        
        # Use provided data or fallback to legacy parameter
        input_data = market_data if market_data is not None else data
        if input_data is None:
            return {'error': 'No market data provided'}
        
        try:
            # Get predictions from base models with all available data
            base_predictions = {}
            
            # Traditional ML prediction
            if self.traditional_ml.is_trained:
                trad_pred = await self.traditional_ml.predict(input_data)
                if trad_pred.get('success', False):
                    base_predictions['traditional_ml'] = {
                        'prediction': trad_pred['ensemble_prediction'],
                        'confidence': trad_pred.get('ensemble_confidence', 0.5),
                        'individual_predictions': trad_pred.get('predictions', {})
                    }
            
            # Transformer ML prediction
            if self.transformer_ml.is_trained:
                trans_pred = await self.transformer_ml.predict(input_data)
                if trans_pred.get('success', False):
                    base_predictions['transformer_ml'] = {
                        'prediction': trans_pred['ensemble_prediction'],
                        'confidence': trans_pred.get('ensemble_confidence', 0.5),
                        'individual_predictions': trans_pred.get('predictions', {})
                    }
            
            # Integrate multi-timeframe analysis if available
            mtf_boost = 0.0
            mtf_confidence_boost = 0.0
            
            if multi_timeframe_data:
                mtf_score = multi_timeframe_data.get('composite_score', 0.0)
                mtf_confidence = multi_timeframe_data.get('confidence', 0.5)
                mtf_signal = multi_timeframe_data.get('signal', 'HOLD')
                
                # Apply multi-timeframe boost to base predictions
                if mtf_signal in ['BUY', 'STRONG_BUY'] and mtf_score > 0.3:
                    mtf_boost = mtf_score * 0.3  # Up to 30% boost for strong signals
                    mtf_confidence_boost = mtf_confidence * 0.2  # Up to 20% confidence boost
                elif mtf_signal in ['SELL', 'STRONG_SELL'] and mtf_score < -0.3:
                    mtf_boost = mtf_score * 0.3  # Negative boost for sell signals
                    mtf_confidence_boost = mtf_confidence * 0.2
                
                # Apply boost to base predictions
                for model_name in base_predictions:
                    base_predictions[model_name]['prediction'] += mtf_boost
                    base_predictions[model_name]['confidence'] = min(1.0, 
                        base_predictions[model_name]['confidence'] + mtf_confidence_boost)
                    
                logger.debug(f"Applied MTF boost: {mtf_boost:.4f}, confidence boost: {mtf_confidence_boost:.4f}")
            
            # Integrate news sentiment if available
            news_sentiment_adjustment = 0.0
            if news_data:
                positive_news = sum(1 for item in news_data 
                                  if item.get('sentiment_data', {}).get('sentiment_score', 0) > 0.1)
                negative_news = sum(1 for item in news_data 
                                  if item.get('sentiment_data', {}).get('sentiment_score', 0) < -0.1)
                
                if positive_news > negative_news:
                    news_sentiment_adjustment = min(0.05, (positive_news - negative_news) * 0.01)
                elif negative_news > positive_news:
                    news_sentiment_adjustment = max(-0.05, (positive_news - negative_news) * 0.01)
                
                # Apply news adjustment
                for model_name in base_predictions:
                    base_predictions[model_name]['prediction'] += news_sentiment_adjustment
            
            # Integrate social sentiment if available
            social_sentiment_adjustment = 0.0
            if social_data:
                sentiment_score = social_data.get('sentiment_score', 0.0)
                social_confidence = social_data.get('confidence', 0.5)
                
                if abs(sentiment_score) > 0.1 and social_confidence > 0.6:
                    social_sentiment_adjustment = sentiment_score * 0.05  # Up to 5% adjustment
                    
                    for model_name in base_predictions:
                        base_predictions[model_name]['prediction'] += social_sentiment_adjustment
            
            if not base_predictions:
                return {'error': 'No base model predictions available'}
            
            # Determine market conditions for RL state
            market_conditions = self._assess_market_conditions(input_data, base_predictions, 
                                                             multi_timeframe_data)
            
            # Use RL to select optimal strategy
            current_state = self.rl_optimizer.get_state(market_conditions)
            optimal_strategy = self.rl_optimizer.select_strategy(current_state)
            
            # Apply ensemble strategies (prioritizing RL-selected strategy)
            ensemble_predictions = {}
            
            # Always include the RL-selected strategy
            if optimal_strategy in self.ensemble_strategies:
                try:
                    pred_result = self.ensemble_strategies[optimal_strategy](base_predictions)
                    pred_result['selected_by_rl'] = True
                    ensemble_predictions[optimal_strategy] = pred_result
                except Exception as e:
                    logger.warning(f"RL-selected strategy {optimal_strategy} failed: {e}")
            
            # Include other active strategies for comparison
            for strategy_name in self.active_strategies:
                if strategy_name != optimal_strategy and strategy_name in self.ensemble_strategies:
                    try:
                        pred_result = self.ensemble_strategies[strategy_name](base_predictions)
                        pred_result['selected_by_rl'] = False
                        ensemble_predictions[strategy_name] = pred_result
                    except Exception as e:
                        logger.warning(f"Ensemble strategy {strategy_name} failed: {e}")
                        ensemble_predictions[strategy_name] = {'prediction': 0.0, 'confidence': 0.0, 'selected_by_rl': False}
            
            # Meta-ensemble: prioritize RL-selected strategy
            if len(ensemble_predictions) > 1:
                # Give extra weight to RL-selected strategy
                rl_prediction = None
                rl_confidence = None
                other_predictions = []
                other_confidences = []
                
                for strategy, pred in ensemble_predictions.items():
                    if pred.get('selected_by_rl', False):
                        rl_prediction = pred['prediction']
                        rl_confidence = pred.get('confidence', 0.5)
                    else:
                        other_predictions.append(pred['prediction'])
                        other_confidences.append(pred.get('confidence', 0.5))
                
                if rl_prediction is not None and other_predictions:
                    # Weighted combination: 70% RL strategy, 30% others
                    other_avg = np.mean(other_predictions)
                    meta_prediction = 0.7 * rl_prediction + 0.3 * other_avg
                    meta_confidence = 0.7 * rl_confidence + 0.3 * np.mean(other_confidences)
                elif rl_prediction is not None:
                    meta_prediction = rl_prediction
                    meta_confidence = rl_confidence
                else:
                    # Fallback to simple average
                    meta_prediction = np.mean([pred['prediction'] for pred in ensemble_predictions.values()])
                    meta_confidence = np.mean([pred.get('confidence', 0.5) for pred in ensemble_predictions.values()])
            else:
                first_ensemble = list(ensemble_predictions.values())[0]
                meta_prediction = first_ensemble['prediction']
                meta_confidence = first_ensemble.get('confidence', 0.5)
            
            # Generate trading signal
            signal_strength = abs(meta_prediction) * meta_confidence
            
            if meta_prediction > 0.01 and signal_strength > 0.02:
                signal_type = 'buy'
            elif meta_prediction < -0.01 and signal_strength > 0.02:
                signal_type = 'sell'
            else:
                signal_type = 'hold'
            
            # Compile comprehensive results
            result = {
                'success': True,
                'base_predictions': base_predictions,
                'ensemble_predictions': ensemble_predictions,
                'meta_prediction': float(meta_prediction),
                'meta_confidence': float(meta_confidence),
                'signal_type': signal_type,
                'signal_strength': float(signal_strength),
                'active_strategies': self.active_strategies,
                'timestamp': datetime.now(),
                'model_weights': {
                    'bayesian': self.bayesian_averaging.get_model_weights(),
                    'adaptive': self.adaptive_weighting.get_weights()
                }
            }
            
            # Store prediction for future ensemble improvement
            pred_for_history = {
                model: data['prediction'] for model, data in base_predictions.items()
            }
            self.prediction_history.append({
                'predictions': pred_for_history,
                'ensemble_result': meta_prediction,
                'timestamp': datetime.now()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            return {'error': str(e)}
    
    def _simple_average(self, base_predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Simple average ensemble"""
        predictions = [data['prediction'] for data in base_predictions.values()]
        avg_prediction = np.mean(predictions)
        avg_confidence = np.mean([data.get('confidence', 0.5) for data in base_predictions.values()])
        
        return {
            'prediction': float(avg_prediction),
            'confidence': float(avg_confidence),
            'method': 'simple_average'
        }
    
    def _weighted_average(self, base_predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Confidence-weighted average ensemble"""
        total_weight = 0
        weighted_prediction = 0
        
        for model_data in base_predictions.values():
            weight = model_data.get('confidence', 0.5)
            weighted_prediction += model_data['prediction'] * weight
            total_weight += weight
        
        if total_weight > 0:
            final_prediction = weighted_prediction / total_weight
            avg_confidence = total_weight / len(base_predictions)
        else:
            final_prediction = np.mean([data['prediction'] for data in base_predictions.values()])
            avg_confidence = 0.5
        
        return {
            'prediction': float(final_prediction),
            'confidence': float(avg_confidence),
            'method': 'weighted_average'
        }
    
    def _bayesian_average(self, base_predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Bayesian model averaging ensemble"""
        model_weights = self.bayesian_averaging.get_model_weights()
        
        if not model_weights:
            return self._simple_average(base_predictions)
        
        weighted_prediction = 0
        total_weight = 0
        
        for model_name, model_data in base_predictions.items():
            weight = model_weights.get(model_name, 0)
            weighted_prediction += model_data['prediction'] * weight
            total_weight += weight
        
        if total_weight > 0:
            final_prediction = weighted_prediction / total_weight
        else:
            final_prediction = np.mean([data['prediction'] for data in base_predictions.values()])
        
        # Confidence based on weight distribution entropy
        if model_weights:
            weights = list(model_weights.values())
            entropy = -sum(w * np.log(w + 1e-10) for w in weights if w > 0)
            max_entropy = np.log(len(weights))
            confidence = 1 - (entropy / max_entropy if max_entropy > 0 else 0)
        else:
            confidence = 0.5
        
        return {
            'prediction': float(final_prediction),
            'confidence': float(confidence),
            'method': 'bayesian_average',
            'model_weights': model_weights
        }
    
    def _adaptive_weighted(self, base_predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Adaptive weighted ensemble"""
        adaptive_weights = self.adaptive_weighting.get_weights()
        
        if not adaptive_weights:
            return self._weighted_average(base_predictions)
        
        weighted_prediction = 0
        total_weight = 0
        
        for model_name, model_data in base_predictions.items():
            weight = adaptive_weights.get(model_name, 0)
            weighted_prediction += model_data['prediction'] * weight
            total_weight += weight
        
        if total_weight > 0:
            final_prediction = weighted_prediction / total_weight
        else:
            final_prediction = np.mean([data['prediction'] for data in base_predictions.values()])
        
        # Confidence based on weight concentration
        if adaptive_weights:
            max_weight = max(adaptive_weights.values())
            confidence = max_weight  # Higher if one model dominates
        else:
            confidence = 0.5
        
        return {
            'prediction': float(final_prediction),
            'confidence': float(confidence),
            'method': 'adaptive_weighted',
            'adaptive_weights': adaptive_weights
        }
    
    def _stacked_ensemble(self, base_predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Stacked ensemble using meta-learner"""
        if not self.meta_learner or not self.meta_learner.is_trained:
            return self._weighted_average(base_predictions)
        
        # Prepare predictions for meta-learner
        model_preds = {model: data['prediction'] for model, data in base_predictions.items()}
        
        meta_prediction = self.meta_learner.predict(model_preds)
        
        # Confidence based on meta-learner performance
        confidence = 0.7  # Default confidence for stacked ensemble
        
        return {
            'prediction': float(meta_prediction),
            'confidence': float(confidence),
            'method': 'stacked'
        }
    
    def _best_model_selection(self, base_predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Select best performing model dynamically"""
        adaptive_weights = self.adaptive_weighting.get_weights()
        
        if adaptive_weights:
            best_model = max(adaptive_weights.items(), key=lambda x: x[1])[0]
        else:
            # Fallback to highest confidence model
            best_model = max(
                base_predictions.items(), 
                key=lambda x: x[1].get('confidence', 0)
            )[0]
        
        if best_model in base_predictions:
            best_prediction = base_predictions[best_model]
            return {
                'prediction': best_prediction['prediction'],
                'confidence': best_prediction.get('confidence', 0.5),
                'method': 'best_model',
                'selected_model': best_model
            }
        else:
            return self._simple_average(base_predictions)
    
    def update_with_actual(self, actual_value: float):
        """Update ensemble with actual outcome for continuous learning"""
        if not self.prediction_history:
            return
        
        # Get the last prediction
        last_prediction = self.prediction_history[-1]
        
        # Update Bayesian averaging
        self.bayesian_averaging.update_model_performance(
            last_prediction['predictions'], actual_value
        )
        
        # Update adaptive weighting
        self.adaptive_weighting.update_performance(
            last_prediction['predictions'], actual_value
        )
        
        # Store actual value
        self.actual_history.append(actual_value)
        
        # Trigger adaptive learning if we have enough data
        self._adaptive_learning_update(actual_value)
        
        # Update reinforcement learning
        self._update_reinforcement_learning(actual_value)
        
        logger.info(f"Updated ensemble with actual value: {actual_value}")
    
    def _update_reinforcement_learning(self, actual_value: float):
        """Update reinforcement learning based on actual results"""
        if len(self.prediction_history) < 2:
            return
        
        try:
            # Get the last two predictions for state transition
            last_prediction = self.prediction_history[-1]
            prev_prediction = self.prediction_history[-2] if len(self.prediction_history) > 1 else None
            
            # Calculate base model error for reward calculation
            base_errors = []
            if 'predictions' in last_prediction:
                for model_pred in last_prediction['predictions'].values():
                    base_errors.append(abs(model_pred - actual_value))
            
            avg_base_error = np.mean(base_errors) if base_errors else 0.1
            ensemble_pred = last_prediction.get('ensemble_result', 0)
            
            # Calculate reward
            reward = self.rl_optimizer.calculate_reward(ensemble_pred, actual_value, avg_base_error)
            
            # Update Q-values if we have state and action history
            if (hasattr(self.rl_optimizer, 'state_history') and 
                hasattr(self.rl_optimizer, 'action_history') and
                self.rl_optimizer.state_history and 
                self.rl_optimizer.action_history):
                
                current_state = self.rl_optimizer.state_history[-1]
                current_action = self.rl_optimizer.action_history[-1]
                
                # Determine next state (if we have previous prediction for comparison)
                next_state = None
                if prev_prediction and 'market_conditions' in last_prediction:
                    next_state = self.rl_optimizer.get_state(last_prediction['market_conditions'])
                
                # Update Q-value
                self.rl_optimizer.update_q_value(current_state, current_action, reward, next_state)
                
                logger.debug(f"RL Update - State: {current_state}, Action: {current_action}, "
                           f"Reward: {reward:.4f}, Epsilon: {self.rl_optimizer.epsilon:.4f}")
        
        except Exception as e:
            logger.warning(f"Reinforcement learning update failed: {e}")
    
    def _adaptive_learning_update(self, actual_value: float):
        """Perform adaptive learning to improve ensemble strategies"""
        if len(self.actual_history) < 20:
            return  # Need at least 20 data points
        
        try:
            # Evaluate recent performance of each strategy
            recent_predictions = self.prediction_history[-10:]
            recent_actuals = self.actual_history[-10:]
            
            if len(recent_predictions) != len(recent_actuals):
                return
            
            strategy_performance = {}
            
            # Calculate performance for each active strategy (simulate)
            for strategy_name in self.active_strategies:
                if strategy_name in self.ensemble_strategies:
                    strategy_errors = []
                    
                    for i, pred_data in enumerate(recent_predictions):
                        if i < len(recent_actuals):
                            # Simulate strategy prediction (simplified)
                            if strategy_name == 'simple_average':
                                strategy_pred = np.mean(list(pred_data['predictions'].values()))
                            elif strategy_name == 'weighted_average':
                                strategy_pred = pred_data.get('ensemble_result', 0)
                            else:
                                strategy_pred = pred_data.get('ensemble_result', 0)
                            
                            error = abs(strategy_pred - recent_actuals[i])
                            strategy_errors.append(error)
                    
                    if strategy_errors:
                        avg_error = np.mean(strategy_errors)
                        strategy_performance[strategy_name] = 1.0 / (1.0 + avg_error)  # Higher is better
            
            # Adaptive strategy selection based on performance
            if strategy_performance:
                best_strategies = sorted(
                    strategy_performance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]  # Keep top 3 strategies
                
                new_active_strategies = [strategy[0] for strategy in best_strategies]
                
                # Only update if there's a significant change
                if set(new_active_strategies) != set(self.active_strategies[:3]):
                    logger.info(f"Adapting strategies from {self.active_strategies} to {new_active_strategies}")
                    self.active_strategies = new_active_strategies
            
        except Exception as e:
            logger.warning(f"Adaptive learning update failed: {e}")
    
    def _assess_market_conditions(self, market_data: pd.DataFrame, 
                                base_predictions: Dict, multi_timeframe_data: Dict) -> Dict[str, Any]:
        """Assess current market conditions for RL state determination"""
        try:
            conditions = {}
            
            # Calculate volatility from price data
            if not market_data.empty and 'Close' in market_data.columns:
                returns = market_data['Close'].pct_change().dropna()
                if len(returns) > 5:
                    conditions['volatility'] = float(returns.std())
                else:
                    conditions['volatility'] = 0.5
            else:
                conditions['volatility'] = 0.5
            
            # Assess trend strength from multi-timeframe data
            if multi_timeframe_data:
                mtf_score = multi_timeframe_data.get('composite_score', 0.0)
                conditions['trend_strength'] = float(mtf_score)
            else:
                conditions['trend_strength'] = 0.0
            
            # Calculate model agreement (how similar are base model predictions)
            if base_predictions and len(base_predictions) > 1:
                predictions = [data['prediction'] for data in base_predictions.values()]
                pred_std = np.std(predictions)
                pred_mean = np.mean(predictions)
                
                # Agreement is inverse of coefficient of variation
                if pred_mean != 0:
                    cv = abs(pred_std / pred_mean)
                    conditions['model_agreement'] = max(0.0, 1.0 - cv)
                else:
                    conditions['model_agreement'] = 1.0 if pred_std < 0.01 else 0.5
            else:
                conditions['model_agreement'] = 0.5
            
            return conditions
            
        except Exception as e:
            logger.warning(f"Market condition assessment failed: {e}")
            return {
                'volatility': 0.5,
                'trend_strength': 0.0,
                'model_agreement': 0.5
            }
    
    def get_ensemble_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive ensemble diagnostics for dashboard"""
        diagnostics = {
            'is_trained': self.is_trained,
            'active_strategies': self.active_strategies,
            'ensemble_weights': [],
            'model_details': []
        }

        if not self.is_trained:
            return diagnostics

        # 1. Get main ensemble weights (meta-learner level)
        bayesian_weights = self.bayesian_averaging.get_model_weights()
        if bayesian_weights:
            for model_name, weight in bayesian_weights.items():
                diagnostics['ensemble_weights'].append({
                    'name': model_name.replace('_', ' ').title(),
                    'weight': weight,
                    'description': f'Weight for the {model_name} sub-ensemble based on Bayesian averaging.'
                })

        # 2. Get details from Traditional ML predictor
        if self.traditional_ml and self.traditional_ml.is_trained:
            trad_diags = self.traditional_ml.get_model_diagnostics()
            trad_weights = trad_diags.get('ensemble_weights', {})
            for model_name, weight in trad_weights.items():
                perf = trad_diags.get('model_performance', {}).get(model_name, {})
                diagnostics['model_details'].append({
                    'name': model_name.replace('_', ' ').title(),
                    'group': 'Traditional ML',
                    'prediction': perf.get('latest_prediction', 0.0), # Placeholder, needs to be implemented
                    'confidence': weight,
                    'description': f'A {model_name} model. Performance (CV R2): {perf.get("cv_r2_mean", 0):.3f}'
                })

        # 3. Get details from Transformer ML predictor
        if self.transformer_ml and self.transformer_ml.is_trained:
            trans_diags = self.transformer_ml.get_model_diagnostics()
            trans_weights = trans_diags.get('model_weights', {}) # Assuming this method will be added
            if not trans_weights:
                # Fallback if weights not directly available
                perfs = trans_diags.get('model_performance', {})
                total_r2 = sum(max(0, p.get('final_r2', 0)) for p in perfs.values())
                if total_r2 > 0:
                    trans_weights = {name: max(0, p.get('final_r2', 0)) / total_r2 for name, p in perfs.items()}

            for model_name, weight in trans_weights.items():
                perf = trans_diags.get('model_performance', {}).get(model_name, {})
                diagnostics['model_details'].append({
                    'name': model_name.replace('_', ' ').title(),
                    'group': 'Transformer ML',
                    'prediction': perf.get('latest_prediction', 0.0), # Placeholder
                    'confidence': weight,
                    'description': f'A {model_name} model. Performance (Val R2): {perf.get("final_r2", 0):.3f}'
                })

        return diagnostics
    
    def save_ensemble(self, filepath_base: str) -> bool:
        """Save complete ensemble"""
        try:
            # Save base models
            trad_success = self.traditional_ml.save_model(f"{filepath_base}_traditional.pkl")
            trans_success = self.transformer_ml.save_models(f"{filepath_base}_transformer")
            
            # Save ensemble-specific data
            ensemble_data = {
                'config': self.config,
                'is_trained': self.is_trained,
                'training_history': self.training_history,
                'prediction_history': self.prediction_history[-100:],  # Keep last 100
                'actual_history': self.actual_history[-100:],
                'active_strategies': self.active_strategies,
                'bayesian_averaging': {
                    'model_priors': self.bayesian_averaging.model_priors,
                    'model_posteriors': self.bayesian_averaging.model_posteriors,
                    'decay_factor': self.bayesian_averaging.decay_factor
                },
                'adaptive_weighting': {
                    'weights': self.adaptive_weighting.weights,
                    'window_size': self.adaptive_weighting.window_size,
                    'min_weight': self.adaptive_weighting.min_weight
                }
            }
            
            # Save meta-learner if available
            if self.meta_learner:
                joblib.dump(self.meta_learner, f"{filepath_base}_meta_learner.pkl")
                ensemble_data['has_meta_learner'] = True
            
            joblib.dump(ensemble_data, f"{filepath_base}_ensemble.pkl")
            
            logger.info(f"Ensemble saved to {filepath_base}_*")
            return trad_success and trans_success
            
        except Exception as e:
            logger.error(f"Failed to save ensemble: {e}")
            return False
    
    def load_ensemble(self, filepath_base: str) -> bool:
        """Load complete ensemble"""
        try:
            # Load ensemble data
            ensemble_data = joblib.load(f"{filepath_base}_ensemble.pkl")
            
            self.config = ensemble_data.get('config', {})
            self.is_trained = ensemble_data.get('is_trained', False)
            self.training_history = ensemble_data.get('training_history', [])
            self.prediction_history = ensemble_data.get('prediction_history', [])
            self.actual_history = ensemble_data.get('actual_history', [])
            self.active_strategies = ensemble_data.get('active_strategies', ['weighted_average'])
            
            # Restore Bayesian averaging
            bma_data = ensemble_data.get('bayesian_averaging', {})
            self.bayesian_averaging.model_priors = bma_data.get('model_priors', {})
            self.bayesian_averaging.model_posteriors = bma_data.get('model_posteriors', {})
            self.bayesian_averaging.decay_factor = bma_data.get('decay_factor', 0.95)
            
            # Restore adaptive weighting
            aw_data = ensemble_data.get('adaptive_weighting', {})
            self.adaptive_weighting.weights = aw_data.get('weights', {})
            self.adaptive_weighting.window_size = aw_data.get('window_size', 50)
            self.adaptive_weighting.min_weight = aw_data.get('min_weight', 0.05)
            
            # Load meta-learner if available
            if ensemble_data.get('has_meta_learner', False):
                self.meta_learner = joblib.load(f"{filepath_base}_meta_learner.pkl")
            
            # Load base models
            trad_success = self.traditional_ml.load_model(f"{filepath_base}_traditional.pkl")
            trans_success = self.transformer_ml.load_models(f"{filepath_base}_transformer")
            
            logger.info(f"Ensemble loaded from {filepath_base}_*")
            return self.is_trained
            
        except Exception as e:
            logger.error(f"Failed to load ensemble: {e}")
            return False