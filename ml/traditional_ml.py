"""
Advanced Traditional Machine Learning Predictor for Trading Bot
Implements multiple classical ML algorithms with comprehensive feature engineering
and advanced optimization techniques
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

import asyncio

# Scikit-learn imports
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    ExtraTreesRegressor, AdaBoostRegressor, VotingRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, HuberRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV,
    TimeSeriesSplit, validation_curve
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, median_absolute_error
)
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import talib

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineering:
    """Advanced feature engineering for financial data"""
    
    @staticmethod
    def create_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical indicators"""
        features = pd.DataFrame(index=data.index)
        
        if 'Close' not in data.columns:
            logger.warning("No 'Close' column found in data")
            return features
            
        try:
            # Debug data types
            logger.debug(f"Input data types: {data.dtypes.to_dict()}")
            logger.debug(f"Data shape: {data.shape}")
            
            # Clean data first - remove any inf/nan values
            clean_data = data.replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean_data) != len(data):
                logger.warning(f"Removed {len(data) - len(clean_data)} rows with invalid values")
                data = clean_data
                features = pd.DataFrame(index=data.index)
            
            # Ensure all arrays are float64 for TALib - with extra validation
            close_series = data['Close'].copy()
            high_series = data['High'].copy() if 'High' in data.columns else close_series.copy()
            low_series = data['Low'].copy() if 'Low' in data.columns else close_series.copy() 
            volume_series = data['Volume'].copy() if 'Volume' in data.columns else pd.Series(np.ones(len(close_series)), index=close_series.index)
            
            # Convert to float64 with validation
            close = pd.to_numeric(close_series, errors='coerce').astype(np.float64).values
            high = pd.to_numeric(high_series, errors='coerce').astype(np.float64).values
            low = pd.to_numeric(low_series, errors='coerce').astype(np.float64).values
            volume = pd.to_numeric(volume_series, errors='coerce').astype(np.float64).values
            
            # Final validation
            if not np.isfinite(close).all():
                logger.error("Close prices contain non-finite values after conversion")
                close = np.nan_to_num(close, nan=100.0, posinf=100.0, neginf=100.0).astype(np.float64)
            
            if not np.isfinite(high).all():
                high = np.nan_to_num(high, nan=close, posinf=close, neginf=close).astype(np.float64)
                
            if not np.isfinite(low).all():
                low = np.nan_to_num(low, nan=close, posinf=close, neginf=close).astype(np.float64)
                
            if not np.isfinite(volume).all():
                volume = np.nan_to_num(volume, nan=1000.0, posinf=1000.0, neginf=1000.0).astype(np.float64)
            
            logger.debug(f"Processed arrays - Close: {close.dtype}, High: {high.dtype}, Low: {low.dtype}, Volume: {volume.dtype}")
            logger.debug(f"Array lengths - Close: {len(close)}, High: {len(high)}, Low: {len(low)}, Volume: {len(volume)}")
            
        except Exception as e:
            logger.error(f"Error preparing data for TALib: {e}")
            return features
        
        # Moving Averages with error handling
        for period in [5, 10, 20, 50, 100, 200]:
            if len(close) > period:
                try:
                    sma = talib.SMA(close, timeperiod=period)
                    ema = talib.EMA(close, timeperiod=period)
                    
                    features[f'sma_{period}'] = sma
                    features[f'ema_{period}'] = ema
                    
                    # Safe division with zero handling
                    features[f'price_sma_{period}_ratio'] = np.where(sma != 0, close / sma, 1.0)
                    features[f'price_ema_{period}_ratio'] = np.where(ema != 0, close / ema, 1.0)
                except Exception as e:
                    logger.warning(f"Error calculating MA for period {period}: {e}")
                    # Fill with neutral values
                    features[f'sma_{period}'] = close
                    features[f'ema_{period}'] = close
                    features[f'price_sma_{period}_ratio'] = np.ones_like(close)
                    features[f'price_ema_{period}_ratio'] = np.ones_like(close)
        
        # Bollinger Bands
        for period in [20, 50]:
            if len(close) > period:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=period)
                features[f'bb_upper_{period}'] = bb_upper
                features[f'bb_lower_{period}'] = bb_lower
                features[f'bb_width_{period}'] = (bb_upper - bb_lower) / bb_middle
                features[f'bb_position_{period}'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # Momentum Indicators with comprehensive error handling
        momentum_indicators = [
            ('rsi_14', lambda: talib.RSI(close, timeperiod=14), 50.0),
            ('rsi_21', lambda: talib.RSI(close, timeperiod=21), 50.0),
            ('cci_14', lambda: talib.CCI(high, low, close, timeperiod=14), 0.0),
            ('williams_r', lambda: talib.WILLR(high, low, close, timeperiod=14), -50.0),
            ('momentum_10', lambda: talib.MOM(close, timeperiod=10), 0.0),
            ('roc_10', lambda: talib.ROC(close, timeperiod=10), 0.0),
        ]
        
        for indicator_name, calc_func, default_value in momentum_indicators:
            try:
                logger.debug(f"Calculating {indicator_name}")
                
                # Verify input arrays before calculation
                if indicator_name.startswith('cci') or indicator_name.startswith('williams'):
                    if not (high.dtype == np.float64 and low.dtype == np.float64 and close.dtype == np.float64):
                        raise ValueError(f"Invalid array types for {indicator_name}")
                    if not (np.isfinite(high).all() and np.isfinite(low).all() and np.isfinite(close).all()):
                        raise ValueError(f"Non-finite values in arrays for {indicator_name}")
                else:
                    if not (close.dtype == np.float64 and np.isfinite(close).all()):
                        raise ValueError(f"Invalid close array for {indicator_name}")
                
                result = calc_func()
                if result is None:
                    raise ValueError(f"TALib returned None for {indicator_name}")
                
                features[indicator_name] = result
                logger.debug(f"Successfully calculated {indicator_name}")
                
            except Exception as e:
                logger.error(f"Error calculating {indicator_name}: {e}")
                logger.error(f"Array info - High: {high.dtype}, Low: {low.dtype}, Close: {close.dtype}")
                features[indicator_name] = np.full_like(close, default_value, dtype=np.float64)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close)
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd_hist
        features['macd_signal_ratio'] = macd / macd_signal
        
        # Stochastic
        stoch_k, stoch_d = talib.STOCH(high, low, close)
        features['stoch_k'] = stoch_k
        features['stoch_d'] = stoch_d
        features['stoch_k_d_diff'] = stoch_k - stoch_d
        
        # Average Directional Index
        features['adx'] = talib.ADX(high, low, close, timeperiod=14)
        features['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        features['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        # Volatility Indicators
        features['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
        features['natr_14'] = talib.NATR(high, low, close, timeperiod=14)
        
        # Volume Indicators
        if 'Volume' in data.columns:
            features['obv'] = talib.OBV(close, volume)
            features['ad_line'] = talib.AD(high, low, close, volume)
            features['chaikin_osc'] = talib.ADOSC(high, low, close, volume)
            
            # Volume moving averages
            for period in [10, 20, 50]:
                if len(volume) > period:
                    features[f'volume_sma_{period}'] = talib.SMA(volume, timeperiod=period)
                    features[f'volume_ratio_{period}'] = volume / features[f'volume_sma_{period}']
        
        # Price patterns
        features['doji'] = talib.CDLDOJI(data['Open'].values, high, low, close) if 'Open' in data.columns else 0
        features['hammer'] = talib.CDLHAMMER(data['Open'].values, high, low, close) if 'Open' in data.columns else 0
        features['engulfing'] = talib.CDLENGULFING(data['Open'].values, high, low, close) if 'Open' in data.columns else 0
        
        return features
    
    @staticmethod
    def create_time_features(data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        features = pd.DataFrame(index=data.index)
        
        if isinstance(data.index, pd.DatetimeIndex):
            features['hour'] = data.index.hour
            features['day_of_week'] = data.index.dayofweek
            features['day_of_month'] = data.index.day
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            features['is_month_start'] = data.index.is_month_start.astype(int)
            features['is_month_end'] = data.index.is_month_end.astype(int)
            features['is_quarter_start'] = data.index.is_quarter_start.astype(int)
            features['is_quarter_end'] = data.index.is_quarter_end.astype(int)
        
        return features
    
    @staticmethod
    def create_statistical_features(data: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        features = pd.DataFrame(index=data.index)
        
        if 'Close' not in data.columns:
            return features
            
        close = data['Close']
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            if len(close) > window:
                features[f'rolling_mean_{window}'] = close.rolling(window).mean()
                features[f'rolling_std_{window}'] = close.rolling(window).std()
                features[f'rolling_skew_{window}'] = close.rolling(window).skew()
                # Use scipy.stats for kurtosis calculation (pandas rolling kurtosis was removed)
                try:
                    from scipy import stats
                    features[f'rolling_kurt_{window}'] = close.rolling(window).apply(lambda x: stats.kurtosis(x, nan_policy='omit') if len(x.dropna()) > 3 else 0.0, raw=False)
                except ImportError:
                    features[f'rolling_kurt_{window}'] = close.rolling(window).apply(lambda x: 0.0, raw=False)
                features[f'rolling_median_{window}'] = close.rolling(window).median()
                features[f'rolling_min_{window}'] = close.rolling(window).min()
                features[f'rolling_max_{window}'] = close.rolling(window).max()
                features[f'rolling_range_{window}'] = features[f'rolling_max_{window}'] - features[f'rolling_min_{window}']
        
        # Price position within rolling window
        for window in [20, 50]:
            if len(close) > window:
                roll_min = close.rolling(window).min()
                roll_max = close.rolling(window).max()
                features[f'price_position_{window}'] = (close - roll_min) / (roll_max - roll_min)
        
        # Returns
        for period in [1, 2, 3, 5, 10, 20]:
            features[f'return_{period}d'] = close.pct_change(period)
            features[f'log_return_{period}d'] = np.log(close / close.shift(period))
        
        # Volatility measures
        for window in [10, 20, 50]:
            if len(close) > window:
                returns = close.pct_change()
                features[f'volatility_{window}'] = returns.rolling(window).std()
                features[f'realized_volatility_{window}'] = np.sqrt(252) * returns.rolling(window).std()
        
        return features

class TraditionalMLPredictor:
    """Advanced traditional machine learning predictor with comprehensive features"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize models with optimized parameters
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=False,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            ),
            'ada_boost': AdaBoostRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            ),
            'svr': SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                epsilon=0.01
            ),
            'ridge': Ridge(
                alpha=1.0
            ),
            'elastic_net': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                max_iter=2000
            ),
            'huber': HuberRegressor(
                epsilon=1.35,
                max_iter=200
            ),
            'bayesian_ridge': BayesianRidge(),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        }
        
        # Scalers
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        
        self.best_scaler = None
        self.best_models = {}
        self.feature_selector = None
        self.feature_names = []
        self.selected_features = []
        self.is_trained = False
        self.training_history = []
        
        # Performance tracking
        self.model_performance = {}
        self.ensemble_weights = {}
        
        # Feature engineering
        self.feature_engineer = AdvancedFeatureEngineering()
    
    def create_comprehensive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        all_features = pd.DataFrame(index=data.index)
        
        # Technical indicators
        tech_features = self.feature_engineer.create_technical_indicators(data)
        all_features = pd.concat([all_features, tech_features], axis=1)
        
        # Time features
        time_features = self.feature_engineer.create_time_features(data)
        all_features = pd.concat([all_features, time_features], axis=1)
        
        # Statistical features
        stat_features = self.feature_engineer.create_statistical_features(data)
        all_features = pd.concat([all_features, stat_features], axis=1)
        
        # Clean features
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        all_features = all_features.fillna(method='ffill').fillna(0)
        
        # Ensure all features are numeric
        for col in all_features.columns:
            all_features[col] = pd.to_numeric(all_features[col], errors='coerce').fillna(0)
        
        logger.info(f"Created {len(all_features.columns)} features")
        return all_features
    
    def create_target_variables(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Create multiple target variables for different prediction horizons"""
        targets = {}
        
        if 'Close' not in data.columns:
            return targets
        
        close = data['Close']
        
        # Future returns for different horizons
        for horizon in [1, 3, 5, 10, 20]:
            targets[f'return_{horizon}d'] = close.shift(-horizon) / close - 1
            targets[f'log_return_{horizon}d'] = np.log(close.shift(-horizon) / close)
        
        # Direction prediction (binary)
        for horizon in [1, 3, 5]:
            targets[f'direction_{horizon}d'] = (close.shift(-horizon) > close).astype(int)
        
        # Volatility prediction
        for horizon in [5, 10, 20]:
            returns = close.pct_change()
            targets[f'volatility_{horizon}d'] = returns.rolling(horizon).std().shift(-horizon)
        
        return targets
    
    def select_best_scaler(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Select best scaler based on cross-validation"""
        best_score = -np.inf
        best_scaler_name = 'standard'
        
        # Use a simple model for scaler selection
        simple_model = Ridge(alpha=1.0)
        
        for scaler_name, scaler in self.scalers.items():
            try:
                X_scaled = scaler.fit_transform(X)
                scores = cross_val_score(simple_model, X_scaled, y, cv=3, scoring='r2')
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_scaler_name = scaler_name
                    
                logger.info(f"Scaler {scaler_name}: R2 = {mean_score:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to evaluate scaler {scaler_name}: {e}")
        
        logger.info(f"Selected scaler: {best_scaler_name}")
        return best_scaler_name
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series, max_features: int = 50) -> List[str]:
        """Advanced feature selection"""
        logger.info("Performing feature selection...")
        
        # Initial filtering - remove features with low variance
        from sklearn.feature_selection import VarianceThreshold
        variance_selector = VarianceThreshold(threshold=0.001)
        X_var = variance_selector.fit_transform(X)
        var_features = X.columns[variance_selector.get_support()].tolist()
        
        logger.info(f"After variance filtering: {len(var_features)} features")
        
        # Correlation-based selection
        X_var_df = X[var_features]
        correlation_matrix = X_var_df.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find pairs with correlation > 0.95
        high_corr_pairs = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > 0.95)
        ]
        
        # Remove highly correlated features
        final_features = [f for f in var_features if f not in high_corr_pairs]
        logger.info(f"After correlation filtering: {len(final_features)} features")
        
        # SelectKBest with f_regression
        if len(final_features) > max_features:
            X_final = X[final_features]
            selector = SelectKBest(f_regression, k=max_features)
            selector.fit(X_final, y)
            selected_mask = selector.get_support()
            final_features = [final_features[i] for i, selected in enumerate(selected_mask) if selected]
        
        logger.info(f"Final selected features: {len(final_features)}")
        return final_features
    
    async def optimize_hyperparameters(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Any:
        """Optimize hyperparameters for specific models"""
        logger.info(f"Optimizing hyperparameters for {model_name}")
        
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5],
                'max_features': ['sqrt', 'log2']
            },
            'gradient_boost': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [6, 8, 10]
            },
            'svr': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'epsilon': [0.01, 0.1]
            }
        }
        
        if model_name in param_grids:
            model = self.models[model_name]
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name],
                cv=3,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            
            try:
                await asyncio.to_thread(grid_search.fit, X, y)
                logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
                return grid_search.best_estimator_
            except Exception as e:
                logger.warning(f"Hyperparameter optimization failed for {model_name}: {e}")
                return self.models[model_name]
        
        return self.models[model_name]
    
    async def train(self, data: pd.DataFrame, target_type: str = 'return_5d') -> Dict[str, Any]:
        """Comprehensive training with advanced techniques"""
        logger.info("Starting comprehensive training...")
        
        try:
            # Create features
            features = self.create_comprehensive_features(data)
            if features.empty:
                return {'success': False, 'error': 'No features could be created'}
            
            # Create targets
            targets = self.create_target_variables(data)
            if target_type not in targets:
                target_type = 'return_5d'  # Default fallback
                if target_type not in targets:
                    return {'success': False, 'error': 'No valid targets could be created'}
            
            target = targets[target_type]
            
            # Align features and target
            aligned_data = pd.concat([features, target], axis=1).dropna()
            if len(aligned_data) < 100:
                return {'success': False, 'error': 'Insufficient data for training'}
            
            X = aligned_data.iloc[:, :-1]
            y = aligned_data.iloc[:, -1]
            
            # Ensure data is numeric and finite
            X = X.select_dtypes(include=[np.number])  # Keep only numeric columns
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)  # Replace inf with 0
            y = pd.to_numeric(y, errors='coerce').fillna(0)  # Convert to numeric
            
            self.feature_names = X.columns.tolist()
            logger.info(f"Training with {len(X)} samples and {len(X.columns)} features")
            
            # Feature selection
            self.selected_features = self.feature_selection(X, y)
            X_selected = X[self.selected_features]
            
            # Ensure proper data types
            X_selected = X_selected.astype(np.float64)
            y = y.astype(np.float64)
            
            # Final check for any remaining issues
            if not np.isfinite(X_selected.values).all():
                logger.warning("Non-finite values detected in features, cleaning...")
                X_selected = X_selected.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            if not np.isfinite(y.values).all():
                logger.warning("Non-finite values detected in target, cleaning...")
                y = y.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            logger.info(f"Data types: X={X_selected.dtypes.iloc[0]}, y={y.dtype}")
            logger.info(f"Data shapes: X={X_selected.shape}, y={y.shape}")
            
            # Select best scaler
            best_scaler_name = self.select_best_scaler(X_selected, y)
            self.best_scaler = self.scalers[best_scaler_name]
            X_scaled = self.best_scaler.fit_transform(X_selected)
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Train and evaluate models
            results = {}
            model_scores = {}
            
            for model_name, model in self.models.items():
                logger.info(f"Training {model_name}...")
                
                try:
                    # Hyperparameter optimization for key models
                    if model_name in ['random_forest', 'gradient_boost', 'svr']:
                        optimized_model = await self.optimize_hyperparameters(model_name, X_scaled, y)
                    else:
                        optimized_model = model
                    
                    # Cross-validation
                    cv_scores = cross_val_score(
                        optimized_model, X_scaled, y, 
                        cv=tscv, scoring='r2'
                    )
                    
                    # Train final model
                    await asyncio.to_thread(optimized_model.fit, X_scaled, y)
                    
                    # Calculate metrics
                    train_pred = optimized_model.predict(X_scaled)
                    
                    results[model_name] = {
                        'cv_r2_mean': np.mean(cv_scores),
                        'cv_r2_std': np.std(cv_scores),
                        'train_r2': r2_score(y, train_pred),
                        'train_mse': mean_squared_error(y, train_pred),
                        'train_mae': mean_absolute_error(y, train_pred)
                    }
                    
                    model_scores[model_name] = np.mean(cv_scores)
                    self.best_models[model_name] = optimized_model
                    
                    logger.info(f"{model_name} - CV R2: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_name}: {e}")
                    results[model_name] = {'error': str(e)}
            
            # Calculate ensemble weights based on performance
            if model_scores:
                total_score = sum(max(0, score) for score in model_scores.values())
                if total_score > 0:
                    self.ensemble_weights = {
                        name: max(0, score) / total_score 
                        for name, score in model_scores.items()
                    }
                else:
                    # Equal weights if all scores are negative
                    self.ensemble_weights = {
                        name: 1.0 / len(model_scores) 
                        for name in model_scores.keys()
                    }
            
            self.model_performance = results
            self.is_trained = True
            
            # Store training information
            training_info = {
                'timestamp': datetime.now(),
                'target_type': target_type,
                'n_samples': len(X),
                'n_features': len(self.selected_features),
                'best_scaler': best_scaler_name,
                'model_performance': results,
                'ensemble_weights': self.ensemble_weights
            }
            self.training_history.append(training_info)
            
            logger.info("Training completed successfully!")
            
            return {
                'success': True,
                'results': results,
                'n_samples': len(X),
                'n_features': len(self.selected_features),
                'best_scaler': best_scaler_name,
                'ensemble_weights': self.ensemble_weights,
                'target_type': target_type
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make comprehensive predictions"""
        if not self.is_trained:
            return {'error': 'Models not trained yet'}
        
        try:
            # Create features
            features = self.create_comprehensive_features(data)
            if features.empty:
                return {'error': 'No features could be created'}
            
            # Select features and scale
            features_selected = features[self.selected_features].iloc[-1:].fillna(0)
            X_scaled = self.best_scaler.transform(features_selected)
            
            # Make predictions with all models
            predictions = {}
            confidence_scores = {}
            
            for model_name, model in self.best_models.items():
                try:
                    pred = model.predict(X_scaled)[0]
                    predictions[model_name] = float(pred)
                    
                    # Calculate confidence based on cross-validation performance
                    cv_performance = self.model_performance.get(model_name, {})
                    cv_r2 = cv_performance.get('cv_r2_mean', 0)
                    confidence_scores[model_name] = max(0, min(1, cv_r2))
                    
                except Exception as e:
                    logger.error(f"Prediction failed for {model_name}: {e}")
                    predictions[model_name] = 0.0
                    confidence_scores[model_name] = 0.0
            
            # Ensemble prediction using learned weights
            if predictions and self.ensemble_weights:
                ensemble_pred = sum(
                    predictions[name] * self.ensemble_weights.get(name, 0)
                    for name in predictions.keys()
                )
                
                # Calculate ensemble confidence
                ensemble_confidence = sum(
                    confidence_scores[name] * self.ensemble_weights.get(name, 0)
                    for name in confidence_scores.keys()
                )
            else:
                ensemble_pred = np.mean(list(predictions.values())) if predictions else 0.0
                ensemble_confidence = np.mean(list(confidence_scores.values())) if confidence_scores else 0.0
            
            # Generate trading signal
            signal_strength = abs(ensemble_pred) * ensemble_confidence
            signal_type = 'buy' if ensemble_pred > 0.001 else 'sell' if ensemble_pred < -0.001 else 'hold'
            
            return {
                'success': True,
                'predictions': predictions,
                'confidence_scores': confidence_scores,
                'ensemble_prediction': float(ensemble_pred),
                'ensemble_confidence': float(ensemble_confidence),
                'signal_type': signal_type,
                'signal_strength': float(signal_strength),
                'ensemble_weights': self.ensemble_weights,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {'error': str(e)}
    
    def get_comprehensive_feature_importance(self) -> Dict[str, Any]:
        """Get comprehensive feature importance analysis"""
        if not self.is_trained:
            return {}
        
        importance_analysis = {}
        
        for model_name, model in self.best_models.items():
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                feature_importance = dict(zip(self.selected_features, importances))
                
                # Sort by importance
                sorted_importance = sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                importance_analysis[model_name] = {
                    'feature_importances': dict(sorted_importance),
                    'top_10_features': sorted_importance[:10]
                }
                
            elif hasattr(model, 'coef_'):
                # Linear models
                coefficients = np.abs(model.coef_)
                feature_importance = dict(zip(self.selected_features, coefficients))
                
                sorted_importance = sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                importance_analysis[model_name] = {
                    'coefficients': dict(sorted_importance),
                    'top_10_features': sorted_importance[:10]
                }
        
        # Aggregate importance across all models
        if importance_analysis:
            aggregated_importance = {}
            for feature in self.selected_features:
                scores = []
                for model_analysis in importance_analysis.values():
                    if 'feature_importances' in model_analysis:
                        scores.append(model_analysis['feature_importances'].get(feature, 0))
                    elif 'coefficients' in model_analysis:
                        scores.append(model_analysis['coefficients'].get(feature, 0))
                
                aggregated_importance[feature] = np.mean(scores) if scores else 0
            
            # Sort aggregated importance
            sorted_aggregated = sorted(
                aggregated_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            importance_analysis['aggregated'] = {
                'feature_importances': dict(sorted_aggregated),
                'top_20_features': sorted_aggregated[:20]
            }
        
        return importance_analysis
    
    def save_model(self, filepath: str) -> bool:
        """Save comprehensive model data"""
        if not self.is_trained:
            return False
        
        try:
            model_data = {
                'best_models': self.best_models,
                'best_scaler': self.best_scaler,
                'selected_features': self.selected_features,
                'feature_names': self.feature_names,
                'ensemble_weights': self.ensemble_weights,
                'model_performance': self.model_performance,
                'training_history': self.training_history,
                'is_trained': self.is_trained,
                'config': self.config
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Comprehensive model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load comprehensive model data"""
        try:
            model_data = joblib.load(filepath)
            
            self.best_models = model_data.get('best_models', {})
            self.best_scaler = model_data.get('best_scaler')
            self.selected_features = model_data.get('selected_features', [])
            self.feature_names = model_data.get('feature_names', [])
            self.ensemble_weights = model_data.get('ensemble_weights', {})
            self.model_performance = model_data.get('model_performance', {})
            self.training_history = model_data.get('training_history', [])
            self.is_trained = model_data.get('is_trained', False)
            self.config = model_data.get('config', {})
            
            logger.info(f"Comprehensive model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive model diagnostics"""
        if not self.is_trained:
            return {'error': 'Models not trained yet'}
        
        diagnostics = {
            'training_summary': {
                'n_models': len(self.best_models),
                'n_features': len(self.selected_features),
                'training_history': len(self.training_history)
            },
            'model_performance': self.model_performance,
            'ensemble_weights': self.ensemble_weights,
            'feature_count': len(self.selected_features),
            'best_performing_model': max(
                self.model_performance.items(),
                key=lambda x: x[1].get('cv_r2_mean', -np.inf)
            )[0] if self.model_performance else None,
            'last_training_date': self.training_history[-1]['timestamp'] if self.training_history else None
        }
        
        return diagnostics
    
    async def train_model(self, data: pd.DataFrame, symbol: str = None) -> Dict[str, Any]:
        """Alias for train method to match expected interface"""
        return await self.train(data)