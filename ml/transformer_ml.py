"""
Advanced Transformer-based Machine Learning Predictor for Trading Bot
Implements state-of-the-art deep learning models including Transformers,
LSTM, GRU, and attention mechanisms for financial time series prediction
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FinancialTimeSeriesDataset(Dataset):
    """Custom dataset for financial time series data"""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray, features: np.ndarray = None):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.features = torch.FloatTensor(features) if features is not None else None
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.features is not None:
            return self.sequences[idx], self.features[idx], self.targets[idx]
        return self.sequences[idx], self.targets[idx]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for financial data"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        return context, attention_weights
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        residual = x
        
        # Linear transformations
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Output projection
        output = self.W_o(context)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output, attention_weights

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward layers"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Multi-head attention
        attn_output, attention_weights = self.attention(x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        return x, attention_weights

class FinancialTransformer(nn.Module):
    """Advanced Transformer model for financial prediction"""
    
    def __init__(
        self,
        seq_len: int,
        feature_dim: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        output_dim: int = 1
    ):
        super(FinancialTransformer, self).__init__()
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, feature_dim = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        x = self.dropout(x)
        
        # Transformer blocks
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x, mask)
            attention_weights.append(attn_weights)
        
        # Global pooling and output
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, d_model)
        
        output = self.output_layers(x)
        
        return output, attention_weights

class AdvancedLSTM(nn.Module):
    """Advanced LSTM with attention mechanism"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True,
        output_dim: int = 1
    ):
        super(AdvancedLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size),
            nn.Tanh(),
            nn.Linear(lstm_output_size, 1, bias=False)
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 4, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def attention_mechanism(self, lstm_output):
        # Calculate attention weights
        attention_weights = self.attention(lstm_output)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention
        context = torch.sum(attention_weights * lstm_output, dim=1)
        
        return context, attention_weights
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        context, attention_weights = self.attention_mechanism(lstm_out)
        
        # Output prediction
        output = self.output_layers(context)
        
        return output, attention_weights

class WaveNet(nn.Module):
    """WaveNet-inspired architecture for financial time series"""
    
    def __init__(
        self,
        input_channels: int,
        residual_channels: int = 64,
        skip_channels: int = 64,
        dilation_cycles: int = 3,
        layers_per_cycle: int = 10,
        output_dim: int = 1
    ):
        super(WaveNet, self).__init__()
        
        self.input_channels = input_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        
        # Input convolution
        self.start_conv = nn.Conv1d(input_channels, residual_channels, kernel_size=1)
        
        # Dilated convolution layers
        self.dilated_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for cycle in range(dilation_cycles):
            for layer in range(layers_per_cycle):
                dilation = 2 ** layer
                
                # Dilated convolution
                self.dilated_convs.append(
                    nn.Conv1d(
                        residual_channels, 2 * residual_channels,
                        kernel_size=2, dilation=dilation, padding=dilation
                    )
                )
                
                # Residual connection
                self.residual_convs.append(
                    nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
                )
                
                # Skip connection
                self.skip_convs.append(
                    nn.Conv1d(residual_channels, skip_channels, kernel_size=1)
                )
        
        # Output layers
        self.end_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.end_conv2 = nn.Conv1d(skip_channels, output_dim, kernel_size=1)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        # Input: (batch_size, seq_len, features)
        # Conv1d expects: (batch_size, features, seq_len)
        x = x.transpose(1, 2)
        
        # Start convolution
        x = self.start_conv(x)
        skip_connections = []
        
        # Dilated convolutions
        for i, (dilated_conv, residual_conv, skip_conv) in enumerate(
            zip(self.dilated_convs, self.residual_convs, self.skip_convs)
        ):
            # Dilated convolution
            conv_out = dilated_conv(x)
            
            # Gated activation
            filter_out, gate_out = conv_out.chunk(2, dim=1)
            conv_out = torch.tanh(filter_out) * torch.sigmoid(gate_out)
            
            # Residual connection
            residual_out = residual_conv(conv_out)
            x = x + residual_out
            
            # Skip connection
            skip_out = skip_conv(conv_out)
            skip_connections.append(skip_out)
        
        # Combine skip connections
        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0)
        
        # Final convolutions
        output = F.relu(skip_sum)
        output = self.end_conv1(output)
        output = F.relu(output)
        output = self.end_conv2(output)
        
        # Global pooling
        output = self.global_pool(output).squeeze(-1)
        
        return output

class TransformerMLPredictor:
    """Advanced transformer-based ML predictor with multiple architectures"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Model configurations
        if isinstance(self.config, dict):
            self.seq_len = self.config.get('seq_len', 60)
            self.batch_size = self.config.get('batch_size', 32)
            self.epochs = self.config.get('epochs', 100)
            self.learning_rate = self.config.get('learning_rate', 0.001)
            self.patience = self.config.get('patience', 15)
        else:
            self.seq_len = getattr(self.config, 'seq_len', 60)
            self.batch_size = getattr(self.config, 'batch_size', 32)
            self.epochs = getattr(self.config, 'epochs', 100)
            self.learning_rate = getattr(self.config, 'learning_rate', 0.001)
            self.patience = getattr(self.config, 'patience', 15)
            
        self.feature_dim = None
        self.models = {}
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        
        self.best_scaler = None
        self.is_trained = False
        self.training_history = []
        self.model_performance = {}
        
    def create_sequences(self, data: np.ndarray, seq_len: int, target_col: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        sequences = []
        targets = []
        
        for i in range(len(data) - seq_len):
            seq = data[i:i + seq_len]
            target = data[i + seq_len, target_col]
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def prepare_financial_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive financial features"""
        features = pd.DataFrame(index=data.index)
        
        if 'Close' not in data.columns:
            return features
        
        close = data['Close']
        
        # Price-based features
        features['close'] = close
        features['log_close'] = np.log(close)
        
        # Returns
        for period in [1, 2, 3, 5, 10]:
            features[f'return_{period}d'] = close.pct_change(period)
            features[f'log_return_{period}d'] = np.log(close / close.shift(period))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = close.rolling(window).mean()
            features[f'ema_{window}'] = close.ewm(span=window).mean()
            features[f'price_sma_{window}_ratio'] = close / features[f'sma_{window}']
            features[f'price_ema_{window}_ratio'] = close / features[f'ema_{window}']
        
        # Volatility features
        for window in [10, 20, 50]:
            returns = close.pct_change()
            features[f'volatility_{window}'] = returns.rolling(window).std()
            features[f'realized_vol_{window}'] = np.sqrt(252) * returns.rolling(window).std()
        
        # Technical indicators
        if len(close) > 14:
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume features if available
        if 'Volume' in data.columns:
            volume = data['Volume']
            features['volume'] = volume
            features['log_volume'] = np.log(volume + 1)
            
            for window in [10, 20]:
                features[f'volume_sma_{window}'] = volume.rolling(window).mean()
                features[f'volume_ratio_{window}'] = volume / features[f'volume_sma_{window}']
        
        # Clean features
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def build_models(self, feature_dim: int) -> Dict[str, nn.Module]:
        """Build different model architectures"""
        models = {}
        
        # Transformer model
        models['transformer'] = FinancialTransformer(
            seq_len=self.seq_len,
            feature_dim=feature_dim,
            d_model=min(512, feature_dim * 8),
            num_heads=8,
            num_layers=4,
            dropout=0.1
        ).to(self.device)
        
        # Advanced LSTM
        models['lstm'] = AdvancedLSTM(
            input_size=feature_dim,
            hidden_size=256,
            num_layers=3,
            dropout=0.2,
            bidirectional=True
        ).to(self.device)
        
        # WaveNet
        models['wavenet'] = WaveNet(
            input_channels=feature_dim,
            residual_channels=64,
            skip_channels=64,
            dilation_cycles=2,
            layers_per_cycle=8
        ).to(self.device)
        
        return models
    
    def _train_single_model(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        model_name: str
    ) -> Dict[str, Any]:
        """Train a single model with advanced techniques"""
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=self.patience // 3
        )
        
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                if len(batch) == 3:  # With additional features
                    sequences, features, targets = batch
                    sequences, features, targets = sequences.to(self.device), features.to(self.device), targets.to(self.device)
                else:
                    sequences, targets = batch
                    sequences, targets = sequences.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                
                if model_name == 'transformer':
                    outputs, _ = model(sequences)
                elif model_name == 'lstm':
                    outputs, _ = model(sequences)
                else:  # wavenet
                    outputs = model(sequences)
                
                outputs = outputs.squeeze()
                loss = criterion(outputs, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        sequences, features, targets = batch
                        sequences, features, targets = sequences.to(self.device), features.to(self.device), targets.to(self.device)
                    else:
                        sequences, targets = batch
                        sequences, targets = sequences.to(self.device), targets.to(self.device)
                    
                    if model_name == 'transformer':
                        outputs, _ = model(sequences)
                    elif model_name == 'lstm':
                        outputs, _ = model(sequences)
                    else:
                        outputs = model(sequences)
                    
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model state
                torch.save(model.state_dict(), f'best_{model_name}_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"{model_name} Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            if patience_counter >= self.patience:
                logger.info(f"Early stopping for {model_name} at epoch {epoch}")
                break
        
        # Load best model state
        model.load_state_dict(torch.load(f'best_{model_name}_model.pth'))
        
        # Calculate final metrics
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        
        r2 = r2_score(val_targets, val_predictions)
        mse = mean_squared_error(val_targets, val_predictions)
        mae = mean_absolute_error(val_targets, val_predictions)
        
        training_results = {
            'best_val_loss': best_val_loss,
            'final_r2': r2,
            'final_mse': mse,
            'final_mae': mae,
            'epochs_trained': epoch + 1,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
        return training_results
    
    async def train(self, data: pd.DataFrame, target_column: str = 'return_5d') -> Dict[str, Any]:
        """Train all transformer models"""
        logger.info("Starting transformer models training...")
        
        try:
            # Prepare features
            features = self.prepare_financial_features(data)
            if features.empty:
                return {'success': False, 'error': 'No features could be created'}
            
            # Create target
            if target_column not in features.columns:
                if 'Close' in data.columns:
                    features['return_5d'] = data['Close'].shift(-5) / data['Close'] - 1
                    target_column = 'return_5d'
                else:
                    return {'success': False, 'error': 'No valid target could be created'}
            
            # Remove NaN values
            clean_data = features.dropna()
            if len(clean_data) < self.seq_len * 2:
                return {'success': False, 'error': 'Insufficient data after cleaning'}
            
            # Separate features and target
            target_col_idx = clean_data.columns.get_loc(target_column)
            feature_data = clean_data.values
            
            # Scale features
            best_scaler_score = -np.inf
            best_scaler_name = 'standard'
            
            for scaler_name, scaler in self.scalers.items():
                scaled_data = scaler.fit_transform(feature_data)
                
                # Simple validation for scaler selection
                X_seq, y_seq = self.create_sequences(scaled_data, self.seq_len, target_col_idx)
                if len(X_seq) > 10:
                    # Use a small portion for validation
                    val_size = min(len(X_seq) // 5, 50)
                    val_r2 = np.corrcoef(X_seq[-val_size:, -1, target_col_idx], y_seq[-val_size:])[0, 1] ** 2
                    
                    if not np.isnan(val_r2) and val_r2 > best_scaler_score:
                        best_scaler_score = val_r2
                        best_scaler_name = scaler_name
            
            # Use best scaler
            self.best_scaler = self.scalers[best_scaler_name]
            scaled_data = self.best_scaler.fit_transform(feature_data)
            
            # Create sequences
            X_sequences, y_targets = self.create_sequences(scaled_data, self.seq_len, target_col_idx)
            
            # Train-validation split (temporal)
            split_idx = int(len(X_sequences) * 0.8)
            
            X_train, X_val = X_sequences[:split_idx], X_sequences[split_idx:]
            y_train, y_val = y_targets[:split_idx], y_targets[split_idx:]
            
            self.feature_dim = X_train.shape[-1]
            
            # Create data loaders
            train_dataset = FinancialTimeSeriesDataset(X_train, y_train)
            val_dataset = FinancialTimeSeriesDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Build and train models
            self.models = self.build_models(self.feature_dim)
            
            training_results = {}
            for model_name, model in self.models.items():
                logger.info(f"Training {model_name}...")
                
                try:
                    results = self._train_single_model(model, train_loader, val_loader, model_name)
                    training_results[model_name] = results
                    logger.info(f"{model_name} training completed - R2: {results['final_r2']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_name}: {e}")
                    training_results[model_name] = {'error': str(e)}
            
            self.model_performance = training_results
            self.is_trained = True
            
            # Store training information
            training_info = {
                'timestamp': datetime.now(),
                'target_column': target_column,
                'n_sequences': len(X_sequences),
                'sequence_length': self.seq_len,
                'feature_dimension': self.feature_dim,
                'best_scaler': best_scaler_name,
                'model_performance': training_results
            }
            self.training_history.append(training_info)
            
            logger.info("Transformer models training completed!")
            
            return {
                'success': True,
                'results': training_results,
                'n_sequences': len(X_sequences),
                'feature_dimension': self.feature_dim,
                'best_scaler': best_scaler_name,
                'target_column': target_column
            }
            
        except Exception as e:
            logger.error(f"Transformer training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions using trained transformer models"""
        if not self.is_trained:
            return {'error': 'Models not trained yet'}
        
        try:
            # Prepare features
            features = self.prepare_financial_features(data)
            if features.empty:
                return {'error': 'No features could be created'}
            
            # Scale features
            scaled_data = self.best_scaler.transform(features.fillna(0).values)
            
            # Create sequence (use last seq_len points)
            if len(scaled_data) < self.seq_len:
                return {'error': f'Need at least {self.seq_len} data points'}
            
            sequence = scaled_data[-self.seq_len:].reshape(1, self.seq_len, self.feature_dim)
            sequence_tensor = torch.FloatTensor(sequence).to(self.device)
            
            # Make predictions with all models
            predictions = {}
            attention_maps = {}
            
            for model_name, model in self.models.items():
                try:
                    model.eval()
                    with torch.no_grad():
                        if model_name == 'transformer':
                            output, attention_weights = model(sequence_tensor)
                            attention_maps[model_name] = [aw.cpu().numpy() for aw in attention_weights]
                        elif model_name == 'lstm':
                            output, attention_weights = model(sequence_tensor)
                            attention_maps[model_name] = attention_weights.cpu().numpy()
                        else:  # wavenet
                            output = model(sequence_tensor)
                        
                        predictions[model_name] = float(output.cpu().numpy().squeeze())
                    
                except Exception as e:
                    logger.error(f"Prediction failed for {model_name}: {e}")
                    predictions[model_name] = 0.0
            
            # Ensemble prediction (weighted by validation performance)
            weights = {}
            total_weight = 0
            
            for model_name in predictions.keys():
                perf = self.model_performance.get(model_name, {})
                r2_score = perf.get('final_r2', 0)
                weight = max(0, r2_score)  # Use R2 as weight
                weights[model_name] = weight
                total_weight += weight
            
            if total_weight > 0:
                ensemble_prediction = sum(
                    predictions[name] * (weights[name] / total_weight)
                    for name in predictions.keys()
                )
            else:
                ensemble_prediction = np.mean(list(predictions.values()))
            
            # Generate trading signal
            signal_strength = abs(ensemble_prediction)
            if ensemble_prediction > 0.01:
                signal_type = 'buy'
            elif ensemble_prediction < -0.01:
                signal_type = 'sell'
            else:
                signal_type = 'hold'
            
            return {
                'success': True,
                'predictions': predictions,
                'ensemble_prediction': float(ensemble_prediction),
                'signal_type': signal_type,
                'signal_strength': float(signal_strength),
                'model_weights': weights,
                'attention_maps': attention_maps,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Transformer prediction failed: {e}")
            return {'error': str(e)}
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive model diagnostics"""
        if not self.is_trained:
            return {'error': 'Models not trained yet'}
        
        diagnostics = {
            'training_summary': {
                'n_models': len(self.models),
                'feature_dimension': self.feature_dim,
                'sequence_length': self.seq_len,
                'device': str(self.device),
                'training_history': len(self.training_history)
            },
            'model_performance': self.model_performance,
            'best_performing_model': max(
                self.model_performance.items(),
                key=lambda x: x[1].get('final_r2', -np.inf)
            )[0] if self.model_performance else None,
            'model_architectures': {
                name: {
                    'total_params': sum(p.numel() for p in model.parameters()),
                    'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
                }
                for name, model in self.models.items()
            }
        }
        
        return diagnostics
    
    def save_models(self, filepath_base: str) -> bool:
        """Save all trained models"""
        if not self.is_trained:
            return False
        
        try:
            # Save model states
            for model_name, model in self.models.items():
                model_path = f"{filepath_base}_{model_name}.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_config': {
                        'feature_dim': self.feature_dim,
                        'seq_len': self.seq_len
                    }
                }, model_path)
            
            # Save other data
            import joblib
            metadata = {
                'best_scaler': self.best_scaler,
                'feature_dim': self.feature_dim,
                'seq_len': self.seq_len,
                'model_performance': self.model_performance,
                'training_history': self.training_history,
                'config': self.config,
                'is_trained': self.is_trained
            }
            
            joblib.dump(metadata, f"{filepath_base}_metadata.pkl")
            logger.info(f"Transformer models saved to {filepath_base}_*.pth")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save transformer models: {e}")
            return False
    
    def load_models(self, filepath_base: str) -> bool:
        """Load all trained models"""
        try:
            # Load metadata
            import joblib
            metadata = joblib.load(f"{filepath_base}_metadata.pkl")
            
            self.best_scaler = metadata['best_scaler']
            self.feature_dim = metadata['feature_dim']
            self.seq_len = metadata['seq_len']
            self.model_performance = metadata['model_performance']
            self.training_history = metadata['training_history']
            self.config = metadata['config']
            self.is_trained = metadata['is_trained']
            
            # Rebuild models
            self.models = self.build_models(self.feature_dim)
            
            # Load model states
            for model_name, model in self.models.items():
                model_path = f"{filepath_base}_{model_name}.pth"
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"Transformer models loaded from {filepath_base}_*.pth")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load transformer models: {e}")
            return False
    
    async def train_model(self, symbol: str = None, market_data: pd.DataFrame = None, 
                         news_data: List = None, social_data: Dict = None, 
                         region: str = None) -> Dict[str, Any]:
        """Interface method to match bot_orchestrator expectations"""
        if market_data is None or market_data.empty:
            return {'success': False, 'error': 'No market data provided'}
        
        # Use market_data as the primary data source
        return await self.train(market_data)