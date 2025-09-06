"""
Utility functions for the trading bot
Common functionality used across modules
"""

import os
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import json
import hashlib
import asyncio
from functools import wraps
import pandas as pd
import numpy as np

def setup_logging(log_level: str = "INFO", log_file: str = "logs/trading_bot.log"):
    """
    Setup comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
    """
    
    # Create logs directory if it doesn't exist
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler with colored output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Set specific log levels for noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")

def create_directories():
    """Create necessary directories for the trading bot"""
    
    directories = [
        'logs',
        'data',
        'data/cache',
        'data/models',
        'data/reports',
        'data/backups'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logging.info("Created necessary directories")

def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent

def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file with error handling
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary from JSON file
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"JSON file not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON file {file_path}: {e}")
        return {}

def save_json_file(data: Dict[str, Any], file_path: Union[str, Path]):
    """
    Save data to JSON file with error handling
    
    Args:
        data: Data to save
        file_path: Path to save file
    """
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logging.debug(f"Saved JSON file: {file_path}")
        
    except Exception as e:
        logging.error(f"Error saving JSON file {file_path}: {e}")

def generate_id(prefix: str = "", length: int = 8) -> str:
    """
    Generate unique ID with optional prefix
    
    Args:
        prefix: Optional prefix for the ID
        length: Length of random part
        
    Returns:
        Unique ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_part = hashlib.md5(f"{timestamp}{np.random.random()}".encode()).hexdigest()[:length]
    
    if prefix:
        return f"{prefix}_{timestamp}_{random_part}"
    else:
        return f"{timestamp}_{random_part}"

def normalize_symbol(symbol: str) -> str:
    """
    Normalize stock symbol (remove exchange suffixes, etc.)
    
    Args:
        symbol: Raw symbol
        
    Returns:
        Normalized symbol
    """
    # Remove common exchange suffixes
    symbol = symbol.upper()
    
    # Keep exchange suffixes for European stocks
    if any(suffix in symbol for suffix in ['.AS', '.PA', '.SW', '.DE']):
        return symbol
    
    # Remove US exchange suffixes
    if '.' in symbol:
        symbol = symbol.split('.')[0]
    
    return symbol

def calculate_correlation(series1: pd.Series, series2: pd.Series, min_periods: int = 30) -> float:
    """
    Calculate correlation between two series with minimum period requirement
    
    Args:
        series1: First time series
        series2: Second time series
        min_periods: Minimum periods required for calculation
        
    Returns:
        Correlation coefficient or 0.0 if insufficient data
    """
    try:
        # Align series by index
        aligned = pd.concat([series1, series2], axis=1).dropna()
        
        if len(aligned) < min_periods:
            return 0.0
        
        correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        return correlation if not np.isnan(correlation) else 0.0
        
    except Exception as e:
        logging.error(f"Error calculating correlation: {e}")
        return 0.0

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """
    Calculate Sharpe ratio for a return series
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sharpe ratio
    """
    try:
        if len(returns) == 0:
            return 0.0
        
        # Convert to annual terms
        annual_return = returns.mean() * 252  # Assuming daily returns
        annual_volatility = returns.std() * np.sqrt(252)
        
        if annual_volatility == 0:
            return 0.0
        
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        return sharpe_ratio
        
    except Exception as e:
        logging.error(f"Error calculating Sharpe ratio: {e}")
        return 0.0

def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Calculate maximum drawdown from cumulative returns
    
    Args:
        cumulative_returns: Series of cumulative returns
        
    Returns:
        Maximum drawdown (negative value)
    """
    try:
        if len(cumulative_returns) == 0:
            return 0.0
        
        # Calculate running maximum
        running_max = cumulative_returns.expanding().max()
        
        # Calculate drawdowns
        drawdowns = cumulative_returns - running_max
        
        # Return maximum drawdown (most negative value)
        max_drawdown = drawdowns.min()
        return max_drawdown
        
    except Exception as e:
        logging.error(f"Error calculating max drawdown: {e}")
        return 0.0

def retry_async(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for retrying async functions with exponential backoff
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff: Backoff multiplier
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logging.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    logging.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}, retrying in {current_delay}s")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
        return wrapper
    return decorator

def rate_limit(calls_per_second: float = 1.0):
    """
    Decorator for rate limiting function calls
    
    Args:
        calls_per_second: Maximum calls per second
    """
    min_interval = 1.0 / calls_per_second
    last_called = {}
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            func_id = id(func)
            now = datetime.now().timestamp()
            
            if func_id in last_called:
                elapsed = now - last_called[func_id]
                if elapsed < min_interval:
                    sleep_time = min_interval - elapsed
                    await asyncio.sleep(sleep_time)
            
            last_called[func_id] = datetime.now().timestamp()
            return await func(*args, **kwargs)
            
        return wrapper
    return decorator

def validate_symbol_list(symbols: List[str]) -> List[str]:
    """
    Validate and clean a list of stock symbols
    
    Args:
        symbols: List of symbols to validate
        
    Returns:
        List of valid symbols
    """
    valid_symbols = []
    
    for symbol in symbols:
        try:
            # Basic validation
            if not symbol or not isinstance(symbol, str):
                continue
            
            # Clean and normalize
            clean_symbol = normalize_symbol(symbol.strip())
            
            # Check length (typical symbols are 1-5 characters plus exchange)
            if 1 <= len(clean_symbol.split('.')[0]) <= 5:
                valid_symbols.append(clean_symbol)
            else:
                logging.warning(f"Invalid symbol format: {symbol}")
                
        except Exception as e:
            logging.error(f"Error validating symbol {symbol}: {e}")
    
    return list(set(valid_symbols))  # Remove duplicates

def convert_timezone(dt: datetime, from_tz: str = 'UTC', to_tz: str = 'Europe/Paris') -> datetime:
    """
    Convert datetime between timezones
    
    Args:
        dt: Datetime to convert
        from_tz: Source timezone
        to_tz: Target timezone
        
    Returns:
        Converted datetime
    """
    try:
        import pytz
        
        if dt.tzinfo is None:
            # Assume source timezone if naive datetime
            source_tz = pytz.timezone(from_tz)
            dt = source_tz.localize(dt)
        
        target_tz = pytz.timezone(to_tz)
        return dt.astimezone(target_tz)
        
    except ImportError:
        logging.warning("pytz not available, returning original datetime")
        return dt
    except Exception as e:
        logging.error(f"Error converting timezone: {e}")
        return dt

def format_currency(amount: float, currency: str = 'EUR', decimal_places: int = 2) -> str:
    """
    Format currency amount for display
    
    Args:
        amount: Amount to format
        currency: Currency code
        decimal_places: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    try:
        if currency == 'EUR':
            symbol = '€'
        elif currency == 'USD':
            symbol = '$'
        else:
            symbol = currency
        
        formatted = f"{symbol}{amount:,.{decimal_places}f}"
        return formatted
        
    except Exception as e:
        logging.error(f"Error formatting currency: {e}")
        return f"{amount:.2f} {currency}"

def format_percentage(value: float, decimal_places: int = 2, include_sign: bool = True) -> str:
    """
    Format percentage for display
    
    Args:
        value: Percentage value (0.05 for 5%)
        decimal_places: Number of decimal places
        include_sign: Whether to include + sign for positive values
        
    Returns:
        Formatted percentage string
    """
    try:
        percentage = value * 100
        
        if include_sign and percentage > 0:
            return f"+{percentage:.{decimal_places}f}%"
        else:
            return f"{percentage:.{decimal_places}f}%"
            
    except Exception as e:
        logging.error(f"Error formatting percentage: {e}")
        return f"{value:.4f}"

def get_market_hours(region: str = 'US') -> Dict[str, str]:
    """
    Get market hours for different regions
    
    Args:
        region: Market region (US, EU)
        
    Returns:
        Dictionary with market open/close times
    """
    market_hours = {
        'US': {
            'open': '09:30',
            'close': '16:00',
            'timezone': 'America/New_York'
        },
        'EU': {
            'open': '09:00',
            'close': '17:30',
            'timezone': 'Europe/Paris'
        }
    }
    
    return market_hours.get(region, market_hours['US'])

def is_market_open(region: str = 'US') -> bool:
    """
    Check if market is currently open
    
    Args:
        region: Market region
        
    Returns:
        True if market is open
    """
    try:
        import pytz
        from datetime import time
        
        market_info = get_market_hours(region)
        market_tz = pytz.timezone(market_info['timezone'])
        
        # Get current time in market timezone
        now = datetime.now(market_tz)
        current_time = now.time()
        current_weekday = now.weekday()
        
        # Check if it's a weekday (0-4 = Monday-Friday)
        if current_weekday >= 5:  # Weekend
            return False
        
        # Parse market hours
        open_time = time(*map(int, market_info['open'].split(':')))
        close_time = time(*map(int, market_info['close'].split(':')))
        
        # Check if current time is within market hours
        return open_time <= current_time <= close_time
        
    except ImportError:
        logging.warning("pytz not available, assuming market is open")
        return True
    except Exception as e:
        logging.error(f"Error checking market hours: {e}")
        return True

def clean_text(text: str) -> str:
    """
    Clean text for analysis (remove special characters, normalize whitespace)
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    try:
        if not text:
            return ""
        
        import re
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags (but keep the text)
        text = re.sub(r'[@#](\w+)', r'\1', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.strip()
        
    except Exception as e:
        logging.error(f"Error cleaning text: {e}")
        return text

def calculate_portfolio_metrics(positions: Dict[str, Dict], cash: float, 
                              initial_capital: float) -> Dict[str, float]:
    """
    Calculate basic portfolio metrics
    
    Args:
        positions: Dictionary of positions
        cash: Current cash amount
        initial_capital: Starting capital
        
    Returns:
        Dictionary of portfolio metrics
    """
    try:
        # Calculate current portfolio value
        positions_value = sum(
            pos.get('quantity', 0) * pos.get('current_price', pos.get('avg_price_eur', 0))
            for pos in positions.values()
        )
        
        total_value = cash + positions_value
        
        # Calculate metrics
        total_return = (total_value - initial_capital) / initial_capital
        cash_percentage = cash / total_value if total_value > 0 else 1.0
        positions_percentage = positions_value / total_value if total_value > 0 else 0.0
        
        # Calculate unrealized P&L
        unrealized_pnl = sum(
            (pos.get('current_price', pos.get('avg_price_eur', 0)) - pos.get('avg_price_eur', 0)) * pos.get('quantity', 0)
            for pos in positions.values()
        )
        
        return {
            'total_value': total_value,
            'positions_value': positions_value,
            'cash': cash,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'cash_percentage': cash_percentage * 100,
            'positions_percentage': positions_percentage * 100,
            'unrealized_pnl': unrealized_pnl,
            'num_positions': len(positions)
        }
        
    except Exception as e:
        logging.error(f"Error calculating portfolio metrics: {e}")
        return {
            'total_value': initial_capital,
            'total_return_pct': 0.0,
            'error': str(e)
        }

# Context managers for timing operations
class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        logging.debug(f"{self.description} completed in {duration:.2f} seconds")

# Data validation functions
def validate_price_data(df: pd.DataFrame) -> bool:
    """
    Validate price data DataFrame
    
    Args:
        df: DataFrame with price data
        
    Returns:
        True if valid
    """
    try:
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check if required columns exist
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Check for negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        if (df[price_columns] < 0).any().any():
            return False
        
        # Check for logical price relationships
        invalid_rows = (
            (df['High'] < df['Low']) |
            (df['High'] < df['Open']) |
            (df['High'] < df['Close']) |
            (df['Low'] > df['Open']) |
            (df['Low'] > df['Close'])
        )
        
        if invalid_rows.any():
            logging.warning(f"Found {invalid_rows.sum()} rows with invalid price relationships")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating price data: {e}")
        return False

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with default value for zero denominator
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
        
    Returns:
        Division result or default
    """
    try:
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return default
        
        result = numerator / denominator
        
        if np.isnan(result) or np.isinf(result):
            return default
        
        return result
        
    except Exception:
        return default

def memory_usage_mb() -> float:
    """
    Get current memory usage in MB
    
    Returns:
        Memory usage in MB
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0
    except Exception as e:
        logging.error(f"Error getting memory usage: {e}")
        return 0.0

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for monitoring
    
    Returns:
        Dictionary with system information
    """
    try:
        import platform
        import psutil
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage_pct': psutil.disk_usage('/').percent,
            'process_memory_mb': memory_usage_mb()
        }
    except ImportError:
        return {'platform': 'unknown', 'python_version': 'unknown'}
    except Exception as e:
        logging.error(f"Error getting system info: {e}")
        return {'error': str(e)}

# Configuration helpers
def load_config_from_env(prefix: str = "TRADING_BOT_") -> Dict[str, Any]:
    """
    Load configuration from environment variables
    
    Args:
        prefix: Prefix for environment variables
        
    Returns:
        Dictionary of configuration values
    """
    config = {}
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            
            # Try to convert to appropriate type
            if value.lower() in ('true', 'false'):
                config[config_key] = value.lower() == 'true'
            elif value.isdigit():
                config[config_key] = int(value)
            elif '.' in value and value.replace('.', '').isdigit():
                config[config_key] = float(value)
            else:
                config[config_key] = value
    
    return config

# File operations
def backup_file(file_path: Union[str, Path], backup_dir: str = "data/backups") -> Optional[str]:
    """
    Create backup of a file
    
    Args:
        file_path: Path to file to backup
        backup_dir: Directory for backups
        
    Returns:
        Path to backup file or None if failed
    """
    try:
        import shutil
        
        file_path = Path(file_path)
        if not file_path.exists():
            return None
        
        # Create backup directory
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_file_path = backup_path / backup_filename
        
        # Copy file
        shutil.copy2(file_path, backup_file_path)
        
        logging.info(f"Backup created: {backup_file_path}")
        return str(backup_file_path)
        
    except Exception as e:
        logging.error(f"Error creating backup of {file_path}: {e}")
        return None

def cleanup_old_files(directory: Union[str, Path], days_old: int = 30, pattern: str = "*") -> int:
    """
    Clean up old files in directory
    
    Args:
        directory: Directory to clean
        days_old: Delete files older than this many days
        pattern: File pattern to match
        
    Returns:
        Number of files deleted
    """
    try:
        directory = Path(directory)
        if not directory.exists():
            return 0
        
        cutoff_time = datetime.now() - timedelta(days=days_old)
        deleted_count = 0
        
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
        
        if deleted_count > 0:
            logging.info(f"Cleaned up {deleted_count} old files from {directory}")
        
        return deleted_count
        
    except Exception as e:
        logging.error(f"Error cleaning up old files in {directory}: {e}")
        return 0

# Health check functions
async def health_check_api(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Perform health check on API endpoint
    
    Args:
        url: API endpoint URL
        timeout: Request timeout
        
    Returns:
        Health check results
    """
    try:
        import aiohttp
        
        start_time = datetime.now()
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(url) as response:
                duration = (datetime.now() - start_time).total_seconds()
                
                return {
                    'url': url,
                    'status_code': response.status,
                    'response_time_ms': duration * 1000,
                    'healthy': 200 <= response.status < 300,
                    'timestamp': datetime.now()
                }
                
    except ImportError:
        return {'url': url, 'healthy': False, 'error': 'aiohttp not available'}
    except Exception as e:
        return {
            'url': url,
            'healthy': False,
            'error': str(e),
            'timestamp': datetime.now()
        }

def validate_environment() -> Dict[str, bool]:
    """
    Validate environment setup
    
    Returns:
        Dictionary of validation results
    """
    validations = {}
    
    # Check Python version
    import sys
    validations['python_version_ok'] = sys.version_info >= (3, 8)
    
    # Check required directories
    required_dirs = ['logs', 'data']
    for directory in required_dirs:
        validations[f'directory_{directory}_exists'] = Path(directory).exists()
    
    # Check environment variables
    required_env_vars = ['GEMINI_API_KEY']
    for var in required_env_vars:
        validations[f'env_var_{var.lower()}_set'] = bool(os.getenv(var))
    
    # Check disk space
    try:
        import shutil
        _, _, free_bytes = shutil.disk_usage('.')
        free_gb = free_bytes / (1024**3)
        validations['sufficient_disk_space'] = free_gb > 1.0  # At least 1GB free
    except Exception:
        validations['sufficient_disk_space'] = True  # Assume OK if can't check
    
    return validations

# Performance monitoring
class PerformanceMonitor:
    """Simple performance monitoring for functions"""
    
    def __init__(self):
        self.metrics = {}
    
    def record(self, operation: str, duration: float, success: bool = True):
        """Record operation metrics"""
        if operation not in self.metrics:
            self.metrics[operation] = {
                'total_calls': 0,
                'total_duration': 0.0,
                'successful_calls': 0,
                'failed_calls': 0,
                'avg_duration': 0.0,
                'last_call': None
            }
        
        metric = self.metrics[operation]
        metric['total_calls'] += 1
        metric['total_duration'] += duration
        metric['avg_duration'] = metric['total_duration'] / metric['total_calls']
        metric['last_call'] = datetime.now()
        
        if success:
            metric['successful_calls'] += 1
        else:
            metric['failed_calls'] += 1
    
    def get_metrics(self) -> Dict[str, Dict]:
        """Get all recorded metrics"""
        return self.metrics.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics"""
        total_calls = sum(m['total_calls'] for m in self.metrics.values())
        total_duration = sum(m['total_duration'] for m in self.metrics.values())
        
        return {
            'total_operations': len(self.metrics),
            'total_calls': total_calls,
            'total_duration': total_duration,
            'avg_duration_per_call': total_duration / total_calls if total_calls > 0 else 0,
            'operations': list(self.metrics.keys())
        }

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def monitor_performance(operation_name: str):
    """
    Decorator to monitor function performance
    
    Args:
        operation_name: Name of the operation being monitored
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.now()
            success = True
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = (datetime.now() - start_time).total_seconds()
                performance_monitor.record(operation_name, duration, success)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = datetime.now()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = (datetime.now() - start_time).total_seconds()
                performance_monitor.record(operation_name, duration, success)
        
        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Exception handling utilities
class TradingBotException(Exception):
    """Base exception for trading bot"""
    pass

class DataCollectionException(TradingBotException):
    """Exception for data collection issues"""
    pass

class MLModelException(TradingBotException):
    """Exception for ML model issues"""
    pass

class TradingException(TradingBotException):
    """Exception for trading execution issues"""
    pass

class ConfigurationException(TradingBotException):
    """Exception for configuration issues"""
    pass

def handle_exception(exception_type: type = Exception, default_return: Any = None, log_level: str = "ERROR"):
    """
    Decorator to handle exceptions gracefully
    
    Args:
        exception_type: Type of exception to catch
        default_return: Default return value on exception
        log_level: Logging level for exceptions
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except exception_type as e:
                log_func = getattr(logging, log_level.lower())
                log_func(f"Exception in {func.__name__}: {e}")
                return default_return
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_type as e:
                log_func = getattr(logging, log_level.lower())
                log_func(f"Exception in {func.__name__}: {e}")
                return default_return
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator