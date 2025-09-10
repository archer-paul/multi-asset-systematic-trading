"""
Parallel ML Training System
Trains multiple models on multiple symbols simultaneously using threading/multiprocessing
"""

import logging
import asyncio
import concurrent.futures
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading
import queue
import json
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

class ParallelMLTrainer:
    """Manages parallel training of ML models across multiple symbols"""
    
    def __init__(self, config, traditional_ml_class, transformer_ml_class):
        self.config = config
        self.traditional_ml_class = traditional_ml_class
        self.transformer_ml_class = transformer_ml_class
        
        # Threading configuration
        self.max_workers = min(4, len(config.ALL_SYMBOLS) // 10)  # Conservative threading
        self.gpu_queue = queue.Queue(maxsize=1) if hasattr(config, 'USE_GPU') else None
        self.training_results = {}
        self.training_lock = threading.Lock()
        
        # Progress tracking
        self.total_tasks = 0
        self.completed_tasks = 0
        self.progress_callback = None
        
        logger.info(f"Parallel trainer initialized with {self.max_workers} workers")
    
    def set_progress_callback(self, callback):
        """Set progress callback function"""
        self.progress_callback = callback
    
    def _update_progress(self, symbol: str, model_type: str, status: str):
        """Update training progress"""
        with self.training_lock:
            self.completed_tasks += 1
            progress = (self.completed_tasks / self.total_tasks) * 100 if self.total_tasks > 0 else 0
            
            if self.progress_callback:
                self.progress_callback(symbol, model_type, status, progress)
            
            logger.info(f"Progress: {progress:.1f}% - {symbol} {model_type}: {status}")
    
    def _train_traditional_model(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Train traditional ML model for a single symbol"""
        try:
            # Create model instance (thread-safe)
            model = self.traditional_ml_class(self.config)
            
            self._update_progress(symbol, 'traditional', 'training')
            
            # Train model
            result = asyncio.run(model.train(market_data))
            
            if result.get('success'):
                self._update_progress(symbol, 'traditional', 'completed')
                return {
                    'symbol': symbol,
                    'model_type': 'traditional',
                    'success': True,
                    'result': result,
                    'model': model
                }
            else:
                self._update_progress(symbol, 'traditional', 'failed')
                return {
                    'symbol': symbol,
                    'model_type': 'traditional',
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                }
                
        except Exception as e:
            self._update_progress(symbol, 'traditional', 'error')
            logger.error(f"Error training traditional model for {symbol}: {e}")
            return {
                'symbol': symbol,
                'model_type': 'traditional',
                'success': False,
                'error': str(e)
            }
    
    def _train_transformer_model(self, symbol: str, market_data: pd.DataFrame, 
                                news_data: List = None, social_data: Dict = None, 
                                region: str = None) -> Dict[str, Any]:
        """Train transformer ML model for a single symbol"""
        try:
            # GPU resource management
            gpu_acquired = False
            if self.gpu_queue and not self.gpu_queue.empty():
                try:
                    self.gpu_queue.get_nowait()
                    gpu_acquired = True
                except queue.Empty:
                    pass
            
            # Create model instance
            model = self.transformer_ml_class(self.config)
            
            self._update_progress(symbol, 'transformer', 'training')
            
            # Train model
            result = asyncio.run(model.train_model(
                symbol=symbol,
                market_data=market_data,
                news_data=news_data or [],
                social_data=social_data or {},
                region=region
            ))
            
            # Release GPU resource
            if gpu_acquired and self.gpu_queue:
                self.gpu_queue.put(True)
            
            if result.get('success'):
                self._update_progress(symbol, 'transformer', 'completed')
                return {
                    'symbol': symbol,
                    'model_type': 'transformer',
                    'success': True,
                    'result': result,
                    'model': model
                }
            else:
                self._update_progress(symbol, 'transformer', 'failed')
                return {
                    'symbol': symbol,
                    'model_type': 'transformer',
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                }
                
        except Exception as e:
            # Release GPU resource on error
            if gpu_acquired and self.gpu_queue:
                self.gpu_queue.put(True)
                
            self._update_progress(symbol, 'transformer', 'error')
            logger.error(f"Error training transformer model for {symbol}: {e}")
            return {
                'symbol': symbol,
                'model_type': 'transformer',
                'success': False,
                'error': str(e)
            }
    
    async def train_models_parallel(self, market_data: Dict[str, pd.DataFrame], 
                                  news_data: List = None, social_data: List = None,
                                  train_traditional: bool = True, 
                                  train_transformer: bool = True) -> Dict[str, Any]:
        """Train models in parallel for all symbols"""
        
        logger.info("Starting parallel ML training...")
        
        # Initialize GPU queue if available
        if self.gpu_queue:
            self.gpu_queue.put(True)  # Single GPU token
        
        # Prepare training tasks
        training_tasks = []
        symbols_to_train = [symbol for symbol in self.config.ALL_SYMBOLS if symbol in market_data]
        
        # Calculate total tasks
        task_count = 0
        if train_traditional:
            task_count += len(symbols_to_train)
        if train_transformer:
            task_count += len(symbols_to_train)
        
        self.total_tasks = task_count
        self.completed_tasks = 0
        
        logger.info(f"Training {len(symbols_to_train)} symbols with {task_count} total tasks")
        
        # Create thread pool executor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            
            # Submit traditional ML training tasks
            if train_traditional:
                for symbol in symbols_to_train:
                    if symbol in market_data:
                        future = executor.submit(
                            self._train_traditional_model, 
                            symbol, 
                            market_data[symbol]
                        )
                        training_tasks.append(future)
            
            # Submit transformer ML training tasks (with GPU serialization)
            if train_transformer:
                for symbol in symbols_to_train:
                    if symbol in market_data:
                        # Get symbol-specific data
                        symbol_news = [
                            item for item in (news_data or [])
                            if symbol in item.get('companies_mentioned', [])
                        ]
                        symbol_social = [
                            item for item in (social_data or [])
                            if symbol in item.get('symbols', [])
                        ]
                        region = self.config.get_symbol_region(symbol) if hasattr(self.config, 'get_symbol_region') else 'US'
                        
                        future = executor.submit(
                            self._train_transformer_model,
                            symbol,
                            market_data[symbol],
                            symbol_news,
                            symbol_social,
                            region
                        )
                        training_tasks.append(future)
            
            # Wait for all tasks to complete
            logger.info("Waiting for training tasks to complete...")
            results = []
            
            for future in concurrent.futures.as_completed(training_tasks):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Store successful results
                    if result['success']:
                        key = f"{result['symbol']}_{result['model_type']}"
                        self.training_results[key] = result
                    
                except Exception as e:
                    logger.error(f"Training task failed with exception: {e}")
                    results.append({
                        'success': False,
                        'error': str(e),
                        'symbol': 'unknown',
                        'model_type': 'unknown'
                    })
        
        # Compile final results
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        summary = {
            'total_tasks': len(results),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'success_rate': len(successful_results) / len(results) * 100 if results else 0,
            'training_results': self.training_results,
            'detailed_results': results,
            'timestamp': datetime.now()
        }
        
        logger.info(f"Parallel training completed: {len(successful_results)}/{len(results)} successful ({summary['success_rate']:.1f}%)")
        
        return summary
    
    def get_trained_models(self) -> Dict[str, Any]:
        """Get all successfully trained models"""
        models = {}
        for key, result in self.training_results.items():
            if result['success'] and 'model' in result:
                models[key] = result['model']
        return models
    
    def save_training_summary(self, filepath: str, summary: Dict[str, Any]):
        """Save training summary to file"""
        try:
            # Remove non-serializable model objects for JSON
            summary_to_save = summary.copy()
            if 'training_results' in summary_to_save:
                cleaned_results = {}
                for key, result in summary_to_save['training_results'].items():
                    cleaned_result = result.copy()
                    if 'model' in cleaned_result:
                        del cleaned_result['model']  # Remove model object
                    cleaned_results[key] = cleaned_result
                summary_to_save['training_results'] = cleaned_results
            
            # Convert datetime to string
            if 'timestamp' in summary_to_save:
                summary_to_save['timestamp'] = summary_to_save['timestamp'].isoformat()
            
            with open(filepath, 'w') as f:
                json.dump(summary_to_save, f, indent=2, default=str)
            
            logger.info(f"Training summary saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save training summary: {e}")

class BatchMLTrainer:
    """Batch training system that trains on all symbols simultaneously"""
    
    def __init__(self, config, traditional_ml_class, transformer_ml_class):
        self.config = config
        self.traditional_ml_class = traditional_ml_class
        self.transformer_ml_class = transformer_ml_class
        
        logger.info("Batch ML trainer initialized for cross-symbol training")
    
    def _prepare_batch_data(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare batched data from all symbols for cross-symbol learning"""
        
        batch_data_list = []
        
        for symbol, data in market_data.items():
            if data.empty or len(data) < 60:
                continue
            
            # Add symbol identifier
            data_copy = data.copy()
            data_copy['symbol'] = symbol
            data_copy['symbol_encoded'] = hash(symbol) % 1000  # Simple symbol encoding
            
            # Add region encoding if available
            if hasattr(self.config, 'get_symbol_region'):
                region = self.config.get_symbol_region(symbol)
                data_copy['region_encoded'] = hash(region) % 10
            else:
                data_copy['region_encoded'] = 0
            
            batch_data_list.append(data_copy)
        
        if not batch_data_list:
            return pd.DataFrame()
        
        # Concatenate all data
        batch_data = pd.concat(batch_data_list, ignore_index=True)
        batch_data.sort_index(inplace=True)
        
        logger.info(f"Prepared batch data: {len(batch_data)} rows from {len(batch_data_list)} symbols")
        return batch_data
    
    async def train_batch_models(self, market_data: Dict[str, pd.DataFrame], 
                               news_data: List = None, social_data: List = None) -> Dict[str, Any]:
        """Train models on batched data from all symbols"""
        
        logger.info("Starting batch training (cross-symbol learning)...")
        
        # Prepare batched data
        batch_data = self._prepare_batch_data(market_data)
        if batch_data.empty:
            return {'success': False, 'error': 'No data available for batch training'}
        
        results = {}
        
        # Train traditional ML on batch data
        try:
            logger.info("Training traditional ML on batch data...")
            traditional_model = self.traditional_ml_class(self.config)
            traditional_result = await traditional_model.train(batch_data, target_type='return_5d')
            
            results['batch_traditional'] = {
                'success': traditional_result.get('success', False),
                'result': traditional_result,
                'model': traditional_model if traditional_result.get('success') else None
            }
            
            logger.info(f"Batch traditional ML: {'Success' if traditional_result.get('success') else 'Failed'}")
            
        except Exception as e:
            logger.error(f"Batch traditional ML training failed: {e}")
            results['batch_traditional'] = {'success': False, 'error': str(e)}
        
        # Train transformer ML on batch data
        try:
            logger.info("Training transformer ML on batch data...")
            transformer_model = self.transformer_ml_class(self.config)
            transformer_result = await transformer_model.train(batch_data, target_column='return_5d')
            
            results['batch_transformer'] = {
                'success': transformer_result.get('success', False),
                'result': transformer_result,
                'model': transformer_model if transformer_result.get('success') else None
            }
            
            logger.info(f"Batch transformer ML: {'Success' if transformer_result.get('success') else 'Failed'}")
            
        except Exception as e:
            logger.error(f"Batch transformer ML training failed: {e}")
            results['batch_transformer'] = {'success': False, 'error': str(e)}
        
        # Summary
        successful_models = sum(1 for r in results.values() if r['success'])
        total_models = len(results)
        
        summary = {
            'training_approach': 'batch_cross_symbol',
            'total_models': total_models,
            'successful_models': successful_models,
            'success_rate': (successful_models / total_models) * 100 if total_models > 0 else 0,
            'batch_data_size': len(batch_data),
            'symbols_count': len(market_data),
            'results': results,
            'timestamp': datetime.now()
        }
        
        logger.info(f"Batch training completed: {successful_models}/{total_models} models successful")
        
        return summary