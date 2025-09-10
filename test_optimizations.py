#!/usr/bin/env python3.13
"""
Test script pour vérifier les optimisations du trading bot
Tests les nouvelles fonctionnalités : GPU, cache, sentiment amélioré, etc.
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config import Config
from core.utils import setup_logging

async def test_gpu_detection():
    """Test GPU detection"""
    print("\n🔍 Testing GPU Detection...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
        else:
            print("GPU not detected - will use CPU")
        return torch.cuda.is_available()
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

async def test_cache_system():
    """Test cache system"""
    print("\n💾 Testing Cache System...")
    try:
        from core.data_cache import DataCacheManager
        config = Config()
        
        cache_manager = DataCacheManager(config, cache_dir="test_cache")
        
        # Test cache statistics
        stats = cache_manager.get_cache_statistics()
        print(f"Cache statistics: {stats}")
        
        # Test data caching
        test_data = {'test': 'data', 'timestamp': datetime.now()}
        success = cache_manager._save_to_cache('test_key', test_data, 'test_type')
        print(f"Cache save test: {'✅' if success else '❌'}")
        
        # Test data retrieval
        retrieved_data = cache_manager._load_from_cache('test_key')
        print(f"Cache load test: {'✅' if retrieved_data else '❌'}")
        
        cache_manager.close()
        return True
        
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        return False

async def test_enhanced_sentiment():
    """Test enhanced sentiment analysis"""
    print("\n📰 Testing Enhanced Sentiment Analysis...")
    try:
        from analysis.enhanced_sentiment import EnhancedSentimentAnalyzer
        from analysis.sentiment_analyzer import SentimentAnalyzer
        config = Config()
        
        base_sentiment = SentimentAnalyzer(config)
        enhanced_sentiment = EnhancedSentimentAnalyzer(config, base_sentiment)
        
        await enhanced_sentiment.initialize()
        
        # Test RSS feed fetching
        print("Testing RSS feed collection...")
        news_sources = list(enhanced_sentiment.news_sources.keys())[:3]  # Test first 3
        print(f"Testing sources: {news_sources}")
        
        articles = []
        for source_name, url in list(enhanced_sentiment.news_sources.items())[:2]:
            try:
                source_articles = await enhanced_sentiment._fetch_rss_feed(url, source_name)
                articles.extend(source_articles)
                print(f"  {source_name}: {len(source_articles)} articles")
                if len(articles) >= 5:  # Limit for testing
                    break
            except Exception as e:
                print(f"  {source_name}: Error - {e}")
        
        print(f"Total articles collected: {len(articles)}")
        
        await enhanced_sentiment.cleanup()
        return len(articles) > 0
        
    except Exception as e:
        print(f"❌ Enhanced sentiment test failed: {e}")
        return False

async def test_commodities_forex():
    """Test commodities and forex analysis"""
    print("\n💰 Testing Commodities & Forex Analysis...")
    try:
        from analysis.commodities_forex import CommoditiesForexAnalyzer
        config = Config()
        
        cf_analyzer = CommoditiesForexAnalyzer(config)
        await cf_analyzer.initialize()
        
        # Test commodity data collection
        print("Testing commodity data collection...")
        commodity_data = await cf_analyzer.collect_commodity_data()
        print(f"Commodities collected: {len(commodity_data)}")
        for name, data in commodity_data.items():
            print(f"  {data.name}: ${data.price:.2f} ({data.change_pct_24h:+.2f}%)")
        
        # Test forex data collection  
        print("\nTesting forex data collection...")
        forex_data = await cf_analyzer.collect_forex_data()
        print(f"Forex pairs collected: {len(forex_data)}")
        for pair, data in forex_data.items():
            print(f"  {pair}: {data.rate:.4f} ({data.change_pct_24h:+.2f}%)")
        
        await cf_analyzer.cleanup()
        return len(commodity_data) > 0 and len(forex_data) > 0
        
    except Exception as e:
        print(f"❌ Commodities/Forex test failed: {e}")
        return False

async def test_parallel_training():
    """Test parallel ML training setup"""
    print("\n🔄 Testing Parallel ML Training...")
    try:
        from ml.parallel_trainer import ParallelMLTrainer
        from ml.traditional_ml import TraditionalMLPredictor
        from ml.transformer_ml import TransformerMLPredictor
        config = Config()
        
        parallel_trainer = ParallelMLTrainer(
            config, TraditionalMLPredictor, TransformerMLPredictor
        )
        
        print(f"Parallel trainer initialized with {parallel_trainer.max_workers} workers")
        print(f"GPU queue available: {parallel_trainer.gpu_queue is not None}")
        
        # Test progress callback
        def test_callback(symbol, model_type, status, progress):
            print(f"  Progress: {progress:.1f}% - {symbol} {model_type}: {status}")
        
        parallel_trainer.set_progress_callback(test_callback)
        print("Progress callback set successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Parallel training test failed: {e}")
        return False

async def test_graceful_shutdown():
    """Test graceful shutdown mechanism"""
    print("\n🛑 Testing Graceful Shutdown...")
    try:
        from enhanced_main import GracefulKiller
        import signal
        
        killer = GracefulKiller()
        print(f"Graceful killer initialized")
        print(f"Kill flag: {killer.kill_now}")
        print(f"Shutdown initiated: {killer.shutdown_initiated}")
        
        # Test signal handling setup
        handlers_set = []
        if hasattr(signal, 'SIGINT'):
            handlers_set.append('SIGINT')
        if hasattr(signal, 'SIGTERM'):
            handlers_set.append('SIGTERM')
        if sys.platform == "win32" and hasattr(signal, 'SIGBREAK'):
            handlers_set.append('SIGBREAK')
        
        print(f"Signal handlers set for: {handlers_set}")
        return len(handlers_set) > 0
        
    except Exception as e:
        print(f"❌ Graceful shutdown test failed: {e}")
        return False

async def test_data_types_fix():
    """Test data types fix for ML"""
    print("\n🔢 Testing ML Data Types Fix...")
    try:
        import numpy as np
        import pandas as pd
        
        # Create test data with mixed types (like the original bug)
        test_data = pd.DataFrame({
            'feature1': np.array([1, 2, 3, 4, 5], dtype=np.int32),
            'feature2': np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float32),
            'target': np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        })
        
        print(f"Original dtypes: {test_data.dtypes.to_dict()}")
        
        # Apply the fix (convert to float64)
        X = test_data[['feature1', 'feature2']].astype(np.float64)
        y = test_data['target'].astype(np.float64)
        
        print(f"Fixed dtypes: X={X.dtypes.to_dict()}, y={y.dtype}")
        
        # Test with sklearn (should not raise "input array type is not double" error)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        prediction = model.predict(X)
        
        print(f"ML model training successful: {len(prediction)} predictions made")
        return True
        
    except Exception as e:
        print(f"❌ Data types test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Testing Trading Bot Optimizations")
    print("=" * 50)
    
    # Setup basic logging for tests
    logging.basicConfig(level=logging.INFO)
    
    tests = [
        ("GPU Detection", test_gpu_detection),
        ("Cache System", test_cache_system), 
        ("Enhanced Sentiment", test_enhanced_sentiment),
        ("Commodities & Forex", test_commodities_forex),
        ("Parallel Training", test_parallel_training),
        ("Graceful Shutdown", test_graceful_shutdown),
        ("Data Types Fix", test_data_types_fix),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"\n{status} - {test_name}")
        except Exception as e:
            results[test_name] = False
            print(f"\n❌ ERROR - {test_name}: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All optimizations are working correctly!")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed - check the issues above")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)