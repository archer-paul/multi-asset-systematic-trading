#!/usr/bin/env python3.13
"""
Test rapide des corrections ML
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_traditional_ml_fix():
    """Test que traditional ML fonctionne avec les corrections"""
    print("Testing Traditional ML fixes...")
    
    try:
        from ml.traditional_ml import TraditionalMLPredictor
        from core.config import Config
        
        config = Config()
        model = TraditionalMLPredictor(config)
        
        # Create test data with mixed types (reproducing the bug)
        test_data = pd.DataFrame({
            'Open': np.random.randn(200) * 10 + 100,
            'High': np.random.randn(200) * 10 + 105,
            'Low': np.random.randn(200) * 10 + 95,
            'Close': np.random.randn(200) * 10 + 100,
            'Volume': np.random.randint(1000, 10000, 200)
        })
        
        # Add some problematic values
        test_data.iloc[0, 0] = np.inf
        test_data.iloc[1, 1] = -np.inf
        test_data.iloc[2, 2] = np.nan
        
        print(f"Test data shape: {test_data.shape}")
        print(f"Data types: {test_data.dtypes.to_dict()}")
        print("Testing feature creation...")
        
        # Test feature creation
        features = model.create_comprehensive_features(test_data)
        print(f"Features created: {len(features.columns)} features")
        
        # Test target creation
        targets = model.create_target_variables(test_data)
        print(f"Targets created: {len(targets)} targets")
        
        print("✅ Traditional ML preprocessing works!")
        return True
        
    except Exception as e:
        print(f"❌ Traditional ML test failed: {e}")
        return False

def test_wavenet_fix():
    """Test que WaveNet fonctionne avec les corrections"""
    print("\nTesting WaveNet fixes...")
    
    try:
        import torch
        from ml.transformer_ml import WaveNet
        
        # Create test model (corrected parameter name)
        model = WaveNet(input_channels=10, residual_channels=32, skip_channels=32, output_dim=1)
        
        # Test with different input sizes to trigger size mismatch
        test_inputs = [
            torch.randn(2, 60, 10),  # Standard size
            torch.randn(2, 59, 10),  # Size that causes mismatch
            torch.randn(2, 61, 10),  # Another problematic size
        ]
        
        for i, test_input in enumerate(test_inputs):
            print(f"Testing input shape: {test_input.shape}")
            output = model(test_input)
            print(f"Output shape: {output.shape}")
        
        print("✅ WaveNet tensor size fixes work!")
        return True
        
    except Exception as e:
        print(f"❌ WaveNet test failed: {e}")
        return False

def test_redis_connection():
    """Test Redis connection"""
    print("\nTesting Redis connection...")
    
    try:
        import redis
        
        client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        client.ping()
        
        # Test basic operations
        client.set('test_key', 'test_value')
        value = client.get('test_key')
        client.delete('test_key')
        
        print(f"✅ Redis working! Test value: {value}")
        return True
        
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        return False

def main():
    """Run all fix tests"""
    print("🧪 TESTING CRITICAL FIXES")
    print("=" * 40)
    
    tests = [
        ("Traditional ML Data Types", test_traditional_ml_fix),
        ("WaveNet Tensor Sizes", test_wavenet_fix),
        ("Redis Connection", test_redis_connection),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            results[test_name] = False
            print(f"❌ {test_name}: {e}")
    
    # Summary
    print("\n" + "=" * 40)
    print("🎯 FIX TEST SUMMARY")
    print("=" * 40)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    print(f"\nFixed: {passed}/{total} issues ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All critical fixes working! Bot should run much better now.")
    else:
        print(f"\n⚠️ {total-passed} issue(s) still need attention.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)