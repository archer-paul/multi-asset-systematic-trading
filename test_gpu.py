#!/usr/bin/env python3.13
"""
Test GPU availability for ML training
"""

def test_gpu_availability():
    """Test si le GPU est disponible pour PyTorch"""
    
    print("=" * 50)
    print("GPU AVAILABILITY TEST")
    print("=" * 50)
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Test simple
            x = torch.randn(1000, 1000)
            x_gpu = x.cuda()
            result = torch.mm(x_gpu, x_gpu.t())
            print("GPU computation test: SUCCESS")
            
            # Recommended device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Recommended device: {device}")
            
        else:
            print("GPU NOT AVAILABLE - Reasons possibles:")
            print("1. PyTorch installé sans support CUDA")
            print("2. Drivers NVIDIA obsolètes") 
            print("3. GPU non compatible CUDA")
            
            print("\nSOLUTIONS:")
            print("1. Réinstaller PyTorch avec CUDA:")
            print("   pip uninstall torch torchvision torchaudio")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print("2. Vérifier les drivers NVIDIA")
            print("3. Vérifier compatibilité GPU sur: https://developer.nvidia.com/cuda-gpus")
            
    except ImportError:
        print("[ERROR] PyTorch not installed")
        return False
    except Exception as e:
        print(f"[ERROR] GPU test failed: {e}")
        return False
    
    return torch.cuda.is_available()

def check_nvidia_driver():
    """Vérifie le driver NVIDIA"""
    
    print("\n" + "=" * 50)
    print("NVIDIA DRIVER CHECK")
    print("=" * 50)
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("nvidia-smi output:")
            print(result.stdout)
            return True
        else:
            print("nvidia-smi failed:")
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print("[ERROR] nvidia-smi not found - Driver NVIDIA non installé")
        return False
    except Exception as e:
        print(f"[ERROR] Driver check failed: {e}")
        return False

if __name__ == "__main__":
    gpu_available = test_gpu_availability()
    driver_ok = check_nvidia_driver()
    
    print("\n" + "=" * 50)
    print("RÉSUMÉ")
    print("=" * 50)
    print(f"GPU disponible pour PyTorch: {'OUI' if gpu_available else 'NON'}")
    print(f"Driver NVIDIA fonctionnel: {'OUI' if driver_ok else 'NON'}")
    
    if gpu_available:
        print("\n[SUCCESS] Votre bot utilisera le GPU pour l'entraînement ML!")
        print("Performance attendue: 5-10x plus rapide que CPU")
    else:
        print("\n[INFO] Le bot utilisera le CPU (plus lent mais fonctionnel)")
        print("Pour activer le GPU, suivez les solutions proposées ci-dessus")