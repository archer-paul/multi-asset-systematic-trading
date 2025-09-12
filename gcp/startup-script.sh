#!/bin/bash

# Startup script for ML Training VM
# This script runs when the VM starts and sets up the environment

set -e

LOG_FILE="/var/log/startup-script.log"
exec > >(tee -a $LOG_FILE)
exec 2>&1

echo "🚀 Starting ML Training VM setup... $(date)"

# Update system
echo "📦 Updating system packages..."
apt-get update
apt-get install -y git wget curl htop tmux

# Install additional Python packages for trading bot
echo "🐍 Installing additional Python packages..."
pip install --upgrade pip
pip install \
    yfinance \
    TA-Lib \
    psycopg2-binary \
    redis \
    sqlalchemy \
    alembic \
    asyncio-throttle \
    google-generativeai \
    textblob \
    nltk \
    tweepy \
    praw \
    asyncpraw \
    plotly \
    dash \
    streamlit \
    python-dotenv \
    pytz \
    psutil \
    click \
    pytest \
    pytest-asyncio

# Install TA-Lib from source (if not already available)
echo "📈 Installing TA-Lib..."
cd /tmp
if [ ! -f "ta-lib-0.4.0-src.tar.gz" ]; then
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
fi
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr/local
make && make install
ldconfig
cd ..
rm -rf ta-lib*

# Download NLTK data
echo "📚 Downloading NLTK data..."
python3 -c "
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')
"

# Create working directory
echo "📁 Creating working directory..."
mkdir -p /home/jupyter/trading-bot
chown -R jupyter:jupyter /home/jupyter/trading-bot

# Set up environment variables for Jupyter
echo "🔧 Setting up environment..."
cat > /home/jupyter/.env << EOF
# GCP ML Training Environment
CUDA_VISIBLE_DEVICES=0
PYTHONPATH=/home/jupyter/trading-bot
TRADING_MODE=comprehensive_mode
ENABLE_TRADITIONAL_ML=true
ENABLE_TRANSFORMER_ML=true
ML_TRAINING_LOOKBACK_DAYS=3650
ANALYSIS_LOOKBACK_DAYS=90
NEWS_LOOKBACK_DAYS=60
LOG_LEVEL=INFO
DEBUG_MODE=false
EOF

# Create a sample ML training script
cat > /home/jupyter/trading-bot/train_models.py << 'EOF'
#!/usr/bin/env python3
"""
ML Training Script for GCP
Optimized for GPU training on Compute Engine
"""

import os
import sys
import logging
import torch
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"✅ GPU available: {gpu_count} x {gpu_name}")
        logging.info(f"CUDA version: {torch.version.cuda}")
        return True
    else:
        logging.warning("❌ No GPU available, using CPU")
        return False

def main():
    """Main training function"""
    logging.info("🔥 Starting ML model training...")
    logging.info(f"Timestamp: {datetime.now()}")
    
    # Check GPU
    gpu_available = check_gpu()
    
    # TODO: Add your actual ML training code here
    # This is where you would:
    # 1. Load your trading data
    # 2. Preprocess the data
    # 3. Train your models (Traditional ML + Transformer)
    # 4. Save trained models to Cloud Storage
    
    logging.info("🎯 Training placeholder completed!")
    logging.info("Replace this with your actual trading bot ML training code")

if __name__ == "__main__":
    main()
EOF

# Make script executable
chmod +x /home/jupyter/trading-bot/train_models.py
chown jupyter:jupyter /home/jupyter/trading-bot/train_models.py

# Create Jupyter config to allow external access
echo "📓 Configuring Jupyter..."
mkdir -p /home/jupyter/.jupyter
cat > /home/jupyter/.jupyter/jupyter_notebook_config.py << EOF
c.NotebookApp.allow_origin = '*'
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.token = ''
c.NotebookApp.password = ''
c.NotebookApp.open_browser = False
c.NotebookApp.allow_root = True
EOF

chown -R jupyter:jupyter /home/jupyter/.jupyter

# Set up tmux session for long-running training
echo "🖥️ Setting up tmux session..."
cat > /home/jupyter/start_training.sh << 'EOF'
#!/bin/bash
# Start ML training in a tmux session

echo "Starting ML training in tmux session..."
tmux new-session -d -s ml_training
tmux send-keys -t ml_training "cd /home/jupyter/trading-bot" C-m
tmux send-keys -t ml_training "python train_models.py" C-m

echo "Training started in tmux session 'ml_training'"
echo "To attach: tmux attach -t ml_training"
echo "To detach: Ctrl+B then D"
EOF

chmod +x /home/jupyter/start_training.sh
chown jupyter:jupyter /home/jupyter/start_training.sh

# Install Google Cloud SDK (if not already present)
if ! command -v gsutil &> /dev/null; then
    echo "☁️ Installing Google Cloud SDK..."
    curl https://sdk.cloud.google.com | bash
    exec -l $SHELL
fi

# Create helper scripts
cat > /home/jupyter/save_models.sh << 'EOF'
#!/bin/bash
# Save trained models to Cloud Storage

BUCKET_NAME="${GCP_PROJECT_ID:-trading-bot-project}-trading-data"
MODEL_DIR="/home/jupyter/trading-bot/trained_models"

if [ -d "$MODEL_DIR" ]; then
    echo "📤 Uploading models to Cloud Storage..."
    gsutil -m cp -r $MODEL_DIR gs://$BUCKET_NAME/models/$(date +%Y%m%d_%H%M%S)/
    echo "✅ Models uploaded successfully!"
else
    echo "❌ No models directory found at $MODEL_DIR"
fi
EOF

chmod +x /home/jupyter/save_models.sh
chown jupyter:jupyter /home/jupyter/save_models.sh

# Install and start auto-shutdown monitor
echo "🤖 Setting up auto-shutdown for cost optimization..."
cat > /home/jupyter/auto-shutdown.sh << 'EOF'
#!/bin/bash
# Auto-shutdown script - monitors for idle periods
IDLE_THRESHOLD=3600  # 1 hour
CHECK_INTERVAL=300   # 5 minutes

while true; do
    PYTHON_PROCESSES=$(pgrep -f "python.*enhanced_main.py|python.*train|jupyter" | wc -l)
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}' | cut -d'.' -f1)
    
    GPU_USAGE=0
    if command -v nvidia-smi &> /dev/null; then
        GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n1 2>/dev/null || echo "0")
    fi
    
    echo "$(date): Processes: $PYTHON_PROCESSES, CPU: ${CPU_USAGE}%, GPU: ${GPU_USAGE}%"
    
    if [ "$PYTHON_PROCESSES" -eq 0 ] && [ "$CPU_USAGE" -lt 10 ] && [ "$GPU_USAGE" -lt 5 ]; then
        echo "$(date): System idle. Shutting down in 10 minutes (cancel with: sudo pkill -f auto-shutdown)"
        sleep 600  # 10 minute grace period
        
        # Final check
        PYTHON_PROCESSES=$(pgrep -f "python.*enhanced_main.py|python.*train|jupyter" | wc -l)
        if [ "$PYTHON_PROCESSES" -eq 0 ]; then
            echo "$(date): Auto-shutdown - saving costs"
            sudo shutdown -h now
        fi
    fi
    
    sleep $CHECK_INTERVAL
done
EOF

chmod +x /home/jupyter/auto-shutdown.sh
chown jupyter:jupyter /home/jupyter/auto-shutdown.sh

# Start auto-shutdown in background
nohup sudo -u jupyter /home/jupyter/auto-shutdown.sh > /var/log/auto-shutdown.log 2>&1 &

# Create manual control scripts
cat > /home/jupyter/disable-auto-shutdown.sh << 'EOF'
#!/bin/bash
echo "Disabling auto-shutdown..."
sudo pkill -f auto-shutdown
echo "Auto-shutdown disabled. Remember to manually stop the VM when done!"
EOF

chmod +x /home/jupyter/disable-auto-shutdown.sh
chown jupyter:jupyter /home/jupyter/disable-auto-shutdown.sh

echo "✅ Auto-shutdown monitor started (logs: /var/log/auto-shutdown.log)"
echo "🎉 Startup script completed successfully! $(date)"
echo "📝 Next steps:"
echo "1. SSH to this VM and clone your trading bot repository"
echo "2. Run ./start_training.sh to begin ML training" 
echo "3. VM will auto-shutdown after 1h of inactivity to save costs"
echo "4. Disable auto-shutdown: ./disable-auto-shutdown.sh"
echo "5. Access Jupyter notebook at http://$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip -H \"Metadata-Flavor: Google\"):8888"