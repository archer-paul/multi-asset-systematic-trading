#!/bin/bash
# Auto-shutdown script for ML training VM
# This script is installed on the VM and shuts it down after inactivity

IDLE_THRESHOLD=3600  # 1 hour in seconds
CHECK_INTERVAL=300   # Check every 5 minutes

echo "$(date): Starting auto-shutdown monitor (idle threshold: ${IDLE_THRESHOLD}s)"

while true; do
    # Check if any Python/training processes are running
    PYTHON_PROCESSES=$(pgrep -f "python.*enhanced_main.py|python.*train|jupyter" | wc -l)
    
    # Check CPU usage (if below 10% for extended period)
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    CPU_USAGE=${CPU_USAGE%.*}  # Remove decimal part
    
    # Check GPU usage (if nvidia-smi is available)
    GPU_USAGE=0
    if command -v nvidia-smi &> /dev/null; then
        GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n1)
    fi
    
    echo "$(date): Python processes: $PYTHON_PROCESSES, CPU: ${CPU_USAGE}%, GPU: ${GPU_USAGE}%"
    
    # Shutdown conditions:
    # 1. No Python training processes running AND
    # 2. CPU usage below 10% AND 
    # 3. GPU usage below 5%
    if [ "$PYTHON_PROCESSES" -eq 0 ] && [ "$CPU_USAGE" -lt 10 ] && [ "$GPU_USAGE" -lt 5 ]; then
        echo "$(date): System appears idle. Shutting down in 5 minutes..."
        echo "$(date): Cancel with: sudo pkill -f auto-shutdown"
        
        # Wait 5 more minutes before shutdown (grace period)
        sleep 300
        
        # Re-check before final shutdown
        PYTHON_PROCESSES=$(pgrep -f "python.*enhanced_main.py|python.*train|jupyter" | wc -l)
        if [ "$PYTHON_PROCESSES" -eq 0 ]; then
            echo "$(date): Final check - no training processes. Shutting down now."
            sudo shutdown -h now
        else
            echo "$(date): Training process detected during grace period. Continuing..."
        fi
    fi
    
    sleep $CHECK_INTERVAL
done