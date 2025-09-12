#!/bin/bash

# ML Training Setup with GPU on GCP Compute Engine
# This script creates a GPU-enabled VM for intensive ML training

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"trading-bot-project"}
REGION=${GCP_REGION:-"europe-west1"}
ZONE=${GCP_ZONE:-"europe-west1-b"}
VM_NAME="trading-ml-trainer"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
DISK_SIZE="100GB"
DISK_TYPE="pd-ssd"
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"

echo "🔥 Setting up ML Training VM with GPU..."
echo "Project: $PROJECT_ID"
echo "Zone: $ZONE"
echo "Machine Type: $MACHINE_TYPE"
echo "GPU: $GPU_COUNT x $GPU_TYPE"

# Check GPU quota
echo "📊 Checking GPU quota..."
QUOTA_CHECK=$(gcloud compute regions describe $REGION --format="value(quotas[metric=NVIDIA_T4_GPUS].limit)")
if [[ "$QUOTA_CHECK" == "0" ]]; then
    echo "❌ Error: No GPU quota available in region $REGION"
    echo "Please request GPU quota increase in the GCP Console:"
    echo "https://console.cloud.google.com/iam-admin/quotas"
    exit 1
fi

echo "✅ GPU quota available: $QUOTA_CHECK"

# Create the VM instance
echo "🖥️ Creating GPU-enabled VM instance..."
if ! gcloud compute instances describe $VM_NAME --zone=$ZONE &> /dev/null; then
    gcloud compute instances create $VM_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --accelerator=type=$GPU_TYPE,count=$GPU_COUNT \
        --image-family=$IMAGE_FAMILY \
        --image-project=$IMAGE_PROJECT \
        --boot-disk-size=$DISK_SIZE \
        --boot-disk-type=$DISK_TYPE \
        --maintenance-policy=TERMINATE \
        --restart-on-failure \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --tags=ml-training \
        --metadata=install-nvidia-driver=True \
        --metadata-from-file startup-script=startup-script.sh

    echo "✅ VM instance created: $VM_NAME"
else
    echo "ℹ️ VM instance already exists: $VM_NAME"
fi

# Wait for instance to be ready
echo "⏳ Waiting for instance to be ready..."
gcloud compute instances wait-until-ready $VM_NAME --zone=$ZONE

# Get external IP
EXTERNAL_IP=$(gcloud compute instances describe $VM_NAME --zone=$ZONE --format='value(networkInterfaces[0].accessConfigs[0].natIP)')
echo "📡 External IP: $EXTERNAL_IP"

# Create firewall rule for ML training (if needed)
if ! gcloud compute firewall-rules describe allow-ml-training &> /dev/null; then
    gcloud compute firewall-rules create allow-ml-training \
        --direction=INGRESS \
        --priority=1000 \
        --network=default \
        --action=ALLOW \
        --rules=tcp:8888,tcp:8000 \
        --source-ranges=0.0.0.0/0 \
        --target-tags=ml-training
    
    echo "✅ Firewall rule created for ML training"
fi

echo ""
echo "🎉 ML Training VM setup completed!"
echo ""
echo "📋 VM Information:"
echo "=================="
echo "• Instance Name: $VM_NAME"
echo "• External IP: $EXTERNAL_IP"
echo "• Zone: $ZONE"
echo "• Machine Type: $MACHINE_TYPE"
echo "• GPU: $GPU_COUNT x $GPU_TYPE"
echo ""
echo "📝 Next Steps:"
echo "=============="
echo "1. SSH to the instance:"
echo "   gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "2. Clone your trading bot repository and run ML training:"
echo "   git clone <your-repo-url>"
echo "   cd trading-bot"
echo "   python ml_training_script.py"
echo ""
echo "3. To copy trained models back to Cloud Storage:"
echo "   gsutil cp trained_models/* gs://${PROJECT_ID}-trading-data/models/"
echo ""
echo "💡 The VM includes:"
echo "   - NVIDIA drivers and CUDA"
echo "   - PyTorch with GPU support"
echo "   - Jupyter notebook (port 8888)"
echo "   - All Python ML libraries"
echo ""
echo "💰 Cost Optimization:"
echo "   - Stop the VM when not training: gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo "   - Start when needed: gcloud compute instances start $VM_NAME --zone=$ZONE"