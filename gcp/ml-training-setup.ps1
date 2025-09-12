# ML Training Setup with GPU on GCP Compute Engine - PowerShell Version
# This script creates a GPU-enabled VM for intensive ML training

param(
    [string]$ProjectId = $env:GCP_PROJECT_ID,
    [string]$Region = $env:GCP_REGION,
    [string]$Zone = $env:GCP_ZONE
)

# Configuration avec valeurs par défaut
if (-not $ProjectId) { $ProjectId = "multi-asset-trading-framework" }
if (-not $Region) { $Region = "europe-west1" }  
if (-not $Zone) { $Zone = "europe-west1-b" }

$VmName = "trading-ml-trainer"
$MachineType = "n1-standard-4"
$GpuType = "nvidia-tesla-t4"
$GpuCount = 1
$DiskSize = "100GB"
$DiskType = "pd-ssd"
$ImageFamily = "pytorch-latest-gpu"
$ImageProject = "deeplearning-platform-release"

Write-Host "🔥 Setting up ML Training VM with GPU..." -ForegroundColor Green
Write-Host "Project: $ProjectId" -ForegroundColor Cyan
Write-Host "Zone: $Zone" -ForegroundColor Cyan
Write-Host "Machine Type: $MachineType" -ForegroundColor Cyan
Write-Host "GPU: $GpuCount x $GpuType" -ForegroundColor Cyan

# Check GPU quota
Write-Host "📊 Checking GPU quota..." -ForegroundColor Yellow
$quotaCheck = gcloud compute regions describe $Region --format="value(quotas[metric=NVIDIA_T4_GPUS].limit)"
if ($quotaCheck -eq "0" -or -not $quotaCheck) {
    Write-Host "❌ Error: No GPU quota available in region $Region" -ForegroundColor Red
    Write-Host "Please request GPU quota increase in the GCP Console:" -ForegroundColor Yellow
    Write-Host "https://console.cloud.google.com/iam-admin/quotas" -ForegroundColor Blue
    exit 1
}

Write-Host "✅ GPU quota available: $quotaCheck" -ForegroundColor Green

# Create the VM instance
Write-Host "🖥️ Creating GPU-enabled VM instance..." -ForegroundColor Yellow
$vmExists = gcloud compute instances describe $VmName --zone=$Zone 2>$null
if (-not $vmExists) {
    gcloud compute instances create $VmName `
        --zone=$Zone `
        --machine-type=$MachineType `
        --accelerator=type=$GpuType,count=$GpuCount `
        --image-family=$ImageFamily `
        --image-project=$ImageProject `
        --boot-disk-size=$DiskSize `
        --boot-disk-type=$DiskType `
        --maintenance-policy=TERMINATE `
        --restart-on-failure `
        --scopes=https://www.googleapis.com/auth/cloud-platform `
        --tags=ml-training `
        --metadata=install-nvidia-driver=True `
        --metadata-from-file=startup-script=gcp/startup-script.sh

    Write-Host "✅ VM instance created: $VmName" -ForegroundColor Green
} else {
    Write-Host "ℹ️ VM instance already exists: $VmName" -ForegroundColor Blue
}

# Wait for instance to be ready
Write-Host "⏳ Waiting for instance to be ready..." -ForegroundColor Yellow
gcloud compute instances wait-until-ready $VmName --zone=$Zone

# Get external IP
$ExternalIp = gcloud compute instances describe $VmName --zone=$Zone --format='value(networkInterfaces[0].accessConfigs[0].natIP)'
Write-Host "📡 External IP: $ExternalIp" -ForegroundColor Cyan

# Create firewall rule for ML training (if needed)
$firewallExists = gcloud compute firewall-rules describe allow-ml-training 2>$null
if (-not $firewallExists) {
    gcloud compute firewall-rules create allow-ml-training `
        --direction=INGRESS `
        --priority=1000 `
        --network=default `
        --action=ALLOW `
        --rules=tcp:8888,tcp:8000 `
        --source-ranges=0.0.0.0/0 `
        --target-tags=ml-training
    
    Write-Host "✅ Firewall rule created for ML training" -ForegroundColor Green
} else {
    Write-Host "ℹ️ Firewall rule already exists" -ForegroundColor Blue
}

Write-Host ""
Write-Host "🎉 ML Training VM setup completed!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 VM Information:" -ForegroundColor Cyan
Write-Host "==================" -ForegroundColor Cyan
Write-Host "• Instance Name: $VmName" -ForegroundColor White
Write-Host "• External IP: $ExternalIp" -ForegroundColor White
Write-Host "• Zone: $Zone" -ForegroundColor White
Write-Host "• Machine Type: $MachineType" -ForegroundColor White
Write-Host "• GPU: $GpuCount x $GpuType" -ForegroundColor White
Write-Host ""
Write-Host "📝 Next Steps:" -ForegroundColor Cyan
Write-Host "================" -ForegroundColor Cyan
Write-Host "1. SSH to the instance:" -ForegroundColor White
Write-Host "   gcloud compute ssh $VmName --zone=$Zone" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Clone your trading bot repository and run ML training:" -ForegroundColor White
Write-Host "   git clone <your-repo-url>" -ForegroundColor Gray
Write-Host "   cd trading-bot" -ForegroundColor Gray
Write-Host "   python enhanced_main.py" -ForegroundColor Gray
Write-Host ""
Write-Host "3. To copy trained models back to Cloud Storage:" -ForegroundColor White
Write-Host "   gsutil cp trained_models/* gs://$ProjectId-trading-data/models/" -ForegroundColor Gray
Write-Host ""
Write-Host "💡 The VM includes:" -ForegroundColor Yellow
Write-Host "   - NVIDIA drivers and CUDA" -ForegroundColor White
Write-Host "   - PyTorch with GPU support" -ForegroundColor White
Write-Host "   - Jupyter notebook (port 8888)" -ForegroundColor White
Write-Host "   - All Python ML libraries" -ForegroundColor White
Write-Host ""
Write-Host "💰 Cost Optimization:" -ForegroundColor Yellow
Write-Host "   - Stop the VM when not training:" -ForegroundColor White
Write-Host "     gcloud compute instances stop $VmName --zone=$Zone" -ForegroundColor Gray
Write-Host "   - Start when needed:" -ForegroundColor White
Write-Host "     gcloud compute instances start $VmName --zone=$Zone" -ForegroundColor Gray

# Create helper PowerShell scripts
Write-Host ""
Write-Host "📝 Creating helper PowerShell scripts..." -ForegroundColor Yellow

# Stop VM script
$stopScript = @"
# Stop Trading ML VM - Windows PowerShell
param([string]`$Zone = "$Zone")
Write-Host "Stopping ML training VM..." -ForegroundColor Yellow
gcloud compute instances stop $VmName --zone=`$Zone
Write-Host "✅ VM stopped successfully!" -ForegroundColor Green
"@

$stopScript | Out-File -FilePath "Stop-TradingVM.ps1" -Encoding UTF8
Write-Host "Created: Stop-TradingVM.ps1" -ForegroundColor Green

# Start VM script  
$startScript = @"
# Start Trading ML VM - Windows PowerShell
param([string]`$Zone = "$Zone")
Write-Host "Starting ML training VM..." -ForegroundColor Yellow
gcloud compute instances start $VmName --zone=`$Zone
Write-Host "⏳ Waiting for VM to be ready..." -ForegroundColor Yellow
gcloud compute instances wait-until-ready $VmName --zone=`$Zone
`$ip = gcloud compute instances describe $VmName --zone=`$Zone --format='value(networkInterfaces[0].accessConfigs[0].natIP)'
Write-Host "✅ VM started successfully!" -ForegroundColor Green
Write-Host "📡 External IP: `$ip" -ForegroundColor Cyan
Write-Host "🔗 SSH command: gcloud compute ssh $VmName --zone=`$Zone" -ForegroundColor Blue
"@

$startScript | Out-File -FilePath "Start-TradingVM.ps1" -Encoding UTF8  
Write-Host "Created: Start-TradingVM.ps1" -ForegroundColor Green

Write-Host ""
Write-Host "💻 Usage on Windows:" -ForegroundColor Cyan
Write-Host "- To stop VM:  .\Stop-TradingVM.ps1" -ForegroundColor White
Write-Host "- To start VM: .\Start-TradingVM.ps1" -ForegroundColor White