# Trading Bot GCP Deployment Script - Windows PowerShell Version
# This script sets up the entire GCP infrastructure for the trading bot

param(
    [string]$ProjectId = $env:GCP_PROJECT_ID,
    [string]$Region = $env:GCP_REGION,
    [string]$Zone = $env:GCP_ZONE
)

# Configuration avec valeurs par defaut depuis .env
if (-not $ProjectId) { $ProjectId = "multi-asset-trading-framework" }
if (-not $Region) { $Region = "europe-west1" }
if (-not $Zone) { $Zone = "europe-west1-b" }

$ServiceName = "trading-bot"
$DbInstanceName = "trading-db-instance"
$RedisInstanceName = "trading-redis-instance"
$BucketName = "$ProjectId-trading-data"

Write-Host "[INFO] Starting Trading Bot deployment to GCP..." -ForegroundColor Green
Write-Host "Project: $ProjectId" -ForegroundColor Cyan
Write-Host "Region: $Region" -ForegroundColor Cyan
Write-Host "Zone: $Zone" -ForegroundColor Cyan

# Check if gcloud is installed and authenticated
try {
    $gcloudVersion = gcloud version 2>$null
    if (-not $gcloudVersion) {
        throw "gcloud not found"
    }
}
catch {
    Write-Host "[ERROR] gcloud CLI is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install it from: https://cloud.google.com/sdk/docs/install-windows" -ForegroundColor Yellow
    exit 1
}

# Set the project
Write-Host "[INFO] Setting GCP project..." -ForegroundColor Yellow
gcloud config set project $ProjectId
gcloud config set compute/region $Region
gcloud config set compute/zone $Zone

# Skip API enabling since they're already activated manually
Write-Host "[INFO] APIs already activated manually via console" -ForegroundColor Green

# Create Cloud Storage bucket for data and models
Write-Host "[INFO] Creating Cloud Storage bucket..." -ForegroundColor Yellow
$bucketExists = gsutil ls gs://$BucketName 2>$null
if (-not $bucketExists) {
    gsutil mb -l $Region gs://$BucketName
    Write-Host "[SUCCESS] Storage bucket created: gs://$BucketName" -ForegroundColor Green
} else {
    Write-Host "[INFO] Storage bucket already exists: gs://$BucketName" -ForegroundColor Blue
}

# Create Cloud SQL instance (PostgreSQL)
Write-Host "[INFO] Creating Cloud SQL PostgreSQL instance..." -ForegroundColor Yellow
$sqlExists = gcloud sql instances describe $DbInstanceName 2>$null
if (-not $sqlExists) {
    gcloud sql instances create $DbInstanceName `
        --database-version=POSTGRES_15 `
        --tier=db-g1-small `
        --region=$Region `
        --storage-type=SSD `
        --storage-size=20GB `
        --storage-auto-increase `
        --backup-start-time=03:00 `
        --maintenance-window-day=SUN `
        --maintenance-window-hour=04 `
        --deletion-protection

    Write-Host "[SUCCESS] Cloud SQL instance created: $DbInstanceName" -ForegroundColor Green
    
    # Set root password
    Write-Host "[INFO] Setting database root password..." -ForegroundColor Yellow
    $password = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 32 | ForEach-Object {[char]$_})
    gcloud sql users set-password postgres --instance=$DbInstanceName --password=$password
    
    # Create trading database
    Write-Host "[INFO] Creating trading database..." -ForegroundColor Yellow
    gcloud sql databases create trading_bot --instance=$DbInstanceName
} else {
    Write-Host "[INFO] Cloud SQL instance already exists: $DbInstanceName" -ForegroundColor Blue
}

# Create Memorystore Redis instance
Write-Host "[INFO] Creating Memorystore Redis instance..." -ForegroundColor Yellow
$redisExists = gcloud redis instances describe $RedisInstanceName --region=$Region 2>$null
if (-not $redisExists) {
    gcloud redis instances create $RedisInstanceName `
        --region=$Region `
        --size=1GB `
        --redis-version=redis_7_0
    
    Write-Host "[SUCCESS] Memorystore Redis instance created: $RedisInstanceName" -ForegroundColor Green
} else {
    Write-Host "[INFO] Memorystore Redis instance already exists: $RedisInstanceName" -ForegroundColor Blue
}

# Create Artifact Registry repository
Write-Host "[INFO] Creating Artifact Registry repository..." -ForegroundColor Yellow
$repoExists = gcloud artifacts repositories describe trading-bot --location=$Region 2>$null
if (-not $repoExists) {
    gcloud artifacts repositories create trading-bot `
        --repository-format=docker `
        --location=$Region `
        --description="Trading Bot Docker images"
    
    Write-Host "[SUCCESS] Artifact Registry repository created" -ForegroundColor Green
} else {
    Write-Host "[INFO] Artifact Registry repository already exists" -ForegroundColor Blue
}

# Configure Docker authentication for Artifact Registry
Write-Host "[INFO] Configuring Docker authentication..." -ForegroundColor Yellow
gcloud auth configure-docker "$Region-docker.pkg.dev"

# Build and push Docker image
Write-Host "[INFO] Building and pushing Docker image..." -ForegroundColor Yellow
$ImageUrl = "$Region-docker.pkg.dev/$ProjectId/trading-bot/$ServiceName" + ":latest"

# Check if Docker is available
$dockerCheck = docker --version 2>$null
if (-not $dockerCheck) {
    Write-Host "[WARNING] Docker not found in PATH." -ForegroundColor Yellow
    Write-Host "Please ensure Docker Desktop is installed and running." -ForegroundColor Yellow
    Write-Host "You may need to restart PowerShell after installing Docker." -ForegroundColor Yellow
    
    # Try to find Docker in common locations
    $dockerPaths = @(
        "C:\Program Files\Docker\Docker\resources\bin\docker.exe",
        "C:\ProgramData\DockerDesktop\version-bin\docker.exe"
    )
    
    foreach ($path in $dockerPaths) {
        if (Test-Path $path) {
            Write-Host "Found Docker at: $path" -ForegroundColor Yellow
            Write-Host "Consider adding it to your PATH or restart PowerShell" -ForegroundColor Yellow
            break
        }
    }
    
    $continue = Read-Host "Continue anyway? (y/N)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        exit 1
    }
}

docker build -t $ImageUrl .
docker push $ImageUrl

Write-Host "[SUCCESS] Docker image pushed: $ImageUrl" -ForegroundColor Green

# Create secrets for API keys
Write-Host "[INFO] Creating secrets in Secret Manager..." -ForegroundColor Yellow

function Create-Secret-If-Not-Exists {
    param(
        [string]$SecretName,
        [string]$SecretValue
    )
    
    $secretExists = gcloud secrets describe $SecretName 2>$null
    if (-not $secretExists) {
        echo $SecretValue | gcloud secrets create $SecretName --data-file=-
        Write-Host "[SUCCESS] Created secret: $SecretName" -ForegroundColor Green
    } else {
        Write-Host "[INFO] Secret already exists: $SecretName" -ForegroundColor Blue
    }
}

# Load values from .env file
$envFile = Get-Content .env -ErrorAction SilentlyContinue
$envVars = @{}
if ($envFile) {
    foreach ($line in $envFile) {
        if ($line -match '^([^=]+)=(.*)$') {
            $envVars[$matches[1]] = $matches[2]
        }
    }
}

# Create secrets with values from .env
Create-Secret-If-Not-Exists "gemini-api-key" $envVars["GEMINI_API_KEY"]
Create-Secret-If-Not-Exists "news-api-key" $envVars["NEWS_API_KEY"] 
Create-Secret-If-Not-Exists "alpha-vantage-key" $envVars["ALPHA_VANTAGE_KEY"]
Create-Secret-If-Not-Exists "finnhub-key" $envVars["FINNHUB_KEY"]

# Get connection details
Write-Host "[INFO] Getting connection details..." -ForegroundColor Yellow
$DbConnectionName = gcloud sql instances describe $DbInstanceName --format="value(connectionName)"
$RedisHost = gcloud redis instances describe $RedisInstanceName --region=$Region --format="value(host)"
$RedisPort = gcloud redis instances describe $RedisInstanceName --region=$Region --format="value(port)"

# Deploy to Cloud Run
Write-Host "[INFO] Deploying to Cloud Run..." -ForegroundColor Yellow
gcloud run deploy $ServiceName `
    --image=$ImageUrl `
    --platform=managed `
    --region=$Region `
    --allow-unauthenticated `
    --memory=2Gi `
    --cpu=2 `
    --timeout=3600 `
    --concurrency=1 `
    --max-instances=1 `
    --set-env-vars="DATABASE_URL=postgresql://postgres:@/$DbConnectionName/trading_bot" `
    --set-env-vars="REDIS_URL=redis://$($RedisHost):$RedisPort" `
    --set-env-vars="TRADING_MODE=fast_mode" `
    --set-env-vars="ANALYSIS_LOOKBACK_DAYS=90" `
    --set-env-vars="ML_TRAINING_LOOKBACK_DAYS=3650" `
    --set-env-vars="NEWS_LOOKBACK_DAYS=60" `
    --set-env-vars="ENABLE_TRADITIONAL_ML=true" `
    --set-env-vars="ENABLE_TRANSFORMER_ML=false" `
    --set-secrets="GEMINI_API_KEY=gemini-api-key:latest" `
    --set-secrets="NEWS_API_KEY=news-api-key:latest" `
    --set-secrets="ALPHA_VANTAGE_KEY=alpha-vantage-key:latest" `
    --set-secrets="FINNHUB_KEY=finnhub-key:latest" `
    --add-cloudsql-instances=$DbConnectionName

Write-Host "[SUCCESS] Cloud Run service deployed successfully!" -ForegroundColor Green

# Output important information
$serviceUrl = gcloud run services describe $ServiceName --region=$Region --format='value(status.url)'

Write-Host ""
Write-Host "[SUCCESS] Deployment completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Important Information:" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan
Write-Host "* Cloud Run Service URL: $serviceUrl" -ForegroundColor White
Write-Host "* Cloud SQL Connection: $DbConnectionName" -ForegroundColor White  
Write-Host "* Redis Host: $($RedisHost):$RedisPort" -ForegroundColor White
Write-Host "* Storage Bucket: gs://$BucketName" -ForegroundColor White
Write-Host "* Docker Image: $ImageUrl" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "===========" -ForegroundColor Cyan
Write-Host "1. Update API key secrets if needed:" -ForegroundColor White
Write-Host "   gcloud secrets versions add gemini-api-key --data-file=<key-file>" -ForegroundColor Gray
Write-Host "2. Set up ML training on Compute Engine with GPU:" -ForegroundColor White
Write-Host "   .\gcp\ml-training-setup.ps1" -ForegroundColor Gray
Write-Host "3. Configure monitoring and alerting" -ForegroundColor White
Write-Host "4. Set up CI/CD pipeline" -ForegroundColor White
Write-Host ""
Write-Host "For ML training with GPU, use the Compute Engine deployment script." -ForegroundColor Yellow