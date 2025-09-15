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
Write-Host "[INFO] Checking Cloud Storage bucket..." -ForegroundColor Yellow

# First, check if any bucket for this project already exists
$existingBuckets = gcloud storage buckets list --project=$ProjectId --format="value(name)" 2>$null | Where-Object { $_ -match "trading-data" }

if ($existingBuckets) {
    # Use the first existing trading-data bucket
    $BucketName = ($existingBuckets[0] -replace "gs://", "" -replace "/", "")
    Write-Host "[INFO] Using existing storage bucket: gs://$BucketName" -ForegroundColor Blue
} else {
    # Try to create the main bucket name
    $bucketResult = gcloud storage buckets create gs://$BucketName --location=$Region --project=$ProjectId 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[SUCCESS] Storage bucket created: gs://$BucketName" -ForegroundColor Green
    } else {
        if ($bucketResult -match "already exists" -or $bucketResult -match "globally unique") {
            Write-Host "[WARNING] Bucket name already exists globally. Trying with timestamp suffix..." -ForegroundColor Yellow
            $timestamp = Get-Date -Format "yyyyMMdd-HHmm"
            $BucketName = "$ProjectId-trading-data-$timestamp"
            gcloud storage buckets create gs://$BucketName --location=$Region --project=$ProjectId
            Write-Host "[SUCCESS] Storage bucket created: gs://$BucketName" -ForegroundColor Green
        } else {
            Write-Host "[ERROR] Failed to create bucket: $bucketResult" -ForegroundColor Red
            exit 1
        }
    }
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
        --size=1 `
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

# Check if Docker is available and running
$dockerCheck = docker --version 2>$null
if (-not $dockerCheck) {
    Write-Host "[ERROR] Docker not found in PATH." -ForegroundColor Red
    Write-Host "Please ensure Docker Desktop is installed and running." -ForegroundColor Yellow
    exit 1
}

# Test if Docker daemon is running
$dockerInfo = docker info 2>$null
if (-not $dockerInfo) {
    Write-Host "[ERROR] Docker daemon is not running." -ForegroundColor Red
    Write-Host "Please start Docker Desktop and wait for it to fully initialize." -ForegroundColor Yellow
    Write-Host "You should see the Docker icon in your system tray when it's ready." -ForegroundColor Yellow
    exit 1
}

Write-Host "[INFO] Docker is running and ready." -ForegroundColor Green

# Option 1: Use Cloud Build (recommended to save local disk space)
$useCloudBuild = $true

if ($useCloudBuild) {
    Write-Host "[INFO] Using Cloud Build to avoid local Docker build..." -ForegroundColor Yellow
    Write-Host "[INFO] This will save disk space on your local machine" -ForegroundColor Green
    
    # Submit to Cloud Build with real-time logs
    gcloud builds submit --tag $ImageUrl --timeout=20m --machine-type=e2-highcpu-32 .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Cloud Build failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "[SUCCESS] Cloud Build completed and image pushed: $ImageUrl" -ForegroundColor Green
    
} else {
    Write-Host "[INFO] Building Docker image locally with progress logs..." -ForegroundColor Yellow
    
    # Build with progress and live output
    Write-Host "[INFO] Starting Docker build (this may take several minutes)..." -ForegroundColor Yellow
    docker build --progress=plain -t $ImageUrl . 
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Docker build failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "[SUCCESS] Docker image built successfully" -ForegroundColor Green

    Write-Host "[INFO] Pushing image to Artifact Registry..." -ForegroundColor Yellow
    docker push $ImageUrl
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Docker push failed" -ForegroundColor Red
        exit 1
    }
}

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

# Grant Secret Manager access to Cloud Run service account
Write-Host "[INFO] Granting Secret Manager access to Cloud Run service account..." -ForegroundColor Yellow
$projectNumber = gcloud projects describe $ProjectId --format="value(projectNumber)"
$cloudRunServiceAccount = "$projectNumber-compute@developer.gserviceaccount.com"

gcloud projects add-iam-policy-binding $ProjectId `
    --member="serviceAccount:$cloudRunServiceAccount" `
    --role="roles/secretmanager.secretAccessor"

# Grant Cloud SQL Client role
Write-Host "[INFO] Granting Cloud SQL Client role to Cloud Run service account..." -ForegroundColor Yellow
gcloud projects add-iam-policy-binding $ProjectId `
    --member="serviceAccount:$cloudRunServiceAccount" `
    --role="roles/cloudsql.client"

Write-Host "[SUCCESS] Secret Manager permissions granted to Cloud Run service account" -ForegroundColor Green

# Get connection details
Write-Host "[INFO] Getting connection details..." -ForegroundColor Yellow
$DbConnectionName = gcloud sql instances describe $DbInstanceName --format="value(connectionName)"

# Get Redis details only if instance exists
$RedisHost = ""
$RedisPort = ""
$redisInstanceCheck = gcloud redis instances describe $RedisInstanceName --region=$Region --format="value(host)" 2>$null
if ($redisInstanceCheck) {
    $RedisHost = $redisInstanceCheck
    $RedisPort = gcloud redis instances describe $RedisInstanceName --region=$Region --format="value(port)"
    Write-Host "[INFO] Redis instance found - Host: $RedisHost, Port: $RedisPort" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Redis instance not accessible yet. You may need to wait and update environment variables manually." -ForegroundColor Yellow
    $RedisHost = "localhost"
    $RedisPort = "6379"
}

# Deploy to Cloud Run
Write-Host "[INFO] Deploying to Cloud Run..." -ForegroundColor Yellow
gcloud run deploy $ServiceName `
    --image=$ImageUrl `
    --platform=managed `
    --region=$Region `
    --allow-unauthenticated `
    --memory=8Gi `
    --cpu=4 `
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
    --set-env-vars="ENABLE_KNOWLEDGE_GRAPH=true" `
    --set-env-vars="SKIP_ML_TRAINING=true" `
    --set-env-vars="PYTHONUNBUFFERED=1" `
    --set-env-vars="PYTHONPATH=/app" `
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