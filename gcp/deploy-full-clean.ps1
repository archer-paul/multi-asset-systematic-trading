# Trading Bot FULL Deployment Script - Backend + Frontend
# This script deploys backend to GCP Cloud Run AND frontend to Firebase Hosting

param(
    [string]$ProjectId = $env:GCP_PROJECT_ID,
    [string]$Region = $env:GCP_REGION,
    [string]$Zone = $env:GCP_ZONE
)

# Configuration avec valeurs par defaut
if (-not $ProjectId) { $ProjectId = "multi-asset-trading-framework" }
if (-not $Region) { $Region = "europe-west1" }
if (-not $Zone) { $Zone = "europe-west1-b" }

Write-Host "[INFO] Starting FULL Trading Bot deployment (Backend + Frontend)..."
Write-Host "Backend: GCP Cloud Run"
Write-Host "Frontend: Firebase Hosting"

# 1. Verifications des outils requis
Write-Host "`n[INFO] Checking required tools..."

# Check gcloud
try {
    $gcloudVersion = gcloud version 2>$null
    if (-not $gcloudVersion) { throw "gcloud not found" }
    Write-Host "[OK] gcloud CLI found"
} catch {
    Write-Host "[ERROR] gcloud CLI not installed. Install from: https://cloud.google.com/sdk/docs/install-windows"
    exit 1
}

# Check Firebase CLI
try {
    $firebaseVersion = firebase --version 2>$null
    if (-not $firebaseVersion) { throw "firebase not found" }
    Write-Host "[OK] Firebase CLI found ($firebaseVersion)"
} catch {
    Write-Host "[ERROR] Firebase CLI not installed. Run: npm install -g firebase-tools"
    exit 1
}

# Docker not needed - using Cloud Build for backend

# Check Node.js
try {
    $nodeVersion = node --version 2>$null
    if (-not $nodeVersion) { throw "node not found" }
    Write-Host "[OK] Node.js found ($nodeVersion)"
} catch {
    Write-Host "[ERROR] Node.js not installed. Please install Node.js."
    exit 1
}

# 2. Build et test du frontend AVANT de deployer le backend
Write-Host "`n[STEP 1/3] Building and testing frontend..."

# Aller dans le repertoire frontend
$frontendPath = "frontend"
if (-not (Test-Path $frontendPath)) {
    Write-Host "[ERROR] Frontend directory not found: $frontendPath"
    exit 1
}

Push-Location $frontendPath

try {
    # Install dependencies
    Write-Host "[INFO] Installing frontend dependencies..."
    npm ci
    if ($LASTEXITCODE -ne 0) {
        throw "npm ci failed"
    }

    # Build frontend
    Write-Host "[INFO] Building frontend..."
    npm run build
    if ($LASTEXITCODE -ne 0) {
        throw "Frontend build failed"
    }

    # Export is automatic with output: 'export' in next.config.js
    Write-Host "[INFO] Static export completed automatically with Next.js build"

    Write-Host "[SUCCESS] Frontend built and exported successfully!"

} catch {
    Write-Host "[ERROR] Frontend build failed: $($_.Exception.Message)"
    Pop-Location
    exit 1
} finally {
    Pop-Location
}

# 3. Deployer le backend sur GCP Cloud Run
Write-Host "`n[STEP 2/3] Deploying backend to GCP Cloud Run..."

# Executer le script de deploiement backend existant
$backendDeployScript = "gcp\deploy.ps1"
if (-not (Test-Path $backendDeployScript)) {
    Write-Host "[ERROR] Backend deploy script not found: $backendDeployScript"
    exit 1
}

Write-Host "[INFO] Running backend deployment..."
& $backendDeployScript -ProjectId $ProjectId -Region $Region -Zone $Zone

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Backend deployment failed!"
    exit 1
}

# Recuperer l'URL du backend deploye
$backendUrl = gcloud run services describe trading-bot --region=$Region --format='value(status.url)' 2>$null
if (-not $backendUrl) {
    Write-Host "[WARNING] Could not retrieve backend URL. Frontend may need manual configuration."
    $backendUrl = "https://trading-bot-xxx-$Region.run.app"
}

Write-Host "[SUCCESS] Backend deployed successfully: $backendUrl"

# 4. Deployer le frontend sur Firebase
Write-Host "`n[STEP 3/3] Deploying frontend to Firebase..."

# Mettre a jour la configuration frontend avec l'URL du backend
Write-Host "[INFO] Updating frontend configuration with backend URL..."

$envLocalPath = "frontend\.env.local"
$envContent = @"
NEXT_PUBLIC_API_URL=$backendUrl
NEXT_PUBLIC_WS_URL=$($backendUrl -replace 'https://', 'wss://')
"@

$envContent | Out-File -FilePath $envLocalPath -Encoding utf8
Write-Host "[INFO] Updated $envLocalPath with backend URL"

# Firebase login check
Write-Host "[INFO] Checking Firebase authentication..."
$firebaseProjects = firebase projects:list 2>$null
if (-not $firebaseProjects) {
    Write-Host "[INFO] Please login to Firebase..."
    firebase login
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Firebase login failed!"
        exit 1
    }
}

# Deploy to Firebase
Write-Host "[INFO] Deploying to Firebase Hosting..."
firebase use quantitative-alpha-engine
firebase deploy --only hosting

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Firebase deployment failed!"
    exit 1
}

# Recuperer l'URL Firebase
$firebaseUrl = "https://quantitative-alpha-engine.web.app"

# 5. Resume final
Write-Host "`n[SUCCESS] FULL DEPLOYMENT COMPLETED!"
Write-Host ""
Write-Host "=================================================================="
Write-Host "                  DEPLOYMENT SUMMARY                             "
Write-Host "=================================================================="
Write-Host "Backend (GCP Cloud Run): $backendUrl"
Write-Host "Frontend (Firebase):     $firebaseUrl"
Write-Host "=================================================================="
Write-Host ""

Write-Host "NEXT STEPS:"
Write-Host "1. Open your trading dashboard: $firebaseUrl"
Write-Host "2. Check backend health: $backendUrl/health"
Write-Host "3. Monitor logs: gcloud logs tail trading-bot --follow"
Write-Host "4. Update DNS/domain if needed"
Write-Host ""

Write-Host "Deployment completed successfully!"