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

Write-Host "[INFO] Starting FULL Trading Bot deployment (Backend + Frontend)..." -ForegroundColor Green
Write-Host "Backend: GCP Cloud Run" -ForegroundColor Cyan
Write-Host "Frontend: Firebase Hosting" -ForegroundColor Cyan

# 1. Vérifications des outils requis
Write-Host "`n[INFO] Checking required tools..." -ForegroundColor Yellow

# Check gcloud
try {
    $gcloudVersion = gcloud version 2>$null
    if (-not $gcloudVersion) { throw "gcloud not found" }
    Write-Host "[OK] gcloud CLI found" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] gcloud CLI not installed. Install from: https://cloud.google.com/sdk/docs/install-windows" -ForegroundColor Red
    exit 1
}

# Check Firebase CLI
try {
    $firebaseVersion = firebase --version 2>$null
    if (-not $firebaseVersion) { throw "firebase not found" }
    Write-Host "[OK] Firebase CLI found ($firebaseVersion)" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Firebase CLI not installed. Run: npm install -g firebase-tools" -ForegroundColor Red
    exit 1
}

# Docker not needed - using Cloud Build for backend

# Check Node.js
try {
    $nodeVersion = node --version 2>$null
    if (-not $nodeVersion) { throw "node not found" }
    Write-Host "[OK] Node.js found ($nodeVersion)" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Node.js not installed. Please install Node.js." -ForegroundColor Red
    exit 1
}

# 2. Build et test du frontend AVANT de déployer le backend
Write-Host "`n[STEP 1/3] Building and testing frontend..." -ForegroundColor Cyan

# Aller dans le répertoire frontend
$frontendPath = "frontend"
if (-not (Test-Path $frontendPath)) {
    Write-Host "[ERROR] Frontend directory not found: $frontendPath" -ForegroundColor Red
    exit 1
}

Push-Location $frontendPath

try {
    # Install dependencies
    Write-Host "[INFO] Installing frontend dependencies..." -ForegroundColor Yellow
    npm ci
    if ($LASTEXITCODE -ne 0) {
        throw "npm ci failed"
    }

    # Build frontend
    Write-Host "[INFO] Building frontend..." -ForegroundColor Yellow
    npm run build
    if ($LASTEXITCODE -ne 0) {
        throw "Frontend build failed"
    }

    # Export frontend for Firebase
    Write-Host "[INFO] Exporting frontend for static hosting..." -ForegroundColor Yellow
    npm run export
    if ($LASTEXITCODE -ne 0) {
        throw "Frontend export failed"
    }

    Write-Host "[SUCCESS] Frontend built and exported successfully!" -ForegroundColor Green

} catch {
    Write-Host "[ERROR] Frontend build failed: $($_.Exception.Message)" -ForegroundColor Red
    Pop-Location
    exit 1
} finally {
    Pop-Location
}

# 3. Déployer le backend sur GCP Cloud Run
Write-Host "`n[STEP 2/3] Deploying backend to GCP Cloud Run..." -ForegroundColor Cyan

# Exécuter le script de déploiement backend existant
$backendDeployScript = "gcp\deploy.ps1"
if (-not (Test-Path $backendDeployScript)) {
    Write-Host "[ERROR] Backend deploy script not found: $backendDeployScript" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Running backend deployment..." -ForegroundColor Yellow
& $backendDeployScript -ProjectId $ProjectId -Region $Region -Zone $Zone

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Backend deployment failed!" -ForegroundColor Red
    exit 1
}

# Récupérer l'URL du backend déployé
$backendUrl = gcloud run services describe trading-bot --region=$Region --format='value(status.url)' 2>$null
if (-not $backendUrl) {
    Write-Host "[WARNING] Could not retrieve backend URL. Frontend may need manual configuration." -ForegroundColor Yellow
    $backendUrl = "https://trading-bot-xxx-$Region.run.app"
}

Write-Host "[SUCCESS] Backend deployed successfully: $backendUrl" -ForegroundColor Green

# 4. Déployer le frontend sur Firebase
Write-Host "`n[STEP 3/3] Deploying frontend to Firebase..." -ForegroundColor Cyan

# Mettre à jour la configuration frontend avec l'URL du backend
Write-Host "[INFO] Updating frontend configuration with backend URL..." -ForegroundColor Yellow

$envLocalPath = "frontend\.env.local"
$envContent = @"
NEXT_PUBLIC_API_URL=$backendUrl
NEXT_PUBLIC_WS_URL=$($backendUrl -replace 'https://', 'wss://')
"@

$envContent | Out-File -FilePath $envLocalPath -Encoding utf8
Write-Host "[INFO] Updated $envLocalPath with backend URL" -ForegroundColor Green

# Firebase login check
Write-Host "[INFO] Checking Firebase authentication..." -ForegroundColor Yellow
$firebaseProjects = firebase projects:list 2>$null
if (-not $firebaseProjects) {
    Write-Host "[INFO] Please login to Firebase..." -ForegroundColor Yellow
    firebase login
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Firebase login failed!" -ForegroundColor Red
        exit 1
    }
}

# Deploy to Firebase
Write-Host "[INFO] Deploying to Firebase Hosting..." -ForegroundColor Yellow
firebase use quantitative-alpha-engine
firebase deploy --only hosting

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Firebase deployment failed!" -ForegroundColor Red
    exit 1
}

# Récupérer l'URL Firebase
$firebaseUrl = "https://quantitative-alpha-engine.web.app"

# 5. Résumé final
Write-Host "`n[SUCCESS] FULL DEPLOYMENT COMPLETED!" -ForegroundColor Green
Write-Host ""
Write-Host "╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                  DEPLOYMENT SUMMARY                       ║" -ForegroundColor Cyan
Write-Host "╠═══════════════════════════════════════════════════════════╣" -ForegroundColor Cyan
Write-Host "║ Backend (GCP Cloud Run): $backendUrl" -ForegroundColor White
Write-Host "║ Frontend (Firebase):     $firebaseUrl" -ForegroundColor White
Write-Host "╚═══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host "1. Open your trading dashboard: $firebaseUrl" -ForegroundColor White
Write-Host "2. Check backend health: $backendUrl/health" -ForegroundColor White
Write-Host "3. Monitor logs: gcloud logs tail trading-bot --follow" -ForegroundColor Gray
Write-Host "4. Update DNS/domain if needed" -ForegroundColor White
Write-Host ""

Write-Host "Deployment completed successfully!"