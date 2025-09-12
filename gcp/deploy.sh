#!/bin/bash

# Trading Bot GCP Deployment Script
# This script sets up the entire GCP infrastructure for the trading bot

set -e  # Exit on any error

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"trading-bot-project"}
REGION=${GCP_REGION:-"europe-west1"}
ZONE=${GCP_ZONE:-"europe-west1-b"}
SERVICE_NAME="trading-bot"
DB_INSTANCE_NAME="trading-db-instance"
REDIS_INSTANCE_NAME="trading-redis-instance"
BUCKET_NAME="${PROJECT_ID}-trading-data"

echo "🚀 Starting Trading Bot deployment to GCP..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Zone: $ZONE"

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI is not installed. Please install it first."
    exit 1
fi

# Set the project
echo "📋 Setting GCP project..."
gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION
gcloud config set compute/zone $ZONE

# Enable required APIs
echo "🔧 Enabling required GCP APIs..."
gcloud services enable \
    cloudsql.googleapis.com \
    redis.googleapis.com \
    compute.googleapis.com \
    storage.googleapis.com \
    secretmanager.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com \
    run.googleapis.com \
    container.googleapis.com \
    artifactregistry.googleapis.com

# Create Cloud Storage bucket for data and models
echo "💾 Creating Cloud Storage bucket..."
if ! gsutil ls gs://$BUCKET_NAME &> /dev/null; then
    gsutil mb -l $REGION gs://$BUCKET_NAME
    echo "✅ Storage bucket created: gs://$BUCKET_NAME"
else
    echo "ℹ️ Storage bucket already exists: gs://$BUCKET_NAME"
fi

# Create Cloud SQL instance (PostgreSQL)
echo "🗄️ Creating Cloud SQL PostgreSQL instance..."
if ! gcloud sql instances describe $DB_INSTANCE_NAME &> /dev/null; then
    gcloud sql instances create $DB_INSTANCE_NAME \
        --database-version=POSTGRES_15 \
        --tier=db-g1-small \
        --region=$REGION \
        --storage-type=SSD \
        --storage-size=20GB \
        --storage-auto-increase \
        --backup-start-time=03:00 \
        --enable-bin-log \
        --maintenance-window-day=SUN \
        --maintenance-window-hour=04 \
        --deletion-protection
    
    echo "✅ Cloud SQL instance created: $DB_INSTANCE_NAME"
    
    # Set root password
    echo "🔐 Setting database root password..."
    gcloud sql users set-password postgres \
        --instance=$DB_INSTANCE_NAME \
        --password=$(openssl rand -base64 32)
    
    # Create trading database
    echo "📊 Creating trading database..."
    gcloud sql databases create trading_bot \
        --instance=$DB_INSTANCE_NAME
        
else
    echo "ℹ️ Cloud SQL instance already exists: $DB_INSTANCE_NAME"
fi

# Create Memorystore Redis instance
echo "🔴 Creating Memorystore Redis instance..."
if ! gcloud redis instances describe $REDIS_INSTANCE_NAME --region=$REGION &> /dev/null; then
    gcloud redis instances create $REDIS_INSTANCE_NAME \
        --region=$REGION \
        --memory-size=1GB \
        --redis-version=redis_7_0
    
    echo "✅ Memorystore Redis instance created: $REDIS_INSTANCE_NAME"
else
    echo "ℹ️ Memorystore Redis instance already exists: $REDIS_INSTANCE_NAME"
fi

# Create Artifact Registry repository
echo "📦 Creating Artifact Registry repository..."
if ! gcloud artifacts repositories describe trading-bot --location=$REGION &> /dev/null; then
    gcloud artifacts repositories create trading-bot \
        --repository-format=docker \
        --location=$REGION \
        --description="Trading Bot Docker images"
    
    echo "✅ Artifact Registry repository created"
else
    echo "ℹ️ Artifact Registry repository already exists"
fi

# Configure Docker authentication for Artifact Registry
echo "🔧 Configuring Docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build and push Docker image
echo "🐳 Building and pushing Docker image..."
IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/trading-bot/${SERVICE_NAME}:latest"

docker build -t $IMAGE_URL .
docker push $IMAGE_URL

echo "✅ Docker image pushed: $IMAGE_URL"

# Create secrets for API keys
echo "🔐 Creating secrets in Secret Manager..."

# Function to create secret if it doesn't exist
create_secret_if_not_exists() {
    local secret_name=$1
    local secret_value=$2
    
    if ! gcloud secrets describe $secret_name &> /dev/null; then
        echo -n "$secret_value" | gcloud secrets create $secret_name --data-file=-
        echo "✅ Created secret: $secret_name"
    else
        echo "ℹ️ Secret already exists: $secret_name"
    fi
}

# Create secrets (you'll need to set these values)
create_secret_if_not_exists "gemini-api-key" "${GEMINI_API_KEY:-placeholder}"
create_secret_if_not_exists "news-api-key" "${NEWS_API_KEY:-placeholder}"
create_secret_if_not_exists "alpha-vantage-key" "${ALPHA_VANTAGE_KEY:-placeholder}"
create_secret_if_not_exists "finnhub-key" "${FINNHUB_KEY:-placeholder}"

# Get connection details
echo "📋 Getting connection details..."
DB_CONNECTION_NAME=$(gcloud sql instances describe $DB_INSTANCE_NAME --format="value(connectionName)")
REDIS_HOST=$(gcloud redis instances describe $REDIS_INSTANCE_NAME --region=$REGION --format="value(host)")
REDIS_PORT=$(gcloud redis instances describe $REDIS_INSTANCE_NAME --region=$REGION --format="value(port)")

# Deploy to Cloud Run (for the main application without heavy ML training)
echo "☁️ Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image=$IMAGE_URL \
    --platform=managed \
    --region=$REGION \
    --allow-unauthenticated \
    --memory=2Gi \
    --cpu=2 \
    --timeout=3600 \
    --concurrency=1 \
    --max-instances=1 \
    --set-env-vars="DATABASE_URL=postgresql://postgres:@/$DB_CONNECTION_NAME/trading_bot" \
    --set-env-vars="REDIS_URL=redis://$REDIS_HOST:$REDIS_PORT" \
    --set-env-vars="TRADING_MODE=fast_mode" \
    --set-env-vars="ANALYSIS_LOOKBACK_DAYS=90" \
    --set-env-vars="ML_TRAINING_LOOKBACK_DAYS=3650" \
    --set-env-vars="NEWS_LOOKBACK_DAYS=60" \
    --set-env-vars="ENABLE_TRADITIONAL_ML=true" \
    --set-env-vars="ENABLE_TRANSFORMER_ML=false" \
    --set-secrets="GEMINI_API_KEY=gemini-api-key:latest" \
    --set-secrets="NEWS_API_KEY=news-api-key:latest" \
    --set-secrets="ALPHA_VANTAGE_KEY=alpha-vantage-key:latest" \
    --set-secrets="FINNHUB_KEY=finnhub-key:latest" \
    --add-cloudsql-instances=$DB_CONNECTION_NAME

echo "✅ Cloud Run service deployed successfully!"

# Output important information
echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📋 Important Information:"
echo "=========================="
echo "• Cloud Run Service URL: $(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')"
echo "• Cloud SQL Connection: $DB_CONNECTION_NAME"
echo "• Redis Host: $REDIS_HOST:$REDIS_PORT"
echo "• Storage Bucket: gs://$BUCKET_NAME"
echo "• Docker Image: $IMAGE_URL"
echo ""
echo "📝 Next Steps:"
echo "=============="
echo "1. Update API key secrets with actual values:"
echo "   gcloud secrets versions add gemini-api-key --data-file=<key-file>"
echo "2. Set up ML training on Compute Engine with GPU (see ml-training-setup.sh)"
echo "3. Configure monitoring and alerting"
echo "4. Set up CI/CD pipeline"
echo ""
echo "💡 For ML training with GPU, use the Compute Engine deployment script."