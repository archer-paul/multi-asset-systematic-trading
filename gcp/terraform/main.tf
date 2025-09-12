# Trading Bot GCP Infrastructure
# Terraform configuration for automated deployment

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "europe-west1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "europe-west1-b"
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
  default     = "dev"
}

# Provider
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Enable APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "cloudsql.googleapis.com",
    "redis.googleapis.com",
    "compute.googleapis.com",
    "storage.googleapis.com",
    "secretmanager.googleapis.com",
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com"
  ])
  
  project = var.project_id
  service = each.value
  
  disable_on_destroy = false
}

# Cloud Storage bucket for data and models
resource "google_storage_bucket" "trading_data" {
  name          = "${var.project_id}-trading-data"
  location      = var.region
  force_destroy = true
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
}

# Cloud SQL PostgreSQL instance
resource "google_sql_database_instance" "trading_db" {
  name             = "trading-db-${var.environment}"
  database_version = "POSTGRES_15"
  region          = var.region
  
  depends_on = [google_project_service.apis]
  
  settings {
    tier = "db-g1-small"
    
    disk_type       = "PD_SSD"
    disk_size       = 20
    disk_autoresize = true
    
    backup_configuration {
      enabled    = true
      start_time = "03:00"
    }
    
    maintenance_window {
      day  = 7  # Sunday
      hour = 4
    }
    
    ip_configuration {
      ipv4_enabled    = true
      authorized_networks {
        name  = "all"
        value = "0.0.0.0/0"
      }
    }
  }
  
  deletion_protection = false  # Set to true for production
}

# Database
resource "google_sql_database" "trading_database" {
  name     = "trading_bot"
  instance = google_sql_database_instance.trading_db.name
}

# Database user
resource "google_sql_user" "trading_user" {
  name     = "trading_user"
  instance = google_sql_database_instance.trading_db.name
  password = random_password.db_password.result
}

# Random password for database
resource "random_password" "db_password" {
  length  = 32
  special = true
}

# Memorystore Redis instance
resource "google_redis_instance" "trading_cache" {
  name           = "trading-redis-${var.environment}"
  memory_size_gb = 1
  region         = var.region
  
  depends_on = [google_project_service.apis]
}

# Artifact Registry repository
resource "google_artifact_registry_repository" "trading_repo" {
  location      = var.region
  repository_id = "trading-bot"
  description   = "Trading Bot Docker images"
  format        = "DOCKER"
  
  depends_on = [google_project_service.apis]
}

# Secrets for API keys
resource "google_secret_manager_secret" "api_secrets" {
  for_each = toset([
    "gemini-api-key",
    "news-api-key",
    "alpha-vantage-key",
    "finnhub-key"
  ])
  
  secret_id = each.value
  
  replication {
    automatic = true
  }
  
  depends_on = [google_project_service.apis]
}

# Placeholder secret versions (update with actual values)
resource "google_secret_manager_secret_version" "api_secret_versions" {
  for_each = google_secret_manager_secret.api_secrets
  
  secret      = each.value.id
  secret_data = "placeholder-update-with-real-key"
  
  lifecycle {
    ignore_changes = [secret_data]
  }
}

# Service account for Cloud Run
resource "google_service_account" "trading_bot_sa" {
  account_id   = "trading-bot-${var.environment}"
  display_name = "Trading Bot Service Account"
  description  = "Service account for Trading Bot application"
}

# IAM bindings for service account
resource "google_project_iam_member" "trading_bot_permissions" {
  for_each = toset([
    "roles/cloudsql.client",
    "roles/secretmanager.secretAccessor",
    "roles/storage.objectAdmin",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.trading_bot_sa.email}"
}

# Cloud Run service
resource "google_cloud_run_service" "trading_bot" {
  name     = "trading-bot-${var.environment}"
  location = var.region
  
  depends_on = [google_project_service.apis]
  
  template {
    spec {
      service_account_name = google_service_account.trading_bot_sa.email
      
      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/trading-bot/trading-bot:latest"
        
        resources {
          limits = {
            memory = "2Gi"
            cpu    = "2000m"
          }
        }
        
        env {
          name  = "DATABASE_URL"
          value = "postgresql://${google_sql_user.trading_user.name}:${google_sql_user.trading_user.password}@${google_sql_database_instance.trading_db.connection_name}/${google_sql_database.trading_database.name}"
        }
        
        env {
          name  = "REDIS_URL"
          value = "redis://${google_redis_instance.trading_cache.host}:${google_redis_instance.trading_cache.port}"
        }
        
        env {
          name  = "TRADING_MODE"
          value = "fast_mode"
        }
        
        env {
          name  = "ANALYSIS_LOOKBACK_DAYS"
          value = "90"
        }
        
        env {
          name  = "ML_TRAINING_LOOKBACK_DAYS"
          value = "3650"
        }
        
        env {
          name  = "NEWS_LOOKBACK_DAYS"
          value = "60"
        }
        
        env {
          name = "GEMINI_API_KEY"
          value_from {
            secret_key_ref {
              name = google_secret_manager_secret.api_secrets["gemini-api-key"].secret_id
              key  = "latest"
            }
          }
        }
        
        # Add other API key environment variables similarly
      }
    }
    
    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale"        = "1"
        "run.googleapis.com/cloudsql-instances"   = google_sql_database_instance.trading_db.connection_name
        "run.googleapis.com/execution-environment" = "gen2"
      }
    }
  }
  
  traffic {
    percent         = 100
    latest_revision = true
  }
}

# Allow unauthenticated access to Cloud Run service (adjust as needed)
resource "google_cloud_run_service_iam_member" "public_access" {
  service  = google_cloud_run_service.trading_bot.name
  location = google_cloud_run_service.trading_bot.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Outputs
output "cloud_run_url" {
  description = "URL of the deployed Cloud Run service"
  value       = google_cloud_run_service.trading_bot.status[0].url
}

output "database_connection_name" {
  description = "Cloud SQL connection name"
  value       = google_sql_database_instance.trading_db.connection_name
}

output "redis_host" {
  description = "Redis instance host"
  value       = google_redis_instance.trading_cache.host
}

output "storage_bucket" {
  description = "Cloud Storage bucket name"
  value       = google_storage_bucket.trading_data.name
}