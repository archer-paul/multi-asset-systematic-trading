# Multi-stage Docker build for Trading Bot
# Optimized for GCP deployment with GPU support

# Build stage
FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source
RUN cd /tmp && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr/local && \
    make && make install && \
    cd .. && rm -rf ta-lib*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy TA-Lib from builder
COPY --from=builder /usr/local /usr/local
RUN ldconfig

# Create non-root user for security
RUN groupadd -r tradingbot && useradd -r -g tradingbot -m -d /app -s /bin/bash tradingbot

# Set working directory
WORKDIR /app



# Copy application code (as root to avoid permission issues)
COPY . .

# Create necessary directories and fix all permissions
RUN mkdir -p logs cache exports dashboard_data data/cache data/reports data/models data/backups && \
    chown -R tradingbot:tradingbot /app

# Switch to tradingbot user for running the application
USER tradingbot

# Environment variables
ENV PYTHONPATH=/home/tradingbot/.local/lib/python3.11/site-packages:/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port for the dashboard server
EXPOSE 8080

# Command to run the application using Gunicorn
CMD ["gunicorn", "--workers", "1", "--threads", "10", "--worker-class", "eventlet", "-b", ":8080", "--timeout", "300", "--graceful-timeout", "300", "wsgi:app"]