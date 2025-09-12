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
RUN pip install --no-cache-dir --user -r requirements.txt

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

# Copy Python packages from builder
COPY --from=builder /root/.local /home/tradingbot/.local
ENV PATH=/home/tradingbot/.local/bin:$PATH

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs cache exports dashboard_data data_sources && \
    chown -R tradingbot:tradingbot /app

# Switch to non-root user
USER tradingbot

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port for potential web interface
EXPOSE 8000

# Default command
CMD ["python", "enhanced_main.py"]