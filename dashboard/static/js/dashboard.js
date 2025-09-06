// Trading Bot Dashboard JavaScript
class TradingDashboard {
    constructor() {
        this.socket = io();
        this.lastUpdate = null;
        this.init();
    }

    init() {
        this.setupSocketEvents();
        this.loadInitialData();
        this.loadPerformanceChart();
        
        // Auto-refresh every 30 seconds
        setInterval(() => {
            this.loadInitialData();
        }, 30000);
    }

    setupSocketEvents() {
        this.socket.on('connect', () => {
            console.log('Connected to dashboard server');
            this.updateConnectionStatus(true);
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from dashboard server');
            this.updateConnectionStatus(false);
        });

        this.socket.on('dashboard_update', (data) => {
            console.log('Dashboard update received:', data);
            this.updateDashboard(data.data);
            this.updateLastUpdate(data.timestamp);
        });
    }

    updateConnectionStatus(connected) {
        const statusEl = document.getElementById('connection-status');
        if (connected) {
            statusEl.innerHTML = '<i class="fas fa-circle me-1"></i>Connected';
            statusEl.className = 'badge bg-success';
        } else {
            statusEl.innerHTML = '<i class="fas fa-circle me-1"></i>Disconnected';
            statusEl.className = 'badge bg-danger';
        }
    }

    updateLastUpdate(timestamp) {
        const updateTime = new Date(timestamp).toLocaleTimeString();
        document.getElementById('update-time').textContent = updateTime;
    }

    async loadInitialData() {
        try {
            const response = await fetch('/api/dashboard_data');
            const data = await response.json();
            this.updateDashboard(data);
        } catch (error) {
            console.error('Error loading dashboard data:', error);
        }
    }

    async loadPerformanceChart() {
        try {
            const response = await fetch('/api/performance_chart');
            const chartData = await response.json();
            
            if (!chartData.error) {
                Plotly.newPlot('performance-chart', chartData.data, chartData.layout, {
                    responsive: true,
                    displayModeBar: false
                });
            }
        } catch (error) {
            console.error('Error loading performance chart:', error);
        }
    }

    updateDashboard(data) {
        this.updatePerformanceMetrics(data.performance || {});
        this.updateMarketData(data.market_data || {});
        this.updateTradingSignals(data.signals || []);
        this.updateSentimentAnalysis(data.sentiment || []);
        this.updateRiskMetrics(data.risk_metrics || {});
        this.updateMultiTimeframeAnalysis();
    }

    updatePerformanceMetrics(performance) {
        const metrics = [
            { id: 'total-return', key: 'total_return', suffix: '%', precision: 2 },
            { id: 'daily-return', key: 'daily_return', suffix: '%', precision: 2 },
            { id: 'sharpe-ratio', key: 'sharpe_ratio', suffix: '', precision: 2 },
            { id: 'max-drawdown', key: 'max_drawdown', suffix: '%', precision: 2 },
            { id: 'win-rate', key: 'win_rate', suffix: '%', precision: 1 },
            { id: 'active-positions', key: 'current_positions', suffix: '', precision: 0 }
        ];

        metrics.forEach(metric => {
            const element = document.getElementById(metric.id);
            if (element && performance[metric.key] !== undefined) {
                const value = performance[metric.key];
                element.textContent = value.toFixed(metric.precision) + metric.suffix;
                
                // Color coding for returns
                if (metric.key.includes('return') && element.classList.contains('text-success')) {
                    element.className = value >= 0 ? 'text-success' : 'text-danger';
                }
            }
        });
    }

    updateMarketData(marketData) {
        const tbody = document.getElementById('market-data-table');
        tbody.innerHTML = '';

        Object.values(marketData).forEach(data => {
            const row = document.createElement('tr');
            const change = (Math.random() - 0.5) * 4; // Mock change
            const changeClass = change >= 0 ? 'text-success' : 'text-danger';
            
            row.innerHTML = `
                <td><strong>${data.symbol}</strong></td>
                <td>$${data.price?.toFixed(2) || '--'}</td>
                <td class="${changeClass}">${change >= 0 ? '+' : ''}${change.toFixed(2)}%</td>
                <td>${data.volume?.toLocaleString() || '--'}</td>
                <td><span class="badge bg-secondary">${data.source || '--'}</span></td>
                <td><small>${new Date(data.timestamp).toLocaleTimeString()}</small></td>
            `;
            tbody.appendChild(row);
        });
    }

    updateTradingSignals(signals) {
        const container = document.getElementById('signals-list');
        container.innerHTML = '';

        if (signals.length === 0) {
            container.innerHTML = '<p class="text-muted">No recent signals</p>';
            return;
        }

        signals.slice(-10).reverse().forEach(signal => {
            const div = document.createElement('div');
            div.className = `signal-item signal-${signal.signal_type?.toLowerCase() || 'hold'}`;
            
            const confidence = (signal.confidence * 100).toFixed(1);
            const time = new Date(signal.timestamp).toLocaleTimeString();
            
            div.innerHTML = `
                <div class="d-flex justify-content-between">
                    <strong>${signal.symbol}</strong>
                    <span class="badge bg-${this.getSignalBadgeColor(signal.signal_type)}">${signal.signal_type?.toUpperCase()}</span>
                </div>
                <div class="d-flex justify-content-between">
                    <small>$${signal.price?.toFixed(2)} - ${confidence}%</small>
                    <small class="text-muted">${time}</small>
                </div>
                ${signal.strategy ? `<small class="text-info">${signal.strategy}</small>` : ''}
            `;
            container.appendChild(div);
        });
    }

    updateSentimentAnalysis(sentimentData) {
        const container = document.getElementById('sentiment-analysis');
        container.innerHTML = '';

        if (!Array.isArray(sentimentData) || sentimentData.length === 0) {
            container.innerHTML = '<p class="text-muted">No sentiment data available</p>';
            return;
        }

        sentimentData.forEach(item => {
            const div = document.createElement('div');
            const sentimentClass = this.getSentimentClass(item.sentiment);
            div.className = `sentiment-item ${sentimentClass}`;
            
            div.innerHTML = `
                <div class="fw-bold mb-1" style="font-size: 0.9em;">${item.title}</div>
                <div class="d-flex justify-content-between">
                    <small>Sentiment: ${item.sentiment?.toFixed(2) || 'N/A'}</small>
                    <small class="text-muted">${new Date(item.published).toLocaleString()}</small>
                </div>
            `;
            container.appendChild(div);
        });
    }

    updateRiskMetrics(riskData) {
        const metrics = [
            { id: 'portfolio-var', key: 'portfolio_var', suffix: '%' },
            { id: 'portfolio-volatility', key: 'portfolio_volatility', suffix: '%' },
            { id: 'portfolio-beta', key: 'beta', suffix: '' },
            { id: 'current-exposure', key: 'current_exposure', suffix: '%' }
        ];

        metrics.forEach(metric => {
            const element = document.getElementById(metric.id);
            if (element && riskData[metric.key] !== undefined) {
                element.textContent = riskData[metric.key].toFixed(2) + metric.suffix;
            }
        });

        // Update exposure bar
        if (riskData.current_exposure && riskData.exposure_limit) {
            const exposureBar = document.getElementById('exposure-bar');
            const percentage = (riskData.current_exposure / riskData.exposure_limit) * 100;
            exposureBar.style.width = percentage + '%';
            
            // Change color based on exposure level
            exposureBar.className = `progress-bar ${percentage > 80 ? 'bg-danger' : percentage > 60 ? 'bg-warning' : 'bg-success'}`;
        }
    }

    getSignalBadgeColor(signalType) {
        const colors = {
            'buy': 'success',
            'sell': 'danger',
            'hold': 'warning'
        };
        return colors[signalType?.toLowerCase()] || 'secondary';
    }

    getSentimentClass(sentiment) {
        if (sentiment > 0.1) return 'sentiment-positive';
        if (sentiment < -0.1) return 'sentiment-negative';
        return 'sentiment-neutral';
    }

    async updateMultiTimeframeAnalysis() {
        try {
            // Fetch multi-timeframe summary
            const response = await fetch('/api/multi_timeframe_summary');
            const summary = await response.json();
            
            if (summary.error) {
                console.error('Multi-timeframe analysis error:', summary.error);
                return;
            }
            
            // Update summary metrics
            document.getElementById('mtf-bullish').textContent = summary.bullish_signals || 0;
            document.getElementById('mtf-bearish').textContent = summary.bearish_signals || 0;
            document.getElementById('mtf-neutral').textContent = summary.neutral_signals || 0;
            
            // Update top opportunities
            const container = document.getElementById('mtf-opportunities');
            container.innerHTML = '';
            
            if (!summary.top_opportunities || summary.top_opportunities.length === 0) {
                container.innerHTML = '<p class="text-muted small">No opportunities available</p>';
                return;
            }
            
            summary.top_opportunities.forEach(opp => {
                const div = document.createElement('div');
                div.className = 'mtf-opportunity-item mb-2 p-2 rounded';
                
                const scoreClass = this.getScoreClass(opp.score);
                const signalBadgeClass = this.getSignalBadgeColor(opp.signal.toLowerCase());
                const regimeLabel = this.getMarketRegimeLabel(opp.market_regime);
                
                div.innerHTML = `
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <strong class="text-white">${opp.symbol}</strong>
                            <span class="badge bg-${signalBadgeClass} ms-2">${opp.signal}</span>
                        </div>
                        <div class="text-end">
                            <div class="${scoreClass} fw-bold">${(opp.score * 100).toFixed(1)}%</div>
                            <small class="text-muted">Conf: ${(opp.confidence * 100).toFixed(0)}%</small>
                        </div>
                    </div>
                    <div class="d-flex justify-content-between mt-1">
                        <small class="text-info">${regimeLabel}</small>
                        <small class="text-muted">Score: ${opp.score.toFixed(3)}</small>
                    </div>
                `;
                
                // Add background color based on score
                if (opp.score > 0.3) {
                    div.style.backgroundColor = 'rgba(63, 185, 80, 0.1)'; // Green tint
                    div.style.borderLeft = '3px solid #3fb950';
                } else if (opp.score < -0.3) {
                    div.style.backgroundColor = 'rgba(248, 81, 73, 0.1)'; // Red tint
                    div.style.borderLeft = '3px solid #f85149';
                } else {
                    div.style.backgroundColor = 'rgba(110, 118, 129, 0.1)'; // Neutral
                    div.style.borderLeft = '3px solid #6e7681';
                }
                
                container.appendChild(div);
            });
            
        } catch (error) {
            console.error('Error updating multi-timeframe analysis:', error);
            
            // Show error state
            document.getElementById('mtf-bullish').textContent = '--';
            document.getElementById('mtf-bearish').textContent = '--';
            document.getElementById('mtf-neutral').textContent = '--';
            document.getElementById('mtf-opportunities').innerHTML = 
                '<p class="text-danger small">Error loading analysis</p>';
        }
    }
    
    getScoreClass(score) {
        if (score > 0.3) return 'text-success';
        if (score < -0.3) return 'text-danger';
        return 'text-warning';
    }
    
    getMarketRegimeLabel(regime) {
        const labels = {
            'uptrending': 'Uptrend',
            'downtrending': 'Downtrend', 
            'ranging': 'Range',
            'volatile': 'Volatile',
            'unknown': 'Unknown'
        };
        return labels[regime] || 'Unknown';
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    new TradingDashboard();
});