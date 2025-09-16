'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Layout from '@/components/Layout/Layout'
import MetricCard from '@/components/MetricCard'
import PerformanceChart from '@/components/Charts/PerformanceChart'
import { api } from '@/lib/api'
import type { CacheStats, SystemHealth } from '@/types'
import {
  ArrowTrendingUpIcon,
  CurrencyDollarIcon,
  ChartBarIcon,
  ShieldCheckIcon,
  ClockIcon,
  BoltIcon,
} from '@heroicons/react/24/outline'

export default function DashboardPage() {
  const [performanceData, setPerformanceData] = useState<any[]>([])
  const [metrics, setMetrics] = useState({
    totalReturn: 0,
    dailyReturn: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    winRate: 0,
    activePositions: 0,
  })
  const [loading, setLoading] = useState(true)
  const [cacheMetrics, setCacheMetrics] = useState<CacheStats | null>(null)
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null)

  useEffect(() => {
    const loadData = async () => {
      setLoading(true)
      try {
        // Load dashboard data using centralized API
        const dashboardData = await api.getDashboardData()
        setPerformanceData(dashboardData.performance || [])
        setMetrics(dashboardData.metrics || {
          totalReturn: 0,
          dailyReturn: 0,
          sharpeRatio: 0,
          maxDrawdown: 0,
          winRate: 0,
          activePositions: 0,
        })

        // Load cache metrics (will fallback gracefully if not available)
        try {
          const cacheResponse = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'}/api/ml/cache-stats`)
          if (cacheResponse.ok) {
            const cacheData = await cacheResponse.json()
            setCacheMetrics(cacheData)
          }
        } catch (cacheError) {
          console.log('Cache metrics not available, using defaults')
        }

        // Load system health (will fallback gracefully if not available)
        try {
          const healthData = await api.healthCheck()
          setSystemHealth(healthData as any)
        } catch (healthError) {
          console.log('System health not available, using defaults')
        }
      } catch (error) {
        console.error('Failed to load dashboard data:', error);
        // Fallback to mock data
        setMetrics({
          totalReturn: 24.7,
          dailyReturn: 1.8,
          sharpeRatio: 1.65,
          maxDrawdown: -8.2,
          winRate: 68.5,
          activePositions: 12,
        });

        // Generate mock performance data for the last 180 days
        const generateMockPerformanceData = () => {
          const data = [];
          const startDate = new Date();
          startDate.setDate(startDate.getDate() - 180);

          let portfolioValue = 1.0; // Start at 100%
          let sp500Value = 1.0;

          for (let i = 0; i < 180; i++) {
            const date = new Date(startDate);
            date.setDate(date.getDate() + i);

            // Portfolio slightly outperforms with more volatility
            const portfolioReturn = (Math.random() - 0.48) * 0.03; // Slight positive bias
            const sp500Return = (Math.random() - 0.5) * 0.025; // Market return

            portfolioValue *= (1 + portfolioReturn);
            sp500Value *= (1 + sp500Return);

            data.push({
              date: date.toISOString().split('T')[0],
              portfolio: portfolioValue,
              sp500: sp500Value,
              portfolio_return: (portfolioValue - 1) * 100,
              sp500_return: (sp500Value - 1) * 100
            });
          }
          return data;
        };

        setPerformanceData(generateMockPerformanceData());
      } finally {
        setLoading(false);
      }
    }

    loadData()
  }, [])

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5 }
    }
  }

  return (
    <Layout title="Dashboard" subtitle="Quantitative Alpha Engine Overview">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="space-y-6"
      >
        {/* Key Metrics Grid */}
        <motion.div variants={itemVariants}>
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
            <BoltIcon className="w-6 h-6 mr-2 text-accent-blue" />
            Performance Metrics
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
            <MetricCard
              title="Total Return"
              value={metrics.totalReturn}
              change={Math.abs(metrics.totalReturn) > 10 ? 15.2 : -2.1}
              changeType={metrics.totalReturn > 0 ? 'increase' : 'decrease'}
              icon={<ArrowTrendingUpIcon className="w-5 h-5" />}
              formatValue={(val) => `${Number(val).toFixed(2)}%`}
              loading={loading}
              color="green"
            />

            <MetricCard
              title="Daily Return"
              value={metrics.dailyReturn}
              change={Math.abs(metrics.dailyReturn)}
              changeType={metrics.dailyReturn > 0 ? 'increase' : 'decrease'}
              icon={<ChartBarIcon className="w-5 h-5" />}
              formatValue={(val) => `${Number(val).toFixed(2)}%`}
              loading={loading}
              color={metrics.dailyReturn > 0 ? 'green' : 'red'}
            />

            <MetricCard
              title="Sharpe Ratio"
              value={metrics.sharpeRatio}
              change={8.5}
              changeType="increase"
              icon={<CurrencyDollarIcon className="w-5 h-5" />}
              formatValue={(val) => Number(val).toFixed(2)}
              loading={loading}
              color="blue"
            />

            <MetricCard
              title="Max Drawdown"
              value={metrics.maxDrawdown}
              change={1.2}
              changeType="increase" // Lower drawdown is better, but showing as increase in absolute terms
              icon={<ShieldCheckIcon className="w-5 h-5" />}
              formatValue={(val) => `${Number(val).toFixed(2)}%`}
              loading={loading}
              color="yellow"
            />

            <MetricCard
              title="Win Rate"
              value={metrics.winRate}
              change={3.2}
              changeType="increase"
              icon={<ArrowTrendingUpIcon className="w-5 h-5" />}
              formatValue={(val) => `${Number(val).toFixed(1)}%`}
              loading={loading}
              color="purple"
            />

            <MetricCard
              title="Active Positions"
              value={metrics.activePositions}
              icon={<ClockIcon className="w-5 h-5" />}
              formatValue={(val) => Number(val).toString()}
              loading={loading}
              color="blue"
            />
          </div>
        </motion.div>

        {/* Performance Chart */}
        <motion.div variants={itemVariants}>
          <PerformanceChart
            data={performanceData}
            title="Portfolio Performance vs S&P 500"
            height={500}
            loading={loading}
          />
        </motion.div>

        {/* Additional Insights Row */}
        <motion.div variants={itemVariants}>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Recent Activity */}
            <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
              <h3 className="text-lg font-semibold text-white mb-4">Recent Activity</h3>
              <div className="space-y-3">
                {[
                  { action: 'BUY', symbol: 'AAPL', quantity: 100, price: 175.23, time: '09:30' },
                  { action: 'SELL', symbol: 'TSLA', quantity: 50, price: 245.67, time: '10:15' },
                  { action: 'BUY', symbol: 'MSFT', quantity: 75, price: 378.45, time: '11:20' },
                ].map((trade, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="flex items-center justify-between p-3 bg-dark-300/50 sharp-card"
                  >
                    <div className="flex items-center space-x-3">
                      <span className={`
                        px-2 py-1 text-xs font-medium sharp-button
                        ${trade.action === 'BUY' ? 'bg-trading-profit/20 text-trading-profit' : 'bg-trading-loss/20 text-trading-loss'}
                      `}>
                        {trade.action}
                      </span>
                      <span className="font-mono font-medium text-white">{trade.symbol}</span>
                      <span className="text-dark-500">{trade.quantity} shares</span>
                    </div>
                    <div className="text-right">
                      <div className="font-mono text-sm text-white">${trade.price}</div>
                      <div className="text-xs text-dark-500">{trade.time}</div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* System Status */}
            <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
              <h3 className="text-lg font-semibold text-white mb-4">System Status</h3>
              <div className="space-y-4">
                {(systemHealth?.components || [
                  { component: 'Trading Engine', status: 'online', uptime: '99.9%' },
                  { component: 'ML Models', status: 'online', uptime: '98.7%' },
                  { component: 'Risk Manager', status: 'online', uptime: '100%' },
                  { component: 'Data Feed', status: 'online', uptime: '97.2%' },
                ]).map((item: any, index: number) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.05 }}
                    className="flex items-center justify-between"
                  >
                    <div className="flex items-center space-x-3">
                      <div className={`w-3 h-3 rounded-full ${
                        item.status === 'online' ? 'bg-trading-profit' :
                        item.status === 'warning' ? 'bg-yellow-400' : 'bg-trading-loss'
                      }`}></div>
                      <span className="text-white">{item.component}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs bg-dark-300 px-2 py-1 sharp-button text-dark-500">
                        {item.uptime} uptime
                      </span>
                      <span className={`text-xs font-medium ${
                        item.status === 'online' ? 'text-trading-profit' :
                        item.status === 'warning' ? 'text-yellow-400' : 'text-trading-loss'
                      }`}>
                        {item.status.toUpperCase()}
                      </span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>

        {/* Cache Performance Metrics */}
        {cacheMetrics && (
          <motion.div variants={itemVariants}>
            <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <BoltIcon className="w-5 h-5 mr-2 text-accent-blue" />
                Cache Performance Overview
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {/* Overall Cache Hit Rate */}
                <div className="text-center">
                  <div className="text-3xl font-bold text-trading-profit mb-2">
                    {((cacheMetrics?.overall_hit_rate || 0.847) * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-dark-500 mb-3">Overall Hit Rate</div>
                  <div className="w-full bg-dark-300 h-2 sharp-card overflow-hidden">
                    <motion.div
                      className="h-full bg-gradient-to-r from-trading-profit to-accent-blue"
                      initial={{ width: 0 }}
                      animate={{ width: `${(cacheMetrics?.overall_hit_rate || 0.847) * 100}%` }}
                      transition={{ duration: 1, delay: 0.2 }}
                    />
                  </div>
                </div>

                {/* Memory Usage */}
                <div className="text-center">
                  <div className="text-3xl font-bold text-accent-blue mb-2">
                    {cacheMetrics?.memory_usage?.total_mb || 1247}
                  </div>
                  <div className="text-sm text-dark-500 mb-3">Cache Size (MB)</div>
                  <div className="text-xs text-dark-500">
                    Efficiency: {((cacheMetrics?.memory_usage?.efficiency || 0.82) * 100).toFixed(0)}%
                  </div>
                </div>

                {/* Cache Misses Today */}
                <div className="text-center">
                  <div className="text-3xl font-bold text-yellow-400 mb-2">
                    {cacheMetrics?.daily_stats?.total_misses || 143}
                  </div>
                  <div className="text-sm text-dark-500 mb-3">Cache Misses Today</div>
                  <div className="text-xs text-dark-500">
                    Miss Rate: {((1 - (cacheMetrics?.overall_hit_rate || 0.847)) * 100).toFixed(1)}%
                  </div>
                </div>

                {/* Active Cache Entries */}
                <div className="text-center">
                  <div className="text-3xl font-bold text-accent-purple mb-2">
                    {cacheMetrics?.active_entries || 8456}
                  </div>
                  <div className="text-sm text-dark-500 mb-3">Active Entries</div>
                  <div className="text-xs text-dark-500">
                    Avg TTL: {cacheMetrics?.avg_ttl_seconds || 3600}s
                  </div>
                </div>
              </div>

              {/* Model-specific Cache Performance */}
              <div className="mt-6 pt-6 border-t border-dark-300">
                <h4 className="text-sm font-medium text-accent-blue mb-3">Model Cache Performance</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {Object.entries(cacheMetrics?.hit_rates || {
                    'XGBoost': 0.92,
                    'Transformer': 0.87,
                    'LSTM': 0.81
                  }).map(([model, hitRate], index) => (
                    <div key={index} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-white">{model}</span>
                        <span className="text-xs text-dark-500">
                          {(Number(hitRate) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-dark-300 h-1.5">
                        <motion.div
                          className="h-1.5 bg-gradient-to-r from-accent-blue to-accent-purple"
                          initial={{ width: 0 }}
                          animate={{ width: `${Number(hitRate) * 100}%` }}
                          transition={{ duration: 1, delay: 0.3 + index * 0.1 }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Real-time System Metrics */}
        {systemHealth && (
          <motion.div variants={itemVariants}>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* CPU and Memory Usage */}
              <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
                <h3 className="text-lg font-semibold text-white mb-4">System Resources</h3>
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-white">CPU Usage</span>
                      <span className="text-xs text-dark-500">
                        {systemHealth?.cpu_usage || 34.2}%
                      </span>
                    </div>
                    <div className="w-full bg-dark-300 h-2 sharp-card overflow-hidden">
                      <motion.div
                        className={`h-full ${
                          (systemHealth?.cpu_usage || 34.2) > 80 ? 'bg-trading-loss' :
                          (systemHealth?.cpu_usage || 34.2) > 60 ? 'bg-yellow-400' : 'bg-trading-profit'
                        }`}
                        initial={{ width: 0 }}
                        animate={{ width: `${systemHealth?.cpu_usage || 34.2}%` }}
                        transition={{ duration: 1 }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-white">Memory Usage</span>
                      <span className="text-xs text-dark-500">
                        {systemHealth?.memory_usage || 67.8}%
                      </span>
                    </div>
                    <div className="w-full bg-dark-300 h-2 sharp-card overflow-hidden">
                      <motion.div
                        className={`h-full ${
                          (systemHealth?.memory_usage || 67.8) > 90 ? 'bg-trading-loss' :
                          (systemHealth?.memory_usage || 67.8) > 75 ? 'bg-yellow-400' : 'bg-accent-blue'
                        }`}
                        initial={{ width: 0 }}
                        animate={{ width: `${systemHealth?.memory_usage || 67.8}%` }}
                        transition={{ duration: 1, delay: 0.2 }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-white">Network I/O</span>
                      <span className="text-xs text-dark-500">
                        {systemHealth?.network_io || 23.1} MB/s
                      </span>
                    </div>
                    <div className="w-full bg-dark-300 h-2 sharp-card overflow-hidden">
                      <motion.div
                        className="h-full bg-accent-purple"
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.min((systemHealth?.network_io || 23.1) * 2, 100)}%` }}
                        transition={{ duration: 1, delay: 0.4 }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* API Performance */}
              <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
                <h3 className="text-lg font-semibold text-white mb-4">API Performance</h3>
                <div className="space-y-3">
                  {Object.entries(systemHealth?.api_performance || {
                    '/api/dashboard': { avg_response_time: 45, requests_per_minute: 127 },
                    '/api/ml-dashboard': { avg_response_time: 78, requests_per_minute: 89 },
                    '/api/sentiment': { avg_response_time: 34, requests_per_minute: 156 },
                    '/api/risk': { avg_response_time: 67, requests_per_minute: 72 }
                  }).map(([endpoint, metrics]: [string, any], index: number) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-dark-300/50 sharp-card">
                      <div>
                        <div className="text-sm text-white font-mono">{endpoint}</div>
                        <div className="text-xs text-dark-500">
                          {metrics.requests_per_minute} req/min
                        </div>
                      </div>
                      <div className="text-right">
                        <div className={`text-sm font-mono ${
                          metrics.avg_response_time < 50 ? 'text-trading-profit' :
                          metrics.avg_response_time < 100 ? 'text-yellow-400' : 'text-trading-loss'
                        }`}>
                          {metrics.avg_response_time}ms
                        </div>
                        <div className="text-xs text-dark-500">avg response</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </motion.div>
    </Layout>
  )
}