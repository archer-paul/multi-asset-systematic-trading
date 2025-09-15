'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Layout from '@/components/Layout/Layout'
import MetricCard from '@/components/MetricCard'
import PerformanceChart from '@/components/Charts/PerformanceChart'
import { api } from '@/lib/api'
import {
  ArrowTrendingUpIcon as TrendingUpIcon,
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

  useEffect(() => {
    // Simulate API call
    const loadData = async () => {
      setLoading(true)

      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 1500))

      // No data available - connect to backend API
      setPerformanceData([])

      // Default metrics when no data
      const totalReturn = 0
      const dailyReturn = 0

      setMetrics({
        totalReturn: 0,
        dailyReturn: 0,
        sharpeRatio: 0,
        maxDrawdown: 0,
        winRate: 0,
        activePositions: 0,
      })

      setLoading(false)
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
              icon={<TrendingUpIcon className="w-5 h-5" />}
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
              icon={<TrendingUpIcon className="w-5 h-5" />}
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
                {[
                  { component: 'Trading Engine', status: 'online', uptime: '99.9%' },
                  { component: 'ML Models', status: 'online', uptime: '98.7%' },
                  { component: 'Risk Manager', status: 'online', uptime: '100%' },
                  { component: 'Data Feed', status: 'online', uptime: '97.2%' },
                ].map((item, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.05 }}
                    className="flex items-center justify-between"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="w-3 h-3 bg-trading-profit rounded-full"></div>
                      <span className="text-white">{item.component}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs bg-dark-300 px-2 py-1 sharp-button text-dark-500">
                        {item.uptime} uptime
                      </span>
                      <span className="text-xs text-trading-profit font-medium">
                        {item.status.toUpperCase()}
                      </span>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </Layout>
  )
}