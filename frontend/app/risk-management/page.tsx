'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Layout from '@/components/Layout/Layout'
import MetricCard from '@/components/MetricCard'
import {
  ShieldCheckIcon,
  ExclamationTriangleIcon,
  ChartPieIcon,
  ArrowTrendingDownIcon as TrendingDownIcon,
  BellAlertIcon,
  CurrencyDollarIcon,
  ScaleIcon,
  ClockIcon,
} from '@heroicons/react/24/outline'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  RadialBarChart,
  RadialBar,
  PieChart,
  Pie,
  Cell,
} from 'recharts'

// Mock risk data
const portfolioRisk = [
  { date: '2024-01-01', var95: -2.1, var99: -3.8, expectedShortfall: -4.2, realized: -1.8 },
  { date: '2024-01-02', var95: -2.3, var99: -4.1, expectedShortfall: -4.6, realized: -2.1 },
  { date: '2024-01-03', var95: -1.9, var99: -3.5, expectedShortfall: -3.9, realized: -1.5 },
  { date: '2024-01-04', var95: -2.5, var99: -4.3, expectedShortfall: -4.8, realized: -3.2 },
  { date: '2024-01-05', var95: -2.0, var99: -3.7, expectedShortfall: -4.1, realized: -1.9 },
  { date: '2024-01-06', var95: -2.2, var99: -3.9, expectedShortfall: -4.3, realized: -2.0 },
  { date: '2024-01-07', var95: -2.1, var99: -3.8, expectedShortfall: -4.2, realized: -1.7 },
]

const sectorExposure = [
  { sector: 'Technology', exposure: 0.35, risk: 0.18, color: '#3b82f6' },
  { sector: 'Healthcare', exposure: 0.22, risk: 0.14, color: '#10b981' },
  { sector: 'Financial', exposure: 0.18, risk: 0.22, color: '#f59e0b' },
  { sector: 'Consumer', exposure: 0.15, risk: 0.16, color: '#ef4444' },
  { sector: 'Energy', exposure: 0.10, risk: 0.28, color: '#8b5cf6' },
]

const riskAlerts = [
  {
    id: 1,
    type: 'high',
    title: 'High Correlation Warning',
    description: 'AAPL and MSFT correlation exceeded 0.85',
    time: '2 minutes ago',
    action: 'Consider rebalancing'
  },
  {
    id: 2,
    type: 'medium',
    title: 'Sector Concentration',
    description: 'Technology sector exposure at 35%',
    time: '15 minutes ago',
    action: 'Monitor exposure'
  },
  {
    id: 3,
    type: 'low',
    title: 'VaR Breach',
    description: 'Daily VaR exceeded by 0.3%',
    time: '1 hour ago',
    action: 'Review positions'
  },
]

const stressTestScenarios = [
  { scenario: 'Market Crash (-20%)', portfolioImpact: -12.5, probability: 0.05 },
  { scenario: 'Tech Selloff (-30%)', portfolioImpact: -8.7, probability: 0.12 },
  { scenario: 'Interest Rate Shock', portfolioImpact: -5.2, probability: 0.25 },
  { scenario: 'Inflation Spike', portfolioImpact: -3.8, probability: 0.35 },
  { scenario: 'Geopolitical Crisis', portfolioImpact: -15.2, probability: 0.08 },
]

export default function RiskManagementPage() {
  const [loading, setLoading] = useState(true)
  const [selectedTimeframe, setSelectedTimeframe] = useState('1D')
  const [stressTestResults, setStressTestResults] = useState(null)
  const [monteCarloResults, setMonteCarloResults] = useState(null)

  useEffect(() => {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
    const loadData = async () => {
      setLoading(true)
      try {
        // Load stress test results
        const stressResponse = await fetch(`${API_URL}/api/risk/stress-tests`);
        if (stressResponse.ok) {
          const stressData = await stressResponse.json();
          setStressTestResults(stressData);
        }

        // Load Monte Carlo simulation results
        const monteCarloResponse = await fetch(`${API_URL}/api/risk/monte-carlo`);
        if (monteCarloResponse.ok) {
          const monteCarloData = await monteCarloResponse.json();
          setMonteCarloResults(monteCarloData);
        }
      } catch (error) {
        console.error('Failed to load risk management data:', error);
      } finally {
        setLoading(false);
      }
    }

    loadData()
  }, [])

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-dark-200 border border-dark-300 sharp-card p-4 shadow-lg">
          <p className="text-white font-medium mb-2">{label}</p>
          {payload.map((entry: any, index: number) => (
            <div key={index} className="flex items-center space-x-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-sm text-dark-500">{entry.name}:</span>
              <span className="text-sm text-white font-mono">
                {entry.value.toFixed(2)}%
              </span>
            </div>
          ))}
        </div>
      )
    }
    return null
  }

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
    <Layout title="Risk Management" subtitle="Portfolio Risk Analysis & Control">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="space-y-6"
      >
        {/* Risk Metrics Overview */}
        <motion.div variants={itemVariants}>
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
            <ShieldCheckIcon className="w-6 h-6 mr-2 text-accent-blue" />
            Risk Metrics Overview
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              title="Portfolio VaR (95%)"
              value={-2.1}
              change={0.2}
              changeType="increase"
              icon={<TrendingDownIcon className="w-5 h-5" />}
              formatValue={(val) => `${Number(val).toFixed(2)}%`}
              loading={loading}
              color="red"
            />

            <MetricCard
              title="Max Drawdown"
              value={-5.2}
              change={-0.8}
              changeType="decrease"
              icon={<ChartPieIcon className="w-5 h-5" />}
              formatValue={(val) => `${Number(val).toFixed(2)}%`}
              loading={loading}
              color="yellow"
            />

            <MetricCard
              title="Portfolio Beta"
              value={1.15}
              change={-0.05}
              changeType="decrease"
              icon={<ScaleIcon className="w-5 h-5" />}
              formatValue={(val) => Number(val).toFixed(2)}
              loading={loading}
              color="blue"
            />

            <MetricCard
              title="Active Alerts"
              value={3}
              icon={<BellAlertIcon className="w-5 h-5" />}
              loading={loading}
              color="red"
            />
          </div>
        </motion.div>

        {/* Risk Charts */}
        <motion.div variants={itemVariants} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* VaR Analysis */}
          <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white flex items-center">
                <TrendingDownIcon className="w-5 h-5 mr-2 text-trading-loss" />
                Value at Risk Analysis
              </h3>

              <div className="flex space-x-2">
                {['1D', '1W', '1M'].map((period) => (
                  <button
                    key={period}
                    onClick={() => setSelectedTimeframe(period)}
                    className={`px-3 py-1 sharp-button text-xs font-medium transition-all duration-200 ${
                      selectedTimeframe === period
                        ? 'bg-accent-blue text-white'
                        : 'bg-dark-300 text-dark-500 hover:bg-dark-400'
                    }`}
                  >
                    {period}
                  </button>
                ))}
              </div>
            </div>

            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={portfolioRisk}>
                <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" opacity={0.3} />
                <XAxis
                  dataKey="date"
                  stroke="#718096"
                  fontSize={12}
                  tickFormatter={(value) => new Date(value).toLocaleDateString()}
                />
                <YAxis stroke="#718096" fontSize={12} />
                <Tooltip content={<CustomTooltip />} />

                <Line
                  type="monotone"
                  dataKey="var95"
                  stroke="#ef4444"
                  strokeWidth={2}
                  name="VaR 95%"
                  strokeDasharray="5 5"
                />
                <Line
                  type="monotone"
                  dataKey="var99"
                  stroke="#dc2626"
                  strokeWidth={2}
                  name="VaR 99%"
                />
                <Line
                  type="monotone"
                  dataKey="realized"
                  stroke="#3b82f6"
                  strokeWidth={3}
                  name="Realized P&L"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Sector Risk Exposure */}
          <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <ChartPieIcon className="w-5 h-5 mr-2 text-accent-purple" />
              Sector Risk Exposure
            </h3>

            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={sectorExposure} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" opacity={0.3} />
                <XAxis type="number" stroke="#718096" fontSize={12} />
                <YAxis dataKey="sector" type="category" stroke="#718096" fontSize={12} />
                <Tooltip
                  formatter={(value, name) => [
                    `${(Number(value) * 100).toFixed(1)}%`,
                    name === 'exposure' ? 'Exposure' : 'Risk'
                  ]}
                />

                <Bar dataKey="exposure" fill="#3b82f6" name="Exposure" radius={[0, 2, 2, 0]} />
                <Bar dataKey="risk" fill="#ef4444" name="Risk" radius={[0, 2, 2, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Risk Alerts */}
        <motion.div variants={itemVariants}>
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
            <BellAlertIcon className="w-6 h-6 mr-2 text-trading-loss" />
            Active Risk Alerts
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {riskAlerts.map((alert, index) => (
              <motion.div
                key={alert.id}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                className={`
                  bg-dark-200 sharp-card p-6 border-l-4 border border-dark-300
                  ${alert.type === 'high' ? 'border-l-trading-loss' : ''}
                  ${alert.type === 'medium' ? 'border-l-accent-yellow' : ''}
                  ${alert.type === 'low' ? 'border-l-accent-blue' : ''}
                `}
              >
                <div className="flex items-start space-x-3">
                  <div className={`
                    p-2 sharp-card
                    ${alert.type === 'high' ? 'bg-trading-loss/20' : ''}
                    ${alert.type === 'medium' ? 'bg-accent-yellow/20' : ''}
                    ${alert.type === 'low' ? 'bg-accent-blue/20' : ''}
                  `}>
                    <ExclamationTriangleIcon className={`
                      w-5 h-5
                      ${alert.type === 'high' ? 'text-trading-loss' : ''}
                      ${alert.type === 'medium' ? 'text-accent-yellow' : ''}
                      ${alert.type === 'low' ? 'text-accent-blue' : ''}
                    `} />
                  </div>

                  <div className="flex-1 min-w-0">
                    <h4 className="text-white font-semibold mb-1">{alert.title}</h4>
                    <p className="text-dark-500 text-sm mb-2">{alert.description}</p>

                    <div className="flex items-center justify-between">
                      <span className="text-xs text-dark-500 flex items-center">
                        <ClockIcon className="w-4 h-4 mr-1" />
                        {alert.time}
                      </span>
                      <button className={`
                        px-3 py-1 sharp-button text-xs font-medium transition-colors duration-200
                        ${alert.type === 'high' ? 'bg-trading-loss/20 text-trading-loss hover:bg-trading-loss/30' : ''}
                        ${alert.type === 'medium' ? 'bg-accent-yellow/20 text-accent-yellow hover:bg-accent-yellow/30' : ''}
                        ${alert.type === 'low' ? 'bg-accent-blue/20 text-accent-blue hover:bg-accent-blue/30' : ''}
                      `}>
                        {alert.action}
                      </button>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Stress Testing */}
        <motion.div variants={itemVariants}>
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
            <ExclamationTriangleIcon className="w-6 h-6 mr-2 text-accent-yellow" />
            Stress Testing Scenarios
          </h2>

          <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-dark-300">
                    <th className="text-left py-3 px-4 text-dark-500 font-medium">Scenario</th>
                    <th className="text-right py-3 px-4 text-dark-500 font-medium">Portfolio Impact</th>
                    <th className="text-right py-3 px-4 text-dark-500 font-medium">Probability</th>
                    <th className="text-right py-3 px-4 text-dark-500 font-medium">Risk Score</th>
                  </tr>
                </thead>
                <tbody>
                  {stressTestScenarios.map((scenario, index) => {
                    const riskScore = Math.abs(scenario.portfolioImpact) * scenario.probability
                    return (
                      <motion.tr
                        key={index}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="border-b border-dark-400/50 hover:bg-dark-300/30 transition-colors duration-200"
                      >
                        <td className="py-3 px-4 text-white">{scenario.scenario}</td>
                        <td className="py-3 px-4 text-right">
                          <span className="font-mono text-trading-loss">
                            {scenario.portfolioImpact.toFixed(1)}%
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right">
                          <span className="font-mono text-dark-500">
                            {(scenario.probability * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right">
                          <div className="flex items-center justify-end space-x-2">
                            <div className={`
                              w-2 h-2 rounded-full
                              ${riskScore > 1.5 ? 'bg-trading-loss' :
                                riskScore > 0.8 ? 'bg-accent-yellow' : 'bg-trading-profit'}
                            `} />
                            <span className="font-mono text-sm text-white">
                              {riskScore.toFixed(2)}
                            </span>
                          </div>
                        </td>
                      </motion.tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </motion.div>

        {/* Enhanced Stress Testing with Real Data */}
        {stressTestResults && (
          <motion.div variants={itemVariants}>
            <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <ExclamationTriangleIcon className="w-5 h-5 mr-2 text-accent-yellow" />
                Advanced Stress Testing Results
              </h3>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Historical Stress Tests */}
                <div>
                  <h4 className="text-sm font-medium text-accent-blue mb-3">Historical Crisis Scenarios</h4>
                  <div className="space-y-3">
                    {Object.entries(stressTestResults.historical_scenarios || {}).map(([scenario, impact], index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-dark-300/50 sharp-card">
                        <div>
                          <div className="text-sm text-white font-medium">{scenario}</div>
                          <div className="text-xs text-dark-500">Based on historical data</div>
                        </div>
                        <div className={`text-sm font-mono font-medium ${
                          Number(impact) < -10 ? 'text-trading-loss' :
                          Number(impact) < -5 ? 'text-yellow-400' : 'text-trading-profit'
                        }`}>
                          {Number(impact) > 0 ? '+' : ''}{Number(impact).toFixed(1)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Custom Stress Scenarios */}
                <div>
                  <h4 className="text-sm font-medium text-accent-blue mb-3">Custom Scenario Analysis</h4>
                  <div className="space-y-3">
                    {Object.entries(stressTestResults.custom_scenarios || {}).map(([scenario, data], index) => (
                      <div key={index} className="p-3 bg-dark-300/50 sharp-card">
                        <div className="flex items-center justify-between mb-2">
                          <div className="text-sm text-white font-medium">{scenario}</div>
                          <div className={`text-sm font-mono font-medium ${
                            Number(data.impact) < -10 ? 'text-trading-loss' :
                            Number(data.impact) < -5 ? 'text-yellow-400' : 'text-trading-profit'
                          }`}>
                            {Number(data.impact) > 0 ? '+' : ''}{Number(data.impact).toFixed(1)}%
                          </div>
                        </div>
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-dark-500">Confidence: {Number(data.confidence * 100).toFixed(0)}%</span>
                          <span className="text-dark-500">Duration: {data.duration}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Stress Test Summary */}
              <div className="mt-6 pt-6 border-t border-dark-300">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-trading-loss mb-1">
                      {stressTestResults.summary?.worst_case_scenario || '-18.7%'}
                    </div>
                    <div className="text-xs text-dark-500">Worst Case Impact</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-yellow-400 mb-1">
                      {stressTestResults.summary?.average_impact || '-7.2%'}
                    </div>
                    <div className="text-xs text-dark-500">Average Impact</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-accent-blue mb-1">
                      {stressTestResults.summary?.recovery_time || '8.3'}
                    </div>
                    <div className="text-xs text-dark-500">Avg Recovery (days)</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-white mb-1">
                      {stressTestResults.summary?.scenarios_tested || '47'}
                    </div>
                    <div className="text-xs text-dark-500">Scenarios Tested</div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Monte Carlo Simulation Results */}
        {monteCarloResults && (
          <motion.div variants={itemVariants}>
            <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <ChartPieIcon className="w-5 h-5 mr-2 text-accent-purple" />
                Monte Carlo Simulation Analysis
              </h3>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Simulation Parameters */}
                <div>
                  <h4 className="text-sm font-medium text-accent-blue mb-3">Simulation Parameters</h4>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">Number of Simulations</span>
                      <span className="text-xs text-dark-500 font-mono">
                        {monteCarloResults.parameters?.num_simulations || '10,000'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">Time Horizon</span>
                      <span className="text-xs text-dark-500 font-mono">
                        {monteCarloResults.parameters?.time_horizon || '252 days'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">Confidence Level</span>
                      <span className="text-xs text-dark-500 font-mono">
                        {monteCarloResults.parameters?.confidence_level || '95%'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">Correlation Model</span>
                      <span className="text-xs text-dark-500 font-mono">
                        {monteCarloResults.parameters?.correlation_model || 'Dynamic'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Distribution Analysis */}
                <div>
                  <h4 className="text-sm font-medium text-accent-blue mb-3">Return Distribution</h4>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">Expected Return</span>
                      <span className="text-xs text-trading-profit font-mono">
                        +{Number(monteCarloResults.distribution?.expected_return || 0.087) * 100}%
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">Standard Deviation</span>
                      <span className="text-xs text-yellow-400 font-mono">
                        {Number(monteCarloResults.distribution?.std_deviation || 0.156) * 100}%
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">Skewness</span>
                      <span className="text-xs text-dark-500 font-mono">
                        {Number(monteCarloResults.distribution?.skewness || -0.23).toFixed(2)}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">Kurtosis</span>
                      <span className="text-xs text-dark-500 font-mono">
                        {Number(monteCarloResults.distribution?.kurtosis || 3.45).toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Risk Percentiles */}
                <div>
                  <h4 className="text-sm font-medium text-accent-blue mb-3">Risk Percentiles</h4>
                  <div className="space-y-2">
                    {Object.entries(monteCarloResults.percentiles || {
                      '5th': -0.087,
                      '10th': -0.064,
                      '25th': -0.032,
                      '75th': 0.045,
                      '90th': 0.078,
                      '95th': 0.098
                    }).map(([percentile, value], index) => (
                      <div key={index} className="space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-white">{percentile} Percentile</span>
                          <span className={`text-xs font-mono ${
                            Number(value) < 0 ? 'text-trading-loss' : 'text-trading-profit'
                          }`}>
                            {Number(value) > 0 ? '+' : ''}{(Number(value) * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full bg-dark-300 h-1">
                          <div
                            className={`h-1 ${
                              Number(value) < 0 ? 'bg-trading-loss' : 'bg-trading-profit'
                            }`}
                            style={{ width: `${Math.abs(Number(value)) * 500}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Tail Risk Analysis */}
              <div className="mt-6 pt-6 border-t border-dark-300">
                <h4 className="text-sm font-medium text-accent-blue mb-3">Tail Risk Analysis</h4>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-lg font-bold text-trading-loss mb-1">
                      {Number(monteCarloResults.tail_risk?.var_95 || -0.034) * 100}%
                    </div>
                    <div className="text-xs text-dark-500">VaR (95%)</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold text-trading-loss mb-1">
                      {Number(monteCarloResults.tail_risk?.var_99 || -0.052) * 100}%
                    </div>
                    <div className="text-xs text-dark-500">VaR (99%)</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold text-trading-loss mb-1">
                      {Number(monteCarloResults.tail_risk?.expected_shortfall || -0.067) * 100}%
                    </div>
                    <div className="text-xs text-dark-500">Expected Shortfall</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-bold text-yellow-400 mb-1">
                      {Number(monteCarloResults.tail_risk?.max_drawdown || -0.089) * 100}%
                    </div>
                    <div className="text-xs text-dark-500">Max Drawdown</div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </motion.div>
    </Layout>
  )
}