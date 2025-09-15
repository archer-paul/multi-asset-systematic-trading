'use client'

import { useMemo } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Area,
  AreaChart,
} from 'recharts'
import { motion } from 'framer-motion'

interface DataPoint {
  date: string
  portfolio: number
  sp500: number
  drawdown?: number
}

interface PerformanceChartProps {
  data: DataPoint[]
  title?: string
  height?: number
  showDrawdown?: boolean
  loading?: boolean
}

export default function PerformanceChart({
  data,
  title = "Portfolio Performance vs S&P 500",
  height = 400,
  showDrawdown = false,
  loading = false
}: PerformanceChartProps) {

  const chartData = useMemo(() => {
    return data.map(point => ({
      ...point,
      portfolioReturn: ((point.portfolio - 1) * 100).toFixed(2),
      sp500Return: ((point.sp500 - 1) * 100).toFixed(2),
      outperformance: (((point.portfolio / point.sp500) - 1) * 100).toFixed(2),
    }))
  }, [data])

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-dark-200 border border-dark-300 sharp-card p-4 shadow-lg">
          <p className="text-white font-medium mb-2">{label}</p>
          <div className="space-y-1">
            {payload.map((entry: any, index: number) => (
              <div key={index} className="flex items-center space-x-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: entry.color }}
                />
                <span className="text-sm text-dark-500">{entry.name}:</span>
                <span className="text-sm text-white font-mono">
                  {entry.value}%
                </span>
              </div>
            ))}
            {payload[0] && payload[1] && (
              <div className="flex items-center space-x-2 pt-1 border-t border-dark-400">
                <span className="text-sm text-dark-500">Outperformance:</span>
                <span className={`text-sm font-mono ${
                  parseFloat(payload[0].payload.outperformance) >= 0
                    ? 'text-trading-profit'
                    : 'text-trading-loss'
                }`}>
                  {payload[0].payload.outperformance}%
                </span>
              </div>
            )}
          </div>
        </div>
      )
    }
    return null
  }

  if (loading) {
    return (
      <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
        <h3 className="text-lg font-semibold text-white mb-4">{title}</h3>
        <div className="flex items-center justify-center h-96">
          <div className="loading-dots">
            <div></div>
            <div></div>
            <div></div>
            <div></div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="bg-dark-200 sharp-card p-6 border border-dark-300"
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-white">{title}</h3>

        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-accent-blue rounded-full"></div>
            <span className="text-sm text-dark-500">Portfolio</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-accent-yellow rounded-full"></div>
            <span className="text-sm text-dark-500">S&P 500</span>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        {showDrawdown ? (
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" opacity={0.3} />
            <XAxis
              dataKey="date"
              stroke="#718096"
              fontSize={12}
              tickFormatter={(value) => new Date(value).toLocaleDateString()}
            />
            <YAxis
              stroke="#718096"
              fontSize={12}
              tickFormatter={(value) => `${value}%`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />

            <Area
              type="monotone"
              dataKey="portfolioReturn"
              stackId="1"
              stroke="#3b82f6"
              fill="url(#portfolioGradient)"
              name="Portfolio Return"
            />
            <Area
              type="monotone"
              dataKey="sp500Return"
              stackId="2"
              stroke="#f59e0b"
              fill="url(#sp500Gradient)"
              name="S&P 500 Return"
            />

            <defs>
              <linearGradient id="portfolioGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.05}/>
              </linearGradient>
              <linearGradient id="sp500Gradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.05}/>
              </linearGradient>
            </defs>
          </AreaChart>
        ) : (
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" opacity={0.3} />
            <XAxis
              dataKey="date"
              stroke="#718096"
              fontSize={12}
              tickFormatter={(value) => new Date(value).toLocaleDateString()}
            />
            <YAxis
              stroke="#718096"
              fontSize={12}
              tickFormatter={(value) => `${value}%`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />

            <Line
              type="monotone"
              dataKey="portfolioReturn"
              stroke="#3b82f6"
              strokeWidth={3}
              dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6, stroke: '#3b82f6', strokeWidth: 2, fill: '#fff' }}
              name="Portfolio Return"
            />
            <Line
              type="monotone"
              dataKey="sp500Return"
              stroke="#f59e0b"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={{ fill: '#f59e0b', strokeWidth: 2, r: 3 }}
              name="S&P 500 Return"
            />
          </LineChart>
        )}
      </ResponsiveContainer>
    </motion.div>
  )
}