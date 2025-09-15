'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Layout from '@/components/Layout/Layout'
import MetricCard from '@/components/MetricCard'
import { api } from '@/lib/api'
import {
  PresentationChartLineIcon,
  TrendingUpIcon,
  ArrowTrendingDownIcon,
  ClockIcon,
  ChartBarIcon,
  SignalIcon,
  EyeIcon,
} from '@heroicons/react/24/outline'

interface TimeframeData {
  trend: string
  rsi: number
  macd: string
  volume: string
}

interface SymbolAnalysis {
  timeframes: {
    [key: string]: TimeframeData
  }
  overall_signal: string
  confidence: number
  support_levels: number[]
  resistance_levels: number[]
}

interface TechnicalData {
  technical_analysis: {
    [symbol: string]: SymbolAnalysis
  }
  summary: {
    bullish_symbols: number
    bearish_symbols: number
    neutral_symbols: number
    high_volume_symbols: string[]
  }
}

const TIMEFRAMES = [
  { key: '1m', label: '1 Minute', color: 'bg-red-500' },
  { key: '5m', label: '5 Minutes', color: 'bg-orange-500' },
  { key: '15m', label: '15 Minutes', color: 'bg-yellow-500' },
  { key: '1h', label: '1 Hour', color: 'bg-green-500' },
  { key: '4h', label: '4 Hours', color: 'bg-blue-500' },
  { key: 'daily', label: 'Daily', color: 'bg-purple-500' },
]

export default function TechnicalAnalysisPage() {
  const [technicalData, setTechnicalData] = useState<TechnicalData | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedSymbol, setSelectedSymbol] = useState<string>('AAPL')

  useEffect(() => {
    const loadTechnicalData = async () => {
      setLoading(true)
      try {
        const response = await api.get('/technical-analysis')
        setTechnicalData(response.data)
      } catch (error) {
        console.error('Failed to load technical analysis data:', error)
        // Mock data fallback
        setTechnicalData({
          technical_analysis: {
            'AAPL': {
              timeframes: {
                '1m': { trend: 'Bullish', rsi: 65.2, macd: 'Positive', volume: 'High' },
                '5m': { trend: 'Bullish', rsi: 62.8, macd: 'Positive', volume: 'Normal' },
                '15m': { trend: 'Neutral', rsi: 58.3, macd: 'Neutral', volume: 'Normal' },
                '1h': { trend: 'Bullish', rsi: 61.7, macd: 'Positive', volume: 'High' },
                '4h': { trend: 'Bullish', rsi: 59.2, macd: 'Positive', volume: 'Normal' },
                'daily': { trend: 'Bullish', rsi: 67.5, macd: 'Strong Positive', volume: 'High' },
              },
              overall_signal: 'Buy',
              confidence: 0.78,
              support_levels: [180.0, 175.0, 170.0],
              resistance_levels: [190.0, 195.0, 200.0],
            },
            'TSLA': {
              timeframes: {
                '1m': { trend: 'Bearish', rsi: 35.2, macd: 'Negative', volume: 'High' },
                '5m': { trend: 'Bearish', rsi: 32.8, macd: 'Negative', volume: 'High' },
                '15m': { trend: 'Bearish', rsi: 38.3, macd: 'Negative', volume: 'Normal' },
                '1h': { trend: 'Neutral', rsi: 41.7, macd: 'Neutral', volume: 'Normal' },
                '4h': { trend: 'Bearish', rsi: 39.2, macd: 'Negative', volume: 'High' },
                'daily': { trend: 'Bearish', rsi: 32.5, macd: 'Strong Negative', volume: 'High' },
              },
              overall_signal: 'Sell',
              confidence: 0.72,
              support_levels: [220.0, 210.0, 200.0],
              resistance_levels: [240.0, 250.0, 260.0],
            },
          },
          summary: {
            bullish_symbols: 15,
            bearish_symbols: 3,
            neutral_symbols: 7,
            high_volume_symbols: ['AAPL', 'TSLA', 'NVDA'],
          },
        })
      }
      setLoading(false)
    }

    loadTechnicalData()
  }, [])

  const getTrendColor = (trend: string) => {
    switch (trend.toLowerCase()) {
      case 'bullish':
        return 'text-trading-profit bg-trading-profit/20'
      case 'bearish':
        return 'text-trading-loss bg-trading-loss/20'
      case 'neutral':
        return 'text-yellow-400 bg-yellow-400/20'
      default:
        return 'text-dark-500 bg-dark-300/20'
    }
  }

  const getMacdColor = (macd: string) => {
    if (macd.toLowerCase().includes('positive')) return 'text-trading-profit'
    if (macd.toLowerCase().includes('negative')) return 'text-trading-loss'
    return 'text-yellow-400'
  }

  const getVolumeColor = (volume: string) => {
    switch (volume.toLowerCase()) {
      case 'high':
        return 'text-trading-profit'
      case 'normal':
        return 'text-yellow-400'
      case 'low':
        return 'text-trading-loss'
      default:
        return 'text-dark-500'
    }
  }

  const getRsiColor = (rsi: number) => {
    if (rsi > 70) return 'text-trading-loss' // Overbought
    if (rsi < 30) return 'text-trading-profit' // Oversold
    return 'text-yellow-400' // Neutral
  }

  const getSignalColor = (signal: string) => {
    switch (signal.toLowerCase()) {
      case 'buy':
      case 'strong buy':
        return 'text-trading-profit bg-trading-profit/20'
      case 'sell':
      case 'strong sell':
        return 'text-trading-loss bg-trading-loss/20'
      case 'hold':
        return 'text-yellow-400 bg-yellow-400/20'
      default:
        return 'text-dark-500 bg-dark-300/20'
    }
  }

  const symbols = Object.keys(technicalData?.technical_analysis || {})
  const currentSymbolData = technicalData?.technical_analysis[selectedSymbol]

  return (
    <Layout title="Technical Analysis" subtitle="Multi-timeframe Technical Indicators & Signals">
      <div className="space-y-6">
        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <MetricCard
            title="Bullish Symbols"
            value={technicalData?.summary.bullish_symbols || 0}
            icon={TrendingUpIcon}
            loading={loading}
            trend="positive"
            formatType="number"
          />
          <MetricCard
            title="Bearish Symbols"
            value={technicalData?.summary.bearish_symbols || 0}
            icon={ArrowTrendingDownIcon}
            loading={loading}
            trend="negative"
            formatType="number"
          />
          <MetricCard
            title="Neutral Symbols"
            value={technicalData?.summary.neutral_symbols || 0}
            icon={EyeIcon}
            loading={loading}
            trend="neutral"
            formatType="number"
          />
          <MetricCard
            title="High Volume"
            value={technicalData?.summary.high_volume_symbols.length || 0}
            icon={ChartBarIcon}
            loading={loading}
            trend="neutral"
            formatType="number"
          />
        </div>

        {/* Symbol Selector */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-dark-200 sharp-card p-4 border border-dark-300"
        >
          <div className="flex items-center space-x-4">
            <span className="text-sm font-medium text-white">Select Symbol:</span>
            <div className="flex flex-wrap gap-2">
              {symbols.map((symbol) => (
                <button
                  key={symbol}
                  onClick={() => setSelectedSymbol(symbol)}
                  className={`px-3 py-1 sharp-button text-sm transition-colors ${
                    selectedSymbol === symbol
                      ? 'bg-accent-blue text-white'
                      : 'bg-dark-300 text-dark-500 hover:text-white'
                  }`}
                >
                  {symbol}
                </button>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Symbol Analysis */}
        {currentSymbolData && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-dark-200 sharp-card p-6 border border-dark-300"
          >
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="text-xl font-semibold text-white">{selectedSymbol} Technical Analysis</h3>
                <p className="text-sm text-dark-400">Multi-timeframe technical indicators overview</p>
              </div>
              <div className="flex items-center space-x-4">
                <div className="text-right">
                  <p className="text-xs text-dark-500">Overall Signal</p>
                  <span className={`px-3 py-1 sharp-card text-sm font-medium ${getSignalColor(currentSymbolData.overall_signal)}`}>
                    {currentSymbolData.overall_signal}
                  </span>
                </div>
                <div className="text-right">
                  <p className="text-xs text-dark-500">Confidence</p>
                  <p className="text-lg font-semibold text-white">{(currentSymbolData.confidence * 100).toFixed(0)}%</p>
                </div>
              </div>
            </div>

            {/* Timeframe Analysis Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
              {TIMEFRAMES.map((timeframe) => {
                const data = currentSymbolData.timeframes[timeframe.key]
                if (!data) return null

                return (
                  <motion.div
                    key={timeframe.key}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.4 + TIMEFRAMES.indexOf(timeframe) * 0.1 }}
                    className="bg-dark-300 sharp-card p-4 border border-dark-400"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-sm font-medium text-white">{timeframe.label}</h4>
                      <div className={`w-3 h-3 ${timeframe.color} sharp-card`} />
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-dark-500">Trend</span>
                        <span className={`px-2 py-1 sharp-card text-xs font-medium ${getTrendColor(data.trend)}`}>
                          {data.trend}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-dark-500">RSI</span>
                        <span className={`text-xs font-medium ${getRsiColor(data.rsi)}`}>
                          {data.rsi.toFixed(1)}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-dark-500">MACD</span>
                        <span className={`text-xs font-medium ${getMacdColor(data.macd)}`}>
                          {data.macd}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-dark-500">Volume</span>
                        <span className={`text-xs font-medium ${getVolumeColor(data.volume)}`}>
                          {data.volume}
                        </span>
                      </div>
                    </div>
                  </motion.div>
                )
              })}
            </div>

            {/* Support & Resistance Levels */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="text-sm font-medium text-trading-profit mb-3 flex items-center">
                  <TrendingUpIcon className="w-4 h-4 mr-1" />
                  Support Levels
                </h4>
                <div className="space-y-2">
                  {currentSymbolData.support_levels.map((level, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.6 + index * 0.1 }}
                      className="flex items-center justify-between p-2 bg-dark-300/50 sharp-card"
                    >
                      <span className="text-xs text-dark-500">Support {index + 1}</span>
                      <span className="text-sm font-medium text-trading-profit">${level.toFixed(2)}</span>
                    </motion.div>
                  ))}
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium text-trading-loss mb-3 flex items-center">
                  <ArrowTrendingDownIcon className="w-4 h-4 mr-1" />
                  Resistance Levels
                </h4>
                <div className="space-y-2">
                  {currentSymbolData.resistance_levels.map((level, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.6 + index * 0.1 }}
                      className="flex items-center justify-between p-2 bg-dark-300/50 sharp-card"
                    >
                      <span className="text-xs text-dark-500">Resistance {index + 1}</span>
                      <span className="text-sm font-medium text-trading-loss">${level.toFixed(2)}</span>
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* High Volume Symbols */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="bg-dark-200 sharp-card p-6 border border-dark-300"
        >
          <div className="flex items-center space-x-2 mb-4">
            <SignalIcon className="w-5 h-5 text-accent-blue" />
            <h3 className="text-lg font-semibold text-white">High Volume Activity</h3>
          </div>
          <div className="flex flex-wrap gap-2">
            {technicalData?.summary.high_volume_symbols.map((symbol, index) => (
              <motion.div
                key={symbol}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.8 + index * 0.1 }}
                className="px-3 py-2 bg-accent-blue/20 text-accent-blue sharp-card font-medium cursor-pointer hover:bg-accent-blue/30 transition-colors"
                onClick={() => setSelectedSymbol(symbol)}
              >
                {symbol}
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Technical Indicators Legend */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="bg-dark-200 sharp-card p-6 border border-dark-300"
        >
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <PresentationChartLineIcon className="w-5 h-5 mr-2 text-accent-blue" />
            Technical Indicators Guide
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <h4 className="text-sm font-medium text-accent-blue mb-2">RSI (Relative Strength Index)</h4>
              <div className="space-y-1 text-sm text-dark-400">
                <div className="flex justify-between">
                  <span>&gt; 70:</span>
                  <span className="text-trading-loss">Overbought</span>
                </div>
                <div className="flex justify-between">
                  <span>30-70:</span>
                  <span className="text-yellow-400">Neutral</span>
                </div>
                <div className="flex justify-between">
                  <span>&lt; 30:</span>
                  <span className="text-trading-profit">Oversold</span>
                </div>
              </div>
            </div>
            <div>
              <h4 className="text-sm font-medium text-accent-blue mb-2">MACD (Moving Average Convergence)</h4>
              <div className="space-y-1 text-sm text-dark-400">
                <div className="flex justify-between">
                  <span>Positive:</span>
                  <span className="text-trading-profit">Bullish signal</span>
                </div>
                <div className="flex justify-between">
                  <span>Neutral:</span>
                  <span className="text-yellow-400">No clear trend</span>
                </div>
                <div className="flex justify-between">
                  <span>Negative:</span>
                  <span className="text-trading-loss">Bearish signal</span>
                </div>
              </div>
            </div>
            <div>
              <h4 className="text-sm font-medium text-accent-blue mb-2">Volume Analysis</h4>
              <div className="space-y-1 text-sm text-dark-400">
                <div className="flex justify-between">
                  <span>High:</span>
                  <span className="text-trading-profit">Strong conviction</span>
                </div>
                <div className="flex justify-between">
                  <span>Normal:</span>
                  <span className="text-yellow-400">Average activity</span>
                </div>
                <div className="flex justify-between">
                  <span>Low:</span>
                  <span className="text-trading-loss">Weak conviction</span>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </Layout>
  )
}