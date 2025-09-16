'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Layout from '@/components/Layout/Layout'
import MetricCard from '@/components/MetricCard'
import { api } from '@/lib/api'
import {
  ArrowTrendingUpIcon,
  SparklesIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  CpuChipIcon,
  EyeIcon,
} from '@heroicons/react/24/outline'

interface EmergingStock {
  symbol: string
  company_name: string
  score: number
  growth_potential: string
  timeframe: string
  key_drivers: string[]
  risk_factors: string[]
  market_cap: number
  sector: string
  confidence: number
}

interface EmergingStocksData {
  emerging_stocks: EmergingStock[]
  summary: {
    total_opportunities: number
    avg_score: number
    high_potential_count: number
    sectors_represented: string[]
  }
}

export default function EmergingStocksPage() {
  const [emergingData, setEmergingData] = useState<EmergingStocksData | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedSector, setSelectedSector] = useState<string | null>(null)
  const [selectedTimeframe, setSelectedTimeframe] = useState<string | null>(null)

  useEffect(() => {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
    const loadEmergingStocks = async () => {
      setLoading(true)
      try {
        const response = await fetch(`${API_URL}/api/emerging-stocks`)
        if (response.ok) {
          const data = await response.json();
          setEmergingData(data);
        } else {
          throw new Error('API not available');
        }
      } catch (error) {
        console.error('Failed to load emerging stocks data:', error)
        // Mock data fallback
        setEmergingData({
          emerging_stocks: [
            {
              symbol: 'PLTR',
              company_name: 'Palantir Technologies',
              score: 87.5,
              growth_potential: 'high',
              timeframe: 'medium',
              key_drivers: ['AI expansion', 'Government contracts', 'Data analytics growth'],
              risk_factors: ['High valuation', 'Competition'],
              market_cap: 45.2,
              sector: 'Technology',
              confidence: 0.82
            },
            {
              symbol: 'RIVN',
              company_name: 'Rivian Automotive',
              score: 76.3,
              growth_potential: 'high',
              timeframe: 'long',
              key_drivers: ['EV market growth', 'Amazon partnership', 'Manufacturing scale-up'],
              risk_factors: ['Production challenges', 'Competition from Tesla'],
              market_cap: 18.7,
              sector: 'Automotive',
              confidence: 0.74
            },
            {
              symbol: 'SNOW',
              company_name: 'Snowflake Inc.',
              score: 81.2,
              growth_potential: 'high',
              timeframe: 'medium',
              key_drivers: ['Cloud data growth', 'Enterprise adoption', 'AI integration'],
              risk_factors: ['High competition', 'Valuation concerns'],
              market_cap: 52.8,
              sector: 'Technology',
              confidence: 0.79
            },
            {
              symbol: 'CRSP',
              company_name: 'CRISPR Therapeutics',
              score: 73.6,
              growth_potential: 'high',
              timeframe: 'long',
              key_drivers: ['Gene therapy breakthrough', 'Pipeline progress', 'Regulatory approvals'],
              risk_factors: ['Clinical trial risks', 'Regulatory uncertainty'],
              market_cap: 8.3,
              sector: 'Biotechnology',
              confidence: 0.68
            }
          ],
          summary: {
            total_opportunities: 4,
            avg_score: 79.65,
            high_potential_count: 4,
            sectors_represented: ['Technology', 'Automotive', 'Biotechnology']
          }
        })
      }
      setLoading(false)
    }

    loadEmergingStocks()
  }, [])

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-trading-profit'
    if (score >= 60) return 'text-yellow-400'
    return 'text-trading-loss'
  }

  const getPotentialBadgeColor = (potential: string) => {
    switch (potential) {
      case 'high':
        return 'bg-trading-profit/20 text-trading-profit'
      case 'medium':
        return 'bg-yellow-400/20 text-yellow-400'
      case 'low':
        return 'bg-trading-loss/20 text-trading-loss'
      default:
        return 'bg-dark-300 text-dark-500'
    }
  }

  const getTimeframeBadgeColor = (timeframe: string) => {
    switch (timeframe) {
      case 'short':
        return 'bg-accent-blue/20 text-accent-blue'
      case 'medium':
        return 'bg-accent-purple/20 text-accent-purple'
      case 'long':
        return 'bg-orange-400/20 text-orange-400'
      default:
        return 'bg-dark-300 text-dark-500'
    }
  }

  const filteredStocks = emergingData?.emerging_stocks.filter(stock => {
    const sectorMatch = !selectedSector || stock.sector === selectedSector
    const timeframeMatch = !selectedTimeframe || stock.timeframe === selectedTimeframe
    return sectorMatch && timeframeMatch
  })

  const sectors = Array.from(new Set(emergingData?.emerging_stocks.map(stock => stock.sector) || []))
  const timeframes = Array.from(new Set(emergingData?.emerging_stocks.map(stock => stock.timeframe) || []))

  return (
    <Layout title="Emerging Stocks Detection" subtitle="AI-Powered Growth Stock Identification">
      <div className="space-y-6">
        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <MetricCard
            title="Total Opportunities"
            value={emergingData?.summary.total_opportunities || 0}
            icon={<EyeIcon className="w-5 h-5" />}
            loading={loading}
          />
          <MetricCard
            title="Average Score"
            value={`${emergingData?.summary.avg_score || 0}/100`}
            icon={<ArrowTrendingUpIcon className="w-5 h-5" />}
            loading={loading}
          />
          <MetricCard
            title="High Potential"
            value={emergingData?.summary.high_potential_count || 0}
            icon={<SparklesIcon className="w-5 h-5" />}
            loading={loading}
          />
          <MetricCard
            title="Sectors Covered"
            value={emergingData?.summary.sectors_represented.length || 0}
            icon={<CpuChipIcon className="w-5 h-5" />}
            loading={loading}
          />
        </div>

        {/* Filter Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-dark-200 sharp-card p-4 border border-dark-300"
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="text-sm font-medium text-white mb-2">Filter by Sector</h4>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => setSelectedSector(null)}
                  className={`px-3 py-1 sharp-button text-sm transition-colors ${
                    selectedSector === null
                      ? 'bg-accent-blue text-white'
                      : 'bg-dark-300 text-dark-500 hover:text-white'
                  }`}
                >
                  All Sectors
                </button>
                {sectors.map((sector) => (
                  <button
                    key={sector}
                    onClick={() => setSelectedSector(sector)}
                    className={`px-3 py-1 sharp-button text-sm transition-colors ${
                      selectedSector === sector
                        ? 'bg-accent-blue text-white'
                        : 'bg-dark-300 text-dark-500 hover:text-white'
                    }`}
                  >
                    {sector}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <h4 className="text-sm font-medium text-white mb-2">Filter by Timeframe</h4>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => setSelectedTimeframe(null)}
                  className={`px-3 py-1 sharp-button text-sm transition-colors ${
                    selectedTimeframe === null
                      ? 'bg-accent-blue text-white'
                      : 'bg-dark-300 text-dark-500 hover:text-white'
                  }`}
                >
                  All Timeframes
                </button>
                {timeframes.map((timeframe) => (
                  <button
                    key={timeframe}
                    onClick={() => setSelectedTimeframe(timeframe)}
                    className={`px-3 py-1 sharp-button text-sm transition-colors ${
                      selectedTimeframe === timeframe
                        ? 'bg-accent-blue text-white'
                        : 'bg-dark-300 text-dark-500 hover:text-white'
                    }`}
                  >
                    {timeframe.charAt(0).toUpperCase() + timeframe.slice(1)}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </motion.div>

        {/* Emerging Stocks Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {filteredStocks?.map((stock, index) => (
            <motion.div
              key={stock.symbol}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 + index * 0.1 }}
              className="bg-dark-200 sharp-card p-6 border border-dark-300 hover:border-accent-blue/50 transition-colors"
            >
              {/* Header */}
              <div className="flex items-start justify-between mb-4">
                <div>
                  <div className="flex items-center space-x-2">
                    <h3 className="text-lg font-semibold text-white">{stock.symbol}</h3>
                    <span className={`px-2 py-1 sharp-card text-xs font-medium ${getScoreColor(stock.score)}`}>
                      {stock.score.toFixed(1)}
                    </span>
                  </div>
                  <p className="text-sm text-dark-400">{stock.company_name}</p>
                  <p className="text-xs text-dark-500 mt-1">Market Cap: ${stock.market_cap.toFixed(1)}B</p>
                </div>
                <div className="flex flex-col items-end space-y-1">
                  <span className={`px-2 py-1 sharp-card text-xs font-medium ${getPotentialBadgeColor(stock.growth_potential)}`}>
                    {stock.growth_potential.toUpperCase()}
                  </span>
                  <span className={`px-2 py-1 sharp-card text-xs font-medium ${getTimeframeBadgeColor(stock.timeframe)}`}>
                    {stock.timeframe.toUpperCase()}
                  </span>
                </div>
              </div>

              {/* Confidence Bar */}
              <div className="mb-4">
                <div className="flex items-center justify-between text-xs text-dark-500 mb-1">
                  <span>AI Confidence</span>
                  <span>{(stock.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="w-full bg-dark-300 h-2 sharp-card overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${stock.confidence * 100}%` }}
                    transition={{ delay: 0.5 + index * 0.1, duration: 0.8 }}
                    className="h-full bg-gradient-to-r from-accent-blue to-accent-purple"
                  />
                </div>
              </div>

              {/* Key Drivers */}
              <div className="mb-4">
                <h4 className="text-sm font-medium text-trading-profit mb-2 flex items-center">
                  <ArrowTrendingUpIcon className="w-4 h-4 mr-1" />
                  Growth Drivers
                </h4>
                <div className="space-y-1">
                  {stock.key_drivers.map((driver, i) => (
                    <div key={i} className="flex items-center text-xs text-dark-400">
                      <div className="w-1 h-1 bg-trading-profit mr-2 flex-shrink-0" />
                      {driver}
                    </div>
                  ))}
                </div>
              </div>

              {/* Risk Factors */}
              <div>
                <h4 className="text-sm font-medium text-trading-loss mb-2 flex items-center">
                  <ExclamationTriangleIcon className="w-4 h-4 mr-1" />
                  Risk Factors
                </h4>
                <div className="space-y-1">
                  {stock.risk_factors.map((risk, i) => (
                    <div key={i} className="flex items-center text-xs text-dark-400">
                      <div className="w-1 h-1 bg-trading-loss mr-2 flex-shrink-0" />
                      {risk}
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* AI Detection Methodology */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="bg-dark-200 sharp-card p-6 border border-dark-300"
        >
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <CpuChipIcon className="w-5 h-5 mr-2 text-accent-blue" />
            AI Detection Methodology
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-accent-blue">Multi-Factor Analysis</h4>
              <p className="text-sm text-dark-400">
                Combines news sentiment, technical indicators, fundamental analysis, and market momentum
                to identify growth potential.
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-accent-blue">Growth Keywords</h4>
              <p className="text-sm text-dark-400">
                Scans for innovation indicators: AI, partnerships, breakthrough technologies,
                expansion plans, and market disruption signals.
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-accent-blue">Risk Assessment</h4>
              <p className="text-sm text-dark-400">
                Evaluates market cap constraints, competitive landscape, regulatory risks,
                and execution challenges for balanced scoring.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </Layout>
  )
}