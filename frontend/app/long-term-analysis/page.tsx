'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Layout from '@/components/Layout/Layout'
import MetricCard from '@/components/MetricCard'
import { api } from '@/lib/api'
import {
  ClockIcon,
  ArrowTrendingUpIcon,
  CurrencyDollarIcon,
  ChartBarIcon,
  ShieldCheckIcon,
  SparklesIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline'

interface LongTermRecommendation {
  symbol: string
  company_name: string
  recommendation: string
  target_price_3y: number
  target_price_5y: number
  current_price: number
  dcf_valuation: number
  esg_score: number
  sector_outlook: string
  key_catalysts: string[]
  risks: string[]
  confidence: number
}

interface MarketOutlook {
  overall_sentiment: string
  sector_rotations: string[]
  macro_trends: string[]
  risk_factors: string[]
}

interface LongTermData {
  recommendations: LongTermRecommendation[]
  market_outlook: MarketOutlook
}

export default function LongTermAnalysisPage() {
  const [longTermData, setLongTermData] = useState<LongTermData | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedTimeframe, setSelectedTimeframe] = useState<'3y' | '5y'>('3y')

  useEffect(() => {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
    const loadLongTermData = async () => {
      setLoading(true)
      try {
        const response = await fetch(`${API_URL}/api/long-term-analysis`)
        if (response.ok) {
          const data = await response.json();
          setLongTermData(data);
        } else {
          throw new Error('API not available');
        }
      } catch (error) {
        console.error('Failed to load long-term analysis data:', error)
        // Mock data fallback
        setLongTermData({
          recommendations: [
            {
              symbol: 'AAPL',
              company_name: 'Apple Inc.',
              recommendation: 'Strong Buy',
              target_price_3y: 250.0,
              target_price_5y: 320.0,
              current_price: 185.0,
              dcf_valuation: 235.0,
              esg_score: 8.5,
              sector_outlook: 'Positive',
              key_catalysts: ['Services growth', 'AR/VR adoption', 'Emerging markets'],
              risks: ['Regulatory pressure', 'China dependency'],
              confidence: 0.85
            },
            {
              symbol: 'MSFT',
              company_name: 'Microsoft Corporation',
              recommendation: 'Buy',
              target_price_3y: 420.0,
              target_price_5y: 550.0,
              current_price: 375.0,
              dcf_valuation: 445.0,
              esg_score: 9.2,
              sector_outlook: 'Very Positive',
              key_catalysts: ['Azure growth', 'AI integration', 'Enterprise solutions'],
              risks: ['Competition', 'Economic slowdown'],
              confidence: 0.89
            },
            {
              symbol: 'NVDA',
              company_name: 'NVIDIA Corporation',
              recommendation: 'Strong Buy',
              target_price_3y: 650.0,
              target_price_5y: 900.0,
              current_price: 450.0,
              dcf_valuation: 580.0,
              esg_score: 7.8,
              sector_outlook: 'Very Positive',
              key_catalysts: ['AI revolution', 'Data center growth', 'Autonomous vehicles'],
              risks: ['Cyclical nature', 'Geopolitical tensions'],
              confidence: 0.92
            }
          ],
          market_outlook: {
            overall_sentiment: 'Cautiously Optimistic',
            sector_rotations: ['Technology', 'Healthcare', 'Clean Energy'],
            macro_trends: ['Digital transformation', 'ESG adoption', 'Demographic shifts'],
            risk_factors: ['Inflation', 'Geopolitical tensions', 'Interest rates']
          }
        })
      }
      setLoading(false)
    }

    loadLongTermData()
  }, [])

  const getRecommendationColor = (recommendation: string) => {
    switch (recommendation.toLowerCase()) {
      case 'strong buy':
        return 'text-trading-profit bg-trading-profit/20'
      case 'buy':
        return 'text-yellow-400 bg-yellow-400/20'
      case 'hold':
        return 'text-blue-400 bg-blue-400/20'
      case 'sell':
        return 'text-orange-400 bg-orange-400/20'
      case 'strong sell':
        return 'text-trading-loss bg-trading-loss/20'
      default:
        return 'text-dark-500 bg-dark-300'
    }
  }

  const getSectorOutlookColor = (outlook: string) => {
    switch (outlook.toLowerCase()) {
      case 'very positive':
        return 'text-trading-profit'
      case 'positive':
        return 'text-yellow-400'
      case 'neutral':
        return 'text-blue-400'
      case 'negative':
        return 'text-orange-400'
      case 'very negative':
        return 'text-trading-loss'
      default:
        return 'text-dark-500'
    }
  }

  const calculateUpside = (current: number, target: number) => {
    return ((target - current) / current) * 100
  }

  const avgTargetPrice = (longTermData?.recommendations || []).reduce((sum, rec) =>
    sum + (selectedTimeframe === '3y' ? rec.target_price_3y : rec.target_price_5y), 0
  ) / (longTermData?.recommendations?.length || 1) || 0

  const avgCurrentPrice = (longTermData?.recommendations || []).reduce((sum, rec) =>
    sum + rec.current_price, 0
  ) / (longTermData?.recommendations?.length || 1) || 0

  const avgUpside = calculateUpside(avgCurrentPrice, avgTargetPrice)

  return (
    <Layout title="Long-term Investment Analysis" subtitle="3-5 Year Investment Horizon & DCF Modeling">
      <div className="space-y-6">
        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <MetricCard
            title="Portfolio Recommendations"
            value={longTermData?.recommendations?.length || 0}
            icon={<ChartBarIcon className="w-5 h-5" />}
            loading={loading}
            
            
          />
          <MetricCard
            title="Average Upside"
            value={avgUpside}
            icon={<ArrowTrendingUpIcon className="w-5 h-5" />}
            loading={loading}
            
            
            
          />
          <MetricCard
            title="Market Sentiment"
            value={longTermData?.market_outlook.overall_sentiment || 'Unknown'}
            icon={<SparklesIcon className="w-5 h-5" />}
            loading={loading}
            
            
          />
          <MetricCard
            title="ESG Average"
            value={(longTermData?.recommendations || []).reduce((sum, rec) => sum + rec.esg_score, 0) / (longTermData?.recommendations?.length || 1) || 0}
            icon={<ShieldCheckIcon className="w-5 h-5" />}
            loading={loading}
            
            
            
          />
        </div>

        {/* Timeframe Selector */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-dark-200 sharp-card p-4 border border-dark-300"
        >
          <div className="flex items-center space-x-4">
            <span className="text-sm font-medium text-white">Investment Horizon:</span>
            <div className="flex space-x-2">
              <button
                onClick={() => setSelectedTimeframe('3y')}
                className={`px-4 py-2 sharp-button text-sm transition-colors ${
                  selectedTimeframe === '3y'
                    ? 'bg-accent-blue text-white'
                    : 'bg-dark-300 text-dark-500 hover:text-white'
                }`}
              >
                3 Years
              </button>
              <button
                onClick={() => setSelectedTimeframe('5y')}
                className={`px-4 py-2 sharp-button text-sm transition-colors ${
                  selectedTimeframe === '5y'
                    ? 'bg-accent-blue text-white'
                    : 'bg-dark-300 text-dark-500 hover:text-white'
                }`}
              >
                5 Years
              </button>
            </div>
          </div>
        </motion.div>

        {/* Recommendations Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {longTermData?.recommendations.map((rec, index) => {
            const targetPrice = selectedTimeframe === '3y' ? rec.target_price_3y : rec.target_price_5y
            const upside = calculateUpside(rec.current_price, targetPrice)

            return (
              <motion.div
                key={rec.symbol}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 + index * 0.1 }}
                className="bg-dark-200 sharp-card p-6 border border-dark-300 hover:border-accent-blue/50 transition-colors"
              >
                {/* Header */}
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-white">{rec.symbol}</h3>
                    <p className="text-sm text-dark-400">{rec.company_name}</p>
                  </div>
                  <span className={`px-3 py-1 sharp-card text-xs font-medium ${getRecommendationColor(rec.recommendation)}`}>
                    {rec.recommendation}
                  </span>
                </div>

                {/* Price Analysis */}
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="text-xs text-dark-500">Current Price</p>
                    <p className="text-lg font-semibold text-white">${rec.current_price.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-dark-500">{selectedTimeframe.toUpperCase()} Target</p>
                    <p className="text-lg font-semibold text-trading-profit">${targetPrice.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-dark-500">DCF Valuation</p>
                    <p className="text-sm font-medium text-accent-blue">${rec.dcf_valuation.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-dark-500">Upside Potential</p>
                    <p className={`text-sm font-medium ${upside > 0 ? 'text-trading-profit' : 'text-trading-loss'}`}>
                      {upside > 0 ? '+' : ''}{upside.toFixed(1)}%
                    </p>
                  </div>
                </div>

                {/* ESG & Sector */}
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="text-xs text-dark-500">ESG Score</p>
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-medium text-white">{rec.esg_score.toFixed(1)}/10</span>
                      <div className="flex-1 bg-dark-300 h-1.5 sharp-card overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${rec.esg_score * 10}%` }}
                          transition={{ delay: 0.5 + index * 0.1, duration: 0.8 }}
                          className="h-full bg-gradient-to-r from-accent-blue to-trading-profit"
                        />
                      </div>
                    </div>
                  </div>
                  <div>
                    <p className="text-xs text-dark-500">Sector Outlook</p>
                    <p className={`text-sm font-medium ${getSectorOutlookColor(rec.sector_outlook)}`}>
                      {rec.sector_outlook}
                    </p>
                  </div>
                </div>

                {/* Confidence */}
                <div className="mb-4">
                  <div className="flex items-center justify-between text-xs text-dark-500 mb-1">
                    <span>Analysis Confidence</span>
                    <span>{(rec.confidence * 100).toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-dark-300 h-2 sharp-card overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${rec.confidence * 100}%` }}
                      transition={{ delay: 0.6 + index * 0.1, duration: 0.8 }}
                      className="h-full bg-gradient-to-r from-accent-blue to-accent-purple"
                    />
                  </div>
                </div>

                {/* Catalysts */}
                <div className="mb-4">
                  <h4 className="text-sm font-medium text-trading-profit mb-2 flex items-center">
                    <ArrowTrendingUpIcon className="w-4 h-4 mr-1" />
                    Key Catalysts
                  </h4>
                  <div className="space-y-1">
                    {rec.key_catalysts.slice(0, 3).map((catalyst, i) => (
                      <div key={i} className="flex items-center text-xs text-dark-400">
                        <div className="w-1 h-1 bg-trading-profit mr-2 flex-shrink-0" />
                        {catalyst}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Risks */}
                <div>
                  <h4 className="text-sm font-medium text-trading-loss mb-2 flex items-center">
                    <ExclamationTriangleIcon className="w-4 h-4 mr-1" />
                    Key Risks
                  </h4>
                  <div className="space-y-1">
                    {rec.risks.slice(0, 2).map((risk, i) => (
                      <div key={i} className="flex items-center text-xs text-dark-400">
                        <div className="w-1 h-1 bg-trading-loss mr-2 flex-shrink-0" />
                        {risk}
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )
          })}
        </div>

        {/* Market Outlook */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="bg-dark-200 sharp-card p-6 border border-dark-300"
        >
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <ClockIcon className="w-5 h-5 mr-2 text-accent-blue" />
            Long-term Market Outlook
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div>
              <h4 className="text-sm font-medium text-accent-blue mb-2">Sector Rotations</h4>
              <div className="space-y-1">
                {longTermData?.market_outlook.sector_rotations.map((sector, i) => (
                  <div key={i} className="flex items-center text-sm text-dark-400">
                    <div className="w-1.5 h-1.5 bg-trading-profit mr-2 flex-shrink-0" />
                    {sector}
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h4 className="text-sm font-medium text-accent-blue mb-2">Macro Trends</h4>
              <div className="space-y-1">
                {longTermData?.market_outlook.macro_trends.map((trend, i) => (
                  <div key={i} className="flex items-center text-sm text-dark-400">
                    <div className="w-1.5 h-1.5 bg-accent-blue mr-2 flex-shrink-0" />
                    {trend}
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h4 className="text-sm font-medium text-trading-loss mb-2">Risk Factors</h4>
              <div className="space-y-1">
                {longTermData?.market_outlook.risk_factors.map((risk, i) => (
                  <div key={i} className="flex items-center text-sm text-dark-400">
                    <div className="w-1.5 h-1.5 bg-trading-loss mr-2 flex-shrink-0" />
                    {risk}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>

        {/* DCF Methodology */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="bg-dark-200 sharp-card p-6 border border-dark-300"
        >
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <CurrencyDollarIcon className="w-5 h-5 mr-2 text-accent-blue" />
            DCF Valuation Methodology
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-accent-blue">Free Cash Flow Projections</h4>
              <p className="text-sm text-dark-400">
                5-year detailed FCF projections incorporating revenue growth, margin expansion,
                and capital allocation efficiency.
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-accent-blue">Discount Rate Calculation</h4>
              <p className="text-sm text-dark-400">
                WACC calculation using current risk-free rates, equity risk premiums,
                and company-specific beta adjustments.
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-accent-blue">Terminal Value</h4>
              <p className="text-sm text-dark-400">
                Conservative terminal growth rates based on GDP growth expectations
                and industry maturity analysis.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </Layout>
  )
}