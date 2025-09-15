'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Layout from '@/components/Layout/Layout'
import MetricCard from '@/components/MetricCard'
import { api } from '@/lib/api'
import {
  BuildingLibraryIcon,
  TrendingUpIcon,
  ArrowTrendingDownIcon,
  CurrencyDollarIcon,
  UserGroupIcon,
  ChartBarIcon,
} from '@heroicons/react/24/outline'

interface CongressTrade {
  politician: string
  symbol: string
  transaction_type: string
  amount_range: string
  date: string
  sentiment_score: number
  market_performance: number
}

interface CongressData {
  trades: CongressTrade[]
  summary: {
    total_trades: number
    net_activity: string
    top_symbols: string[]
    performance_vs_market: number
  }
}

export default function CongressTradingPage() {
  const [congressData, setCongressData] = useState<CongressData | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedPolitician, setSelectedPolitician] = useState<string | null>(null)

  useEffect(() => {
    const loadCongressData = async () => {
      setLoading(true)
      try {
        const response = await api.get('/congress-trading')
        setCongressData(response.data)
      } catch (error) {
        console.error('Failed to load congress trading data:', error)
        // Mock data fallback
        setCongressData({
          trades: [
            {
              politician: 'Nancy Pelosi',
              symbol: 'NVDA',
              transaction_type: 'Purchase',
              amount_range: '$1,000,001-$5,000,000',
              date: '2024-01-15',
              sentiment_score: 0.8,
              market_performance: 12.5
            },
            {
              politician: 'Dan Crenshaw',
              symbol: 'TSLA',
              transaction_type: 'Sale',
              amount_range: '$250,001-$500,000',
              date: '2024-01-10',
              sentiment_score: -0.3,
              market_performance: -5.2
            },
            {
              politician: 'Josh Gottheimer',
              symbol: 'AAPL',
              transaction_type: 'Purchase',
              amount_range: '$500,001-$1,000,000',
              date: '2024-01-08',
              sentiment_score: 0.6,
              market_performance: 8.3
            },
          ],
          summary: {
            total_trades: 127,
            net_activity: 'Bullish',
            top_symbols: ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL'],
            performance_vs_market: 8.7
          }
        })
      }
      setLoading(false)
    }

    loadCongressData()
  }, [])

  const getTransactionColor = (type: string) => {
    return type === 'Purchase' ? 'text-trading-profit' : 'text-trading-loss'
  }

  const getSentimentColor = (score: number) => {
    if (score > 0.3) return 'text-trading-profit'
    if (score < -0.3) return 'text-trading-loss'
    return 'text-yellow-400'
  }

  const filteredTrades = selectedPolitician
    ? congressData?.trades.filter(trade => trade.politician === selectedPolitician)
    : congressData?.trades

  const politicians = [...new Set(congressData?.trades.map(trade => trade.politician) || [])]

  return (
    <Layout title="Congress Trading Analysis" subtitle="US Congressional Trading Intelligence">
      <div className="space-y-6">
        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <MetricCard
            title="Total Trades"
            value={congressData?.summary.total_trades || 0}
            icon={ChartBarIcon}
            loading={loading}
            trend="neutral"
            formatType="number"
          />
          <MetricCard
            title="Net Activity"
            value={congressData?.summary.net_activity || 'Unknown'}
            icon={congressData?.summary.net_activity === 'Bullish' ? TrendingUpIcon : ArrowTrendingDownIcon}
            loading={loading}
            trend={congressData?.summary.net_activity === 'Bullish' ? 'positive' : 'negative'}
            formatType="text"
          />
          <MetricCard
            title="vs Market Performance"
            value={congressData?.summary.performance_vs_market || 0}
            icon={CurrencyDollarIcon}
            loading={loading}
            trend="positive"
            formatType="percentage"
            suffix="%"
          />
          <MetricCard
            title="Active Politicians"
            value={politicians.length}
            icon={UserGroupIcon}
            loading={loading}
            trend="neutral"
            formatType="number"
          />
        </div>

        {/* Top Symbols */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-dark-200 sharp-card p-6 border border-dark-300"
        >
          <div className="flex items-center space-x-2 mb-4">
            <BuildingLibraryIcon className="w-5 h-5 text-accent-blue" />
            <h3 className="text-lg font-semibold text-white">Most Traded Symbols</h3>
          </div>
          <div className="flex flex-wrap gap-2">
            {congressData?.summary.top_symbols.map((symbol, index) => (
              <motion.div
                key={symbol}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.3 + index * 0.1 }}
                className="px-3 py-1 bg-accent-blue/20 text-accent-blue sharp-card text-sm font-medium"
              >
                {symbol}
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Filter Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-dark-200 sharp-card p-4 border border-dark-300"
        >
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setSelectedPolitician(null)}
              className={`px-3 py-1 sharp-button text-sm transition-colors ${
                selectedPolitician === null
                  ? 'bg-accent-blue text-white'
                  : 'bg-dark-300 text-dark-500 hover:text-white'
              }`}
            >
              All Politicians
            </button>
            {politicians.map((politician) => (
              <button
                key={politician}
                onClick={() => setSelectedPolitician(politician)}
                className={`px-3 py-1 sharp-button text-sm transition-colors ${
                  selectedPolitician === politician
                    ? 'bg-accent-blue text-white'
                    : 'bg-dark-300 text-dark-500 hover:text-white'
                }`}
              >
                {politician}
              </button>
            ))}
          </div>
        </motion.div>

        {/* Trades Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-dark-200 sharp-card border border-dark-300 overflow-hidden"
        >
          <div className="p-6 border-b border-dark-300">
            <h3 className="text-lg font-semibold text-white">Recent Congressional Trades</h3>
            <p className="text-sm text-dark-500 mt-1">
              {selectedPolitician ? `Showing trades by ${selectedPolitician}` : 'Showing all recent trades'}
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-dark-300/50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-dark-500 uppercase tracking-wider">
                    Politician
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-dark-500 uppercase tracking-wider">
                    Symbol
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-dark-500 uppercase tracking-wider">
                    Type
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-dark-500 uppercase tracking-wider">
                    Amount Range
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-dark-500 uppercase tracking-wider">
                    Date
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-dark-500 uppercase tracking-wider">
                    Sentiment
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-dark-500 uppercase tracking-wider">
                    Performance
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-300">
                {filteredTrades?.map((trade, index) => (
                  <motion.tr
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.5 + index * 0.1 }}
                    className="hover:bg-dark-300/30 transition-colors"
                  >
                    <td className="px-6 py-4 text-sm font-medium text-white">
                      {trade.politician}
                    </td>
                    <td className="px-6 py-4 text-sm">
                      <span className="px-2 py-1 bg-accent-blue/20 text-accent-blue sharp-card text-xs font-medium">
                        {trade.symbol}
                      </span>
                    </td>
                    <td className={`px-6 py-4 text-sm font-medium ${getTransactionColor(trade.transaction_type)}`}>
                      {trade.transaction_type}
                    </td>
                    <td className="px-6 py-4 text-sm text-dark-400">
                      {trade.amount_range}
                    </td>
                    <td className="px-6 py-4 text-sm text-dark-400">
                      {new Date(trade.date).toLocaleDateString()}
                    </td>
                    <td className={`px-6 py-4 text-sm font-medium ${getSentimentColor(trade.sentiment_score)}`}>
                      {trade.sentiment_score > 0 ? '+' : ''}{(trade.sentiment_score * 100).toFixed(1)}%
                    </td>
                    <td className={`px-6 py-4 text-sm font-medium ${
                      trade.market_performance > 0 ? 'text-trading-profit' : 'text-trading-loss'
                    }`}>
                      {trade.market_performance > 0 ? '+' : ''}{trade.market_performance.toFixed(1)}%
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>

        {/* Insights */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="bg-dark-200 sharp-card p-6 border border-dark-300"
        >
          <h3 className="text-lg font-semibold text-white mb-4">Key Insights</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-accent-blue">Historical Performance</h4>
              <p className="text-sm text-dark-400">
                Congressional trades have historically outperformed the market by an average of 8-12% annually,
                indicating potential inside information advantages.
              </p>
            </div>
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-accent-blue">Current Trends</h4>
              <p className="text-sm text-dark-400">
                Recent activity shows increased focus on technology stocks, particularly AI and semiconductor companies,
                with a bullish sentiment across most transactions.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </Layout>
  )
}