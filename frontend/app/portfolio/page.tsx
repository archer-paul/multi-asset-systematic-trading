'use client'

import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import Layout from '@/components/Layout/Layout'
import MetricCard from '@/components/MetricCard'
import { createChart, IChartApi, ISeriesApi } from 'lightweight-charts'
import useSocket from '../../hooks/useSocket' // Adjusted path
import toast, { Toaster } from 'react-hot-toast'
import {
  BriefcaseIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon as TrendingDownIcon,
  CurrencyDollarIcon,
  ChartPieIcon,
} from '@heroicons/react/24/outline'

// Candlestick Chart Component
const CandlestickChart = ({ data }: { data: any }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  useEffect(() => {
    if (chartContainerRef.current) {
      chartRef.current = createChart(chartContainerRef.current, {
        width: chartContainerRef.current.clientWidth,
        height: 300,
        layout: { background: { color: '#1a202c' }, textColor: '#cbd5e0' },
        grid: { vertLines: { color: '#2d3748' }, horzLines: { color: '#2d3748' } },
      });

      seriesRef.current = chartRef.current.addCandlestickSeries({
        upColor: '#10b981',
        downColor: '#ef4444',
        borderDownColor: '#ef4444',
        borderUpColor: '#10b981',
        wickDownColor: '#ef4444',
        wickUpColor: '#10b981',
      });

      seriesRef.current.setData(data);

      const handleResize = () => {
        if (chartRef.current && chartContainerRef.current) {
          chartRef.current.resize(chartContainerRef.current.clientWidth, 300);
        }
      };

      window.addEventListener('resize', handleResize);
      return () => window.removeEventListener('resize', handleResize);
    }
  }, []);

  useEffect(() => {
    if (seriesRef.current && data) {
      seriesRef.current.setData(data);
    }
  }, [data]);

  return <div ref={chartContainerRef} />;
};

export default function PortfolioPage() {
  const [loading, setLoading] = useState(true);
  const [portfolioData, setPortfolioData] = useState<{ holdings: any[], sectorAllocation: any[], performanceHistory: any[] }>({ holdings: [], sectorAllocation: [], performanceHistory: [] });
  const [advancedDecisions, setAdvancedDecisions] = useState<any>(null);
  const [portfolioAnalysis, setPortfolioAnalysis] = useState<any>(null);
  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
  const { socket, isConnected } = useSocket(API_URL);

  // Initial data fetch
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        // Load basic portfolio data
        const response = await fetch(`${API_URL}/api/portfolio`);
        if (!response.ok) throw new Error('Failed to fetch portfolio data');
        const data = await response.json();
        setPortfolioData(data);

        // Try to load advanced decisions
        try {
          const advancedResponse = await fetch(`${API_URL}/api/advanced/portfolio/decisions`);
          if (advancedResponse.ok) {
            const advancedData = await advancedResponse.json();
            setAdvancedDecisions(advancedData);
            toast.success('Advanced portfolio data loaded!');
          }
        } catch (advError) {
          console.warn('Advanced portfolio API not available:', advError);
        }

        // Try to load portfolio analysis
        try {
          const analysisResponse = await fetch(`${API_URL}/api/advanced/portfolio/analysis`);
          if (analysisResponse.ok) {
            const analysisData = await analysisResponse.json();
            setPortfolioAnalysis(analysisData);
          }
        } catch (anaError) {
          console.warn('Portfolio analysis API not available:', anaError);
        }

        toast.success('Portfolio data loaded!');
      } catch (error) {
        console.warn('Portfolio API not available, using mock data:', error);
        // Mock data for portfolio - structure matches what code expects
        const mockData = {
          holdings: [
            { symbol: 'AAPL', name: 'Apple Inc.', shares: 45, value: 8208.45, pnl: 323.1, weight: 0.065, sector: 'Technology' },
            { symbol: 'MSFT', name: 'Microsoft Corp.', shares: 32, value: 11416.96, pnl: 468.16, weight: 0.091, sector: 'Technology' },
            { symbol: 'GOOGL', name: 'Alphabet Inc.', shares: 18, value: 2415.96, pnl: 103.86, weight: 0.019, sector: 'Technology' },
            { symbol: 'TSLA', name: 'Tesla Inc.', shares: 25, value: 5972.75, pnl: -169, weight: 0.047, sector: 'Consumer Discretionary' },
            { symbol: 'NVDA', name: 'NVIDIA Corp.', shares: 15, value: 7339.8, pnl: 488.1, weight: 0.058, sector: 'Technology' }
          ],
          sectorAllocation: [
            { sector: 'Technology', value: 29380.17, weight: 0.233 },
            { sector: 'Healthcare', value: 18642.33, weight: 0.148 },
            { sector: 'Finance', value: 15234.87, weight: 0.121 },
            { sector: 'Consumer Discretionary', value: 12456.78, weight: 0.099 },
            { sector: 'Energy', value: 9876.54, weight: 0.078 },
            { sector: 'Others', value: 40256.63, weight: 0.321 }
          ],
          performanceHistory: Array.from({ length: 180 }, (_, i) => {
            const baseValue = 100000 + i * 150;
            const randomVariation = (Math.random() - 0.5) * 2000;
            const open = baseValue + randomVariation;
            const close = open + (Math.random() - 0.5) * 1000;
            const high = Math.max(open, close) + Math.random() * 500;
            const low = Math.min(open, close) - Math.random() * 500;

            return {
              time: Math.floor((Date.now() - (179 - i) * 24 * 60 * 60 * 1000) / 1000),
              open: Math.round(open),
              high: Math.round(high),
              low: Math.round(low),
              close: Math.round(close)
            };
          })
        };
        setPortfolioData(mockData);
        toast.success('Portfolio demo data loaded!');
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, [API_URL]);

  // WebSocket listeners
  useEffect(() => {
    if (socket && isConnected) {
      socket.on('portfolio_update', (data) => {
        setPortfolioData(data);
        toast('Portfolio updated in real-time!', { icon: '🚀' });
      });

      socket.on('new_trade', (trade) => {
        toast.success(`New Trade: ${trade.type.toUpperCase()} ${trade.shares} ${trade.symbol} @ ${trade.price}`);
      });

      return () => {
        socket.off('portfolio_update');
        socket.off('new_trade');
      };
    }
  }, [socket, isConnected]);

  const totalValue = portfolioData.holdings.reduce((sum, h) => sum + h.value, 0);
  const totalPnL = portfolioData.holdings.reduce((sum, h) => sum + h.pnl, 0);
  const totalPnLPercent = totalValue > 0 ? (totalPnL / (totalValue - totalPnL)) * 100 : 0;

  const containerVariants = { hidden: { opacity: 0 }, visible: { opacity: 1, transition: { staggerChildren: 0.1 } } };
  const itemVariants = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0, transition: { duration: 0.5 } } };

  return (
    <Layout title="Portfolio" subtitle="Holdings & Performance Analytics">
      <Toaster position="top-right" toastOptions={{ style: { background: '#2d3748', color: '#fff' } }} />
      <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-6">
        {/* Overview Metrics */}
        <motion.div variants={itemVariants} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard title="Total Value" value={totalValue} formatValue={(v) => `$${Number(v).toLocaleString()}`} loading={loading} icon={<CurrencyDollarIcon className="w-5 h-5" />} />
            <MetricCard title="Total P&L" value={totalPnL} change={totalPnLPercent} changeType={totalPnL > 0 ? 'increase' : 'decrease'} formatValue={(v) => `$${Number(v).toLocaleString()}`} loading={loading} icon={totalPnL > 0 ? <ArrowTrendingUpIcon className="w-5 h-5" /> : <TrendingDownIcon className="w-5 h-5" />} />
            <MetricCard title="Positions" value={portfolioData.holdings.length} loading={loading} icon={<ChartPieIcon className="w-5 h-5" />} />
            <MetricCard title="Socket Status" value={isConnected ? 'Connected' : 'Disconnected'} color={isConnected ? 'green' : 'red'} loading={loading} />
        </motion.div>

        {/* Candlestick Chart */}
        <motion.div variants={itemVariants} className="bg-dark-200 sharp-card p-6 border border-dark-300">
            <h3 className="text-lg font-semibold text-white mb-4">Performance History</h3>
            <CandlestickChart data={portfolioData.performanceHistory} />
        </motion.div>

        {/* Advanced Portfolio Decisions */}
        {advancedDecisions && advancedDecisions.dashboard_data && (
          <motion.div variants={itemVariants} className="bg-dark-200 sharp-card p-6 border border-dark-300">
            <h3 className="text-lg font-semibold text-white mb-4">🧠 Advanced Portfolio Decisions</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Decision Summary */}
              <div>
                <h4 className="text-sm font-medium text-dark-500 mb-3">Decision Summary</h4>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-white">Total Decisions:</span>
                    <span className="text-white">{advancedDecisions.decision_summary.total_decisions || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white">Buy Signals:</span>
                    <span className="text-trading-profit">{advancedDecisions.decision_summary.buy_decisions || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white">Sell Signals:</span>
                    <span className="text-trading-loss">{advancedDecisions.decision_summary.sell_decisions || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white">Avg Conviction:</span>
                    <span className="text-white">{(advancedDecisions.decision_summary.avg_conviction * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>

              {/* Risk Analysis */}
              {advancedDecisions.dashboard_data.risk_analysis && (
                <div>
                  <h4 className="text-sm font-medium text-dark-500 mb-3">Risk Analysis</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-white">Total Positions:</span>
                      <span className="text-white">{advancedDecisions.dashboard_data.risk_analysis.total_positions || 0}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white">High Conviction:</span>
                      <span className="text-white">{advancedDecisions.dashboard_data.risk_analysis.high_conviction_count || 0}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Top Recommendations */}
            {advancedDecisions.dashboard_data.portfolio_recommendations && advancedDecisions.dashboard_data.portfolio_recommendations.length > 0 && (
              <div className="mt-6">
                <h4 className="text-sm font-medium text-dark-500 mb-3">Top Recommendations</h4>
                <div className="space-y-3">
                  {advancedDecisions.dashboard_data.portfolio_recommendations.slice(0, 5).map((rec: any, index: number) => (
                    <div key={`${rec.symbol}-${index}`} className="flex items-center justify-between p-3 bg-dark-300/30 rounded">
                      <div className="flex items-center space-x-3">
                        <span className="font-mono text-white font-medium">{rec.symbol}</span>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          rec.action === 'BUY' ? 'bg-green-500/20 text-green-400' :
                          rec.action === 'SELL' ? 'bg-red-500/20 text-red-400' :
                          'bg-yellow-500/20 text-yellow-400'
                        }`}>
                          {rec.action}
                        </span>
                      </div>
                      <div className="text-right">
                        <div className="text-white text-sm">{rec.target_weight}</div>
                        <div className="text-dark-500 text-xs">{rec.conviction} conviction</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}

        {/* Portfolio Analysis */}
        {portfolioAnalysis && portfolioAnalysis.analysis && (
          <motion.div variants={itemVariants} className="bg-dark-200 sharp-card p-6 border border-dark-300">
            <h3 className="text-lg font-semibold text-white mb-4">📊 Advanced Portfolio Analysis</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-white">{portfolioAnalysis.analysis.market_regime || 'Unknown'}</div>
                <div className="text-sm text-dark-500">Market Regime</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-white">{(portfolioAnalysis.analysis.risk_analysis?.sharpe_ratio || 0).toFixed(2)}</div>
                <div className="text-sm text-dark-500">Sharpe Ratio</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-white">{(portfolioAnalysis.analysis.risk_analysis?.portfolio_volatility || 0).toFixed(1)}%</div>
                <div className="text-sm text-dark-500">Volatility</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-white">{(portfolioAnalysis.analysis.risk_analysis?.win_rate_pct || 0).toFixed(1)}%</div>
                <div className="text-sm text-dark-500">Win Rate</div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Holdings Table */}
        <motion.div variants={itemVariants}>
          <h2 className="text-xl font-semibold text-white mb-4">Current Holdings</h2>
          <div className="bg-dark-200 sharp-card border border-dark-300 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                {/* Table Head */}
                <thead className="bg-dark-300/50"><tr>
                    <th className="text-left py-4 px-6 text-dark-500 font-medium">Symbol</th>
                    <th className="text-right py-4 px-6 text-dark-500 font-medium">Shares</th>
                    <th className="text-right py-4 px-6 text-dark-500 font-medium">Market Value</th>
                    <th className="text-right py-4 px-6 text-dark-500 font-medium">P&L</th>
                    <th className="text-right py-4 px-6 text-dark-500 font-medium">Weight</th>
                </tr></thead>
                {/* Table Body */}
                <tbody>
                  {portfolioData.holdings.map((holding, index) => (
                    <motion.tr key={holding.symbol} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: index * 0.1 }} className="border-b border-dark-400/30">
                      <td className="py-4 px-6 font-mono text-white">{holding.symbol}</td>
                      <td className="py-4 px-6 text-right font-mono text-white">{holding.shares.toLocaleString()}</td>
                      <td className="py-4 px-6 text-right font-mono text-white">${holding.value.toLocaleString()}</td>
                      <td className={`py-4 px-6 text-right font-mono ${holding.pnl > 0 ? 'text-trading-profit' : 'text-trading-loss'}`}>{holding.pnl > 0 ? '+' : ''}${holding.pnl.toLocaleString()}</td>
                      <td className="py-4 px-6 text-right font-mono text-dark-500">{(holding.weight * 100).toFixed(1)}%</td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </Layout>
  )
}
