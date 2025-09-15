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
  ArrowTrendingUpIcon as TrendingUpIcon,
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
  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
  const { socket, isConnected } = useSocket(API_URL);

  // Initial data fetch
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const response = await fetch(`${API_URL}/api/portfolio`);
        if (!response.ok) throw new Error('Failed to fetch portfolio data');
        const data = await response.json();
        setPortfolioData(data);
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
            <MetricCard title="Total P&L" value={totalPnL} change={totalPnLPercent} changeType={totalPnL > 0 ? 'increase' : 'decrease'} formatValue={(v) => `$${Number(v).toLocaleString()}`} loading={loading} icon={totalPnL > 0 ? <TrendingUpIcon className="w-5 h-5" /> : <TrendingDownIcon className="w-5 h-5" />} />
            <MetricCard title="Positions" value={portfolioData.holdings.length} loading={loading} icon={<ChartPieIcon className="w-5 h-5" />} />
            <MetricCard title="Socket Status" value={isConnected ? 'Connected' : 'Disconnected'} color={isConnected ? 'green' : 'red'} loading={loading} />
        </motion.div>

        {/* Candlestick Chart */}
        <motion.div variants={itemVariants} className="bg-dark-200 sharp-card p-6 border border-dark-300">
            <h3 className="text-lg font-semibold text-white mb-4">Performance History</h3>
            <CandlestickChart data={portfolioData.performanceHistory} />
        </motion.div>

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
