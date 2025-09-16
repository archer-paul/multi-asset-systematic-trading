'use client'

import { useState, useEffect } from 'react';
import Layout from '@/components/Layout/Layout';
import MetricCard from '@/components/MetricCard';
import { motion } from 'framer-motion';
import {
  GlobeAltIcon,
  NewspaperIcon,
  ChatBubbleLeftRightIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon
} from '@heroicons/react/24/outline';
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
  PieChart,
  Pie,
  Cell
} from 'recharts';
import type { MacroSentiment } from '@/types';

// Mock data for comprehensive sentiment analysis
const sentimentTrendData = [
  { time: '09:00', overall: 0.68, news: 0.72, social: 0.64, market: 0.71 },
  { time: '10:00', overall: 0.71, news: 0.74, social: 0.68, market: 0.73 },
  { time: '11:00', overall: 0.65, news: 0.69, social: 0.61, market: 0.67 },
  { time: '12:00', overall: 0.73, news: 0.76, social: 0.70, market: 0.75 },
  { time: '13:00', overall: 0.69, news: 0.71, social: 0.67, market: 0.72 },
  { time: '14:00', overall: 0.66, news: 0.68, social: 0.64, market: 0.69 },
  { time: '15:00', overall: 0.75, news: 0.78, social: 0.72, market: 0.77 },
  { time: '16:00', overall: 0.72, news: 0.75, social: 0.69, market: 0.74 },
];

const sectorSentimentData = [
  { sector: 'Technology', sentiment: 0.78, change: 0.05, volume: 450000, color: '#3b82f6' },
  { sector: 'Healthcare', sentiment: 0.72, change: 0.02, volume: 320000, color: '#10b981' },
  { sector: 'Financial', sentiment: 0.45, change: -0.08, volume: 580000, color: '#f59e0b' },
  { sector: 'Energy', sentiment: 0.38, change: -0.12, volume: 280000, color: '#ef4444' },
  { sector: 'Consumer', sentiment: 0.65, change: 0.03, volume: 370000, color: '#8b5cf6' },
  { sector: 'Industrial', sentiment: 0.58, change: -0.02, volume: 240000, color: '#06b6d4' },
];

const newsSentimentData = [
  { source: 'Reuters', positive: 45, neutral: 35, negative: 20, total: 120 },
  { source: 'Bloomberg', positive: 52, neutral: 28, negative: 20, total: 98 },
  { source: 'CNBC', positive: 38, neutral: 42, negative: 20, total: 85 },
  { source: 'Financial Times', positive: 48, neutral: 32, negative: 20, total: 76 },
  { source: 'MarketWatch', positive: 41, neutral: 39, negative: 20, total: 92 },
];

const socialPlatformData = [
  { name: 'Twitter', value: 0.72, posts: 15420, color: '#1DA1F2' },
  { name: 'Reddit', value: 0.61, posts: 8930, color: '#FF4500' },
  { name: 'StockTwits', value: 0.68, posts: 6750, color: '#40E0D0' },
  { name: 'Discord', value: 0.75, posts: 4200, color: '#5865F2' },
];

const marketMoversData = [
  { symbol: 'AAPL', sentiment: 0.82, price: 185.24, change: 2.34, volume: 'High' },
  { symbol: 'MSFT', sentiment: 0.78, price: 412.89, change: 1.89, volume: 'High' },
  { symbol: 'GOOGL', sentiment: 0.71, price: 142.56, change: -0.45, volume: 'Medium' },
  { symbol: 'TSLA', sentiment: 0.35, price: 248.73, change: -3.21, volume: 'Very High' },
  { symbol: 'NVDA', sentiment: 0.89, price: 875.28, change: 4.56, volume: 'Extreme' },
  { symbol: 'AMD', sentiment: 0.64, price: 152.41, change: 1.23, volume: 'High' },
];

const fearGreedData = [
  { category: 'Volatility', value: 25, status: 'Fear' },
  { category: 'Market Momentum', value: 72, status: 'Greed' },
  { category: 'Stock Price Strength', value: 68, status: 'Greed' },
  { category: 'Junk Bond Demand', value: 45, status: 'Neutral' },
  { category: 'Safe Haven Demand', value: 38, status: 'Greed' },
];

export default function SentimentAnalysisPage() {
  const [sentimentData, setSentimentData] = useState<any>(null);
  const [macroSentiment, setMacroSentiment] = useState<MacroSentiment | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1D');

  useEffect(() => {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
    const loadData = async () => {
      setLoading(true);
      try {
        // Load main sentiment data
        const response = await fetch(`${API_URL}/api/sentiment-summary`);
        if (!response.ok) throw new Error('Failed to fetch sentiment data');
        const data = await response.json();
        setSentimentData(data);

        // Load macro sentiment data
        const macroResponse = await fetch(`${API_URL}/api/macro-sentiment`);
        if (macroResponse.ok) {
          const macroData = await macroResponse.json();
          setMacroSentiment(macroData);
        }
      } catch (error) {
        console.warn('Sentiment API not available, using mock data:', error);
        // Mock data for sentiment analysis
        const mockData = {
          macro_summary: {
            overall_risk_score: 0.35,
            geopolitical_tension_score: 0.42,
            economic_uncertainty_score: 0.28,
            most_impacted_sectors: ['Technology', 'Energy', 'Finance']
          },
          top_news_sentiment: [
            { source: 'Reuters', title: 'Fed signals potential rate cuts ahead', sentiment_score: 0.68, published_at: '2024-01-15T10:30:00Z', impact_score: 0.82 },
            { source: 'Bloomberg', title: 'Tech earnings beat expectations across the board', sentiment_score: 0.82, published_at: '2024-01-15T09:15:00Z', impact_score: 0.91 },
            { source: 'CNBC', title: 'Energy sector faces new regulatory challenges', sentiment_score: 0.25, published_at: '2024-01-15T08:45:00Z', impact_score: 0.67 },
            { source: 'Financial Times', title: 'Market volatility expected to continue amid uncertainty', sentiment_score: 0.35, published_at: '2024-01-15T07:30:00Z', impact_score: 0.74 },
            { source: 'MarketWatch', title: 'AI stocks surge on breakthrough announcements', sentiment_score: 0.89, published_at: '2024-01-15T06:15:00Z', impact_score: 0.95 }
          ],
          social_media_sentiment: {
            twitter: {
              overall_score: 0.72,
              trending_topics: [
                { topic: '#AI', sentiment: 0.85, mentions: 15420, engagement: 'Very High' },
                { topic: '#FedPolicy', sentiment: 0.45, mentions: 8930, engagement: 'High' },
                { topic: '#TechEarnings', sentiment: 0.78, mentions: 6750, engagement: 'High' },
                { topic: '#CleanEnergy', sentiment: 0.67, mentions: 4200, engagement: 'Medium' }
              ]
            },
            reddit: {
              overall_score: 0.61,
              top_discussions: [
                { title: 'Market predictions for Q2 2024', sentiment: 0.55, upvotes: 1247, comments: 342 },
                { title: 'Deep dive into tech stock valuations', sentiment: 0.73, upvotes: 892, comments: 156 },
                { title: 'Energy sector concerns and opportunities', sentiment: 0.28, upvotes: 567, comments: 89 },
                { title: 'Fed policy impact on growth stocks', sentiment: 0.42, upvotes: 433, comments: 78 }
              ]
            }
          }
        };
        setSentimentData(mockData);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, []);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5 }
    }
  };

  if (loading || !sentimentData) {
    return <Layout title="Sentiment Analysis"><div>Loading...</div></Layout>;
  }

  const { macro_summary, top_news_sentiment, social_media_sentiment } = sentimentData;

  return (
    <Layout title="Sentiment Analysis" subtitle="Market, News & Geopolitical Mood">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="space-y-6"
      >

        {/* Sentiment Overview Metrics */}
        <motion.div variants={itemVariants}>
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
            <ChartBarIcon className="w-6 h-6 mr-2 text-accent-blue" />
            Sentiment Overview
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              title="Overall Market Sentiment"
              value={72.5}
              change={3.2}
              changeType="increase"
              icon={<ArrowTrendingUpIcon className="w-5 h-5" />}
              formatValue={(val) => `${Number(val).toFixed(1)}%`}
              loading={loading}
              color="green"
            />

            <MetricCard
              title="News Sentiment Score"
              value={68.3}
              change={1.8}
              changeType="increase"
              icon={<NewspaperIcon className="w-5 h-5" />}
              formatValue={(val) => `${Number(val).toFixed(1)}%`}
              loading={loading}
              color="blue"
            />

            <MetricCard
              title="Social Media Buzz"
              value={71.2}
              change={4.1}
              changeType="increase"
              icon={<ChatBubbleLeftRightIcon className="w-5 h-5" />}
              formatValue={(val) => `${Number(val).toFixed(1)}%`}
              loading={loading}
              color="purple"
            />

            <MetricCard
              title="Fear & Greed Index"
              value={53}
              change={-2}
              changeType="decrease"
              icon={<ExclamationTriangleIcon className="w-5 h-5" />}
              formatValue={(val) => Number(val).toFixed(0)}
              loading={loading}
              color="yellow"
            />
          </div>
        </motion.div>

        {/* Sentiment Trend Chart */}
        <motion.div variants={itemVariants}>
          <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white flex items-center">
                <ArrowTrendingUpIcon className="w-5 h-5 mr-2 text-accent-blue" />
                Intraday Sentiment Trends
              </h3>

              <div className="flex space-x-2">
                {['1D', '5D', '1M'].map((period) => (
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
              <LineChart data={sentimentTrendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" opacity={0.3} />
                <XAxis dataKey="time" stroke="#718096" fontSize={12} />
                <YAxis stroke="#718096" fontSize={12} domain={[0, 1]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#2d3748', border: '1px solid #4a5568' }}
                  labelStyle={{ color: '#fff' }}
                />

                <Line type="monotone" dataKey="overall" stroke="#3b82f6" strokeWidth={3} name="Overall" />
                <Line type="monotone" dataKey="news" stroke="#10b981" strokeWidth={2} name="News" />
                <Line type="monotone" dataKey="social" stroke="#8b5cf6" strokeWidth={2} name="Social" />
                <Line type="monotone" dataKey="market" stroke="#f59e0b" strokeWidth={2} name="Market" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Sector Sentiment Analysis */}
        <motion.div variants={itemVariants}>
          <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <ChartBarIcon className="w-5 h-5 mr-2 text-accent-purple" />
              Sector Sentiment Breakdown
            </h3>

            <div className="space-y-4">
              {sectorSentimentData.map((sector, index) => (
                <div key={index} className="flex items-center justify-between p-4 bg-dark-300/50 sharp-card">
                  <div className="flex items-center space-x-4">
                    <div
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: sector.color }}
                    />
                    <div>
                      <div className="text-white font-medium">{sector.sector}</div>
                      <div className="text-xs text-dark-500">Volume: {sector.volume.toLocaleString()}</div>
                    </div>
                  </div>

                  <div className="flex items-center space-x-6">
                    <div className="text-center">
                      <div className={`text-lg font-bold ${
                        sector.sentiment > 0.6 ? 'text-trading-profit' :
                        sector.sentiment > 0.4 ? 'text-yellow-400' : 'text-trading-loss'
                      }`}>
                        {(sector.sentiment * 100).toFixed(0)}%
                      </div>
                      <div className="text-xs text-dark-500">Sentiment</div>
                    </div>

                    <div className="text-center">
                      <div className={`text-sm font-medium flex items-center ${
                        sector.change > 0 ? 'text-trading-profit' : 'text-trading-loss'
                      }`}>
                        {sector.change > 0 ? <ArrowTrendingUpIcon className="w-4 h-4 mr-1" /> : <ArrowTrendingDownIcon className="w-4 h-4 mr-1" />}
                        {sector.change > 0 ? '+' : ''}{(sector.change * 100).toFixed(1)}%
                      </div>
                      <div className="text-xs text-dark-500">24h Change</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* News Sentiment & Social Media Grid */}
        <motion.div variants={itemVariants} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* News Source Analysis */}
          <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <NewspaperIcon className="w-5 h-5 mr-2 text-accent-blue" />
              News Source Sentiment
            </h3>

            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={newsSentimentData} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" opacity={0.3} />
                <XAxis type="number" stroke="#718096" fontSize={12} />
                <YAxis dataKey="source" type="category" stroke="#718096" fontSize={10} width={80} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#2d3748', border: '1px solid #4a5568' }}
                />

                <Bar dataKey="positive" stackId="a" fill="#10b981" name="Positive" />
                <Bar dataKey="neutral" stackId="a" fill="#718096" name="Neutral" />
                <Bar dataKey="negative" stackId="a" fill="#ef4444" name="Negative" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Social Platform Distribution */}
          <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <ChatBubbleLeftRightIcon className="w-5 h-5 mr-2 text-accent-purple" />
              Social Media Sentiment
            </h3>

            <div className="space-y-4">
              {socialPlatformData.map((platform, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: platform.color }}
                    />
                    <span className="text-white font-medium">{platform.name}</span>
                    <span className="text-xs text-dark-500">{platform.posts.toLocaleString()} posts</span>
                  </div>
                  <div className={`text-sm font-bold ${
                    platform.value > 0.7 ? 'text-trading-profit' :
                    platform.value > 0.5 ? 'text-yellow-400' : 'text-trading-loss'
                  }`}>
                    {(platform.value * 100).toFixed(0)}%
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-6 pt-4 border-t border-dark-300">
              <ResponsiveContainer width="100%" height={120}>
                <PieChart>
                  <Pie
                    data={socialPlatformData}
                    cx="50%"
                    cy="50%"
                    outerRadius={50}
                    dataKey="posts"
                  >
                    {socialPlatformData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => [value.toLocaleString(), 'Posts']} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </motion.div>

        {/* Top Market Movers */}
        <motion.div variants={itemVariants}>
          <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <ArrowTrendingUpIcon className="w-5 h-5 mr-2 text-accent-yellow" />
              Sentiment-Driven Market Movers
            </h3>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-dark-300">
                    <th className="text-left py-3 px-4 text-dark-500 font-medium">Symbol</th>
                    <th className="text-right py-3 px-4 text-dark-500 font-medium">Price</th>
                    <th className="text-right py-3 px-4 text-dark-500 font-medium">Change</th>
                    <th className="text-right py-3 px-4 text-dark-500 font-medium">Sentiment</th>
                    <th className="text-center py-3 px-4 text-dark-500 font-medium">Volume</th>
                  </tr>
                </thead>
                <tbody>
                  {marketMoversData.map((stock, index) => (
                    <tr key={index} className="border-b border-dark-400/50 hover:bg-dark-300/30 transition-colors">
                      <td className="py-3 px-4 text-white font-mono font-bold">{stock.symbol}</td>
                      <td className="py-3 px-4 text-right text-white font-mono">${stock.price}</td>
                      <td className="py-3 px-4 text-right">
                        <span className={`font-medium ${stock.change > 0 ? 'text-trading-profit' : 'text-trading-loss'}`}>
                          {stock.change > 0 ? '+' : ''}{stock.change}%
                        </span>
                      </td>
                      <td className="py-3 px-4 text-right">
                        <span className={`font-bold ${
                          stock.sentiment > 0.7 ? 'text-trading-profit' :
                          stock.sentiment > 0.4 ? 'text-yellow-400' : 'text-trading-loss'
                        }`}>
                          {(stock.sentiment * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td className="py-3 px-4 text-center">
                        <span className={`text-xs px-2 py-1 sharp-card font-medium ${
                          stock.volume === 'Extreme' ? 'bg-trading-loss/20 text-trading-loss' :
                          stock.volume === 'Very High' ? 'bg-yellow-400/20 text-yellow-400' :
                          stock.volume === 'High' ? 'bg-trading-profit/20 text-trading-profit' :
                          'bg-dark-400/20 text-dark-400'
                        }`}>
                          {stock.volume}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </motion.div>

        {/* Fear & Greed Index Breakdown */}
        <motion.div variants={itemVariants}>
          <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <ExclamationTriangleIcon className="w-5 h-5 mr-2 text-accent-yellow" />
              Fear & Greed Index Components
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
              {fearGreedData.map((component, index) => (
                <div key={index} className="text-center p-4 bg-dark-300/30 sharp-card">
                  <div className="text-sm text-white font-medium mb-2">{component.category}</div>
                  <div className={`text-2xl font-bold mb-2 ${
                    component.value > 70 ? 'text-trading-profit' :
                    component.value > 30 ? 'text-yellow-400' : 'text-trading-loss'
                  }`}>
                    {component.value}
                  </div>
                  <div className={`text-xs font-medium px-2 py-1 sharp-card ${
                    component.status === 'Greed' ? 'bg-trading-profit/20 text-trading-profit' :
                    component.status === 'Neutral' ? 'bg-yellow-400/20 text-yellow-400' :
                    'bg-trading-loss/20 text-trading-loss'
                  }`}>
                    {component.status}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Detailed News Analysis */}
        <motion.div variants={itemVariants}>
          <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <InformationCircleIcon className="w-5 h-5 mr-2 text-accent-blue" />
              Latest Market-Moving News
            </h3>

            <div className="space-y-4">
              {(top_news_sentiment || []).map((news: any, index: number) => (
                <div key={index} className="p-4 bg-dark-300/50 sharp-card border-l-4 border-l-accent-blue">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <span className="font-mono font-semibold text-accent-blue text-sm">{news.source}</span>
                        <span className="text-xs text-dark-500">
                          {new Date(news.published_at).toLocaleTimeString()}
                        </span>
                      </div>
                      <h4 className="text-white font-medium mb-1">{news.title}</h4>
                      <div className="flex items-center space-x-4">
                        <div className="flex items-center space-x-2">
                          <span className="text-xs text-dark-500">Sentiment:</span>
                          <span className={`text-sm font-bold ${
                            news.sentiment_score > 0.6 ? 'text-trading-profit' :
                            news.sentiment_score > 0.4 ? 'text-yellow-400' : 'text-trading-loss'
                          }`}>
                            {(news.sentiment_score * 100).toFixed(0)}%
                          </span>
                        </div>
                        {news.impact_score && (
                          <div className="flex items-center space-x-2">
                            <span className="text-xs text-dark-500">Impact:</span>
                            <span className={`text-sm font-bold ${
                              news.impact_score > 0.8 ? 'text-trading-loss' :
                              news.impact_score > 0.6 ? 'text-yellow-400' : 'text-trading-profit'
                            }`}>
                              {(news.impact_score * 100).toFixed(0)}%
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>

      </motion.div>
    </Layout>
  );
}