'use client'

import { useState, useEffect } from 'react';
import Layout from '@/components/Layout/Layout';
import { motion } from 'framer-motion';
import { GlobeAltIcon, NewspaperIcon, ChatBubbleLeftRightIcon } from '@heroicons/react/24/outline';

export default function SentimentAnalysisPage() {
  const [sentimentData, setSentimentData] = useState<any>(null);
  const [macroSentiment, setMacroSentiment] = useState<any>(null);
  const [loading, setLoading] = useState(true);

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
        // Mock data for sentiment analysis - structure matches expected format
        const mockData = {
          macro_summary: {
            overall_risk_score: 0.35,
            geopolitical_tension_score: 0.42,
            economic_uncertainty_score: 0.28,
            most_impacted_sectors: ['Technology', 'Energy', 'Finance']
          },
          top_news_sentiment: [
            { source: 'Reuters', title: 'Fed signals potential rate cuts ahead', sentiment_score: 0.68, published_at: '2024-01-15T10:30:00Z' },
            { source: 'Bloomberg', title: 'Tech earnings beat expectations', sentiment_score: 0.82, published_at: '2024-01-15T09:15:00Z' },
            { source: 'CNBC', title: 'Energy sector faces regulatory challenges', sentiment_score: 0.25, published_at: '2024-01-15T08:45:00Z' },
            { source: 'Financial Times', title: 'Market volatility expected to continue', sentiment_score: 0.35, published_at: '2024-01-15T07:30:00Z' }
          ],
          social_media_sentiment: {
            twitter: {
              overall_score: 0.72,
              trending_topics: [
                { topic: '#AI', sentiment: 0.85, mentions: 15420 },
                { topic: '#FedPolicy', sentiment: 0.45, mentions: 8930 },
                { topic: '#TechEarnings', sentiment: 0.78, mentions: 6750 }
              ]
            },
            reddit: {
              overall_score: 0.61,
              top_discussions: [
                { title: 'Market predictions for Q2', sentiment: 0.55, upvotes: 1247 },
                { title: 'Tech stock analysis', sentiment: 0.73, upvotes: 892 },
                { title: 'Energy sector concerns', sentiment: 0.28, upvotes: 567 }
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

  if (loading || !sentimentData) {
    return <Layout title="Sentiment Analysis"><div>Loading...</div></Layout>;
  }

  const { macro_summary, top_news_sentiment, social_media_sentiment } = sentimentData;

  return (
    <Layout title="Sentiment Analysis" subtitle="Market, News & Geopolitical Mood">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-8">
        
        {/* Macro & Geopolitical Section */}
        <div className="bg-dark-200 p-6 sharp-card border border-dark-300">
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center"><GlobeAltIcon className="w-6 h-6 mr-2 text-accent-blue" />Macro & Geopolitical Overview</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <h3 className="text-md text-dark-500">Overall Risk Score</h3>
              <p className="text-3xl font-bold text-red-400">{((macro_summary?.overall_risk_score || 0) * 100).toFixed(1)}%</p>
            </div>
            <div>
              <h3 className="text-md text-dark-500">Top Risk Type</h3>
              <p className="text-xl font-semibold text-white">{(macro_summary?.top_risk_type || 'N/A').replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}</p>
            </div>
            <div>
              <h3 className="text-md text-dark-500">Top Impacted Sectors</h3>
              <div className="flex flex-wrap gap-2 mt-2">
                {(macro_summary?.top_impacted_sectors || []).map((sector: string) => (
                  <span key={sector} className="bg-red-500/20 text-red-300 text-xs font-medium px-2.5 py-0.5 rounded-full">{sector}</span>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* News Sentiment Section */}
        <div className="bg-dark-200 p-6 sharp-card border border-dark-300">
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center"><NewspaperIcon className="w-6 h-6 mr-2 text-accent-purple" />Top News Sentiment</h2>
          <div className="space-y-3">
            {(top_news_sentiment || []).map((news: any, index: number) => (
              <div key={index} className="flex items-center justify-between p-3 bg-dark-300/50 sharp-card">
                <div>
                  <span className="font-mono font-semibold text-white">{news.symbol}</span>
                  <p className="text-sm text-dark-500 truncate max-w-md">{news.title}</p>
                </div>
                <div className={`text-lg font-bold ${news.sentiment > 0 ? 'text-trading-profit' : 'text-trading-loss'}`}>
                  {(news.sentiment * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Social Media Section */}
        <div className="bg-dark-200 p-6 sharp-card border border-dark-300">
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center"><ChatBubbleLeftRightIcon className="w-6 h-6 mr-2 text-accent-yellow" />Social Media Pulse</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h3 className="text-md text-dark-500">Overall Market Sentiment</h3>
              <p className={`text-3xl font-bold ${social_media_sentiment.market_sentiment > 0 ? 'text-trading-profit' : 'text-trading-loss'}`}>
                {(social_media_sentiment.market_sentiment * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <h3 className="text-md text-dark-500">Top Trending Stocks</h3>
              <div className="space-y-2 mt-2">
                {(social_media_sentiment?.trending_stocks || []).map((stock: any) => (
                  <div key={stock.symbol} className="flex justify-between text-sm">
                    <span className="font-mono text-white">{stock.symbol}</span>
                    <span className={`font-semibold ${stock.sentiment > 0 ? 'text-trading-profit' : 'text-trading-loss'}`}>
                      {(stock.sentiment * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Macro Economic Sentiment */}
        {macroSentiment && (
          <div className="bg-dark-200 p-6 sharp-card border border-dark-300">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
              <GlobeAltIcon className="w-6 h-6 mr-2 text-accent-blue" />
              Macro Economic Sentiment Analysis
            </h2>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Economic Indicators */}
              <div>
                <h3 className="text-lg font-medium text-accent-blue mb-3">Economic Indicators</h3>
                <div className="space-y-3">
                  {Object.entries(macroSentiment.economic_indicators || {
                    'inflation_sentiment': 0.42,
                    'employment_sentiment': 0.78,
                    'gdp_growth_sentiment': 0.65,
                    'interest_rates_sentiment': 0.38
                  }).map(([indicator, sentiment], index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-dark-300/50 sharp-card">
                      <div>
                        <div className="text-sm text-white font-medium">
                          {indicator.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
                        </div>
                        <div className="text-xs text-dark-500">Market confidence</div>
                      </div>
                      <div className={`text-lg font-bold ${
                        Number(sentiment) > 0.6 ? 'text-trading-profit' :
                        Number(sentiment) > 0.4 ? 'text-yellow-400' : 'text-trading-loss'
                      }`}>
                        {(Number(sentiment) * 100).toFixed(0)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Central Bank Sentiment */}
              <div>
                <h3 className="text-lg font-medium text-accent-blue mb-3">Central Bank Communications</h3>
                <div className="space-y-3">
                  {(macroSentiment.central_bank_communications || [
                    {
                      bank: 'Federal Reserve',
                      latest_statement: 'Committed to price stability',
                      sentiment_score: 0.72,
                      impact_on_markets: 'positive'
                    },
                    {
                      bank: 'ECB',
                      latest_statement: 'Monitoring inflation closely',
                      sentiment_score: 0.58,
                      impact_on_markets: 'neutral'
                    },
                    {
                      bank: 'Bank of Japan',
                      latest_statement: 'Maintaining accommodative policy',
                      sentiment_score: 0.81,
                      impact_on_markets: 'positive'
                    }
                  ]).map((comm: any, index: number) => (
                    <div key={index} className="p-3 bg-dark-300/50 sharp-card">
                      <div className="flex items-center justify-between mb-2">
                        <div className="text-sm text-white font-medium">{comm.bank}</div>
                        <div className={`text-sm font-bold ${
                          comm.sentiment_score > 0.7 ? 'text-trading-profit' :
                          comm.sentiment_score > 0.5 ? 'text-yellow-400' : 'text-trading-loss'
                        }`}>
                          {(comm.sentiment_score * 100).toFixed(0)}%
                        </div>
                      </div>
                      <div className="text-xs text-dark-500 mb-1">{comm.latest_statement}</div>
                      <div className={`text-xs font-medium ${
                        comm.impact_on_markets === 'positive' ? 'text-trading-profit' :
                        comm.impact_on_markets === 'neutral' ? 'text-yellow-400' : 'text-trading-loss'
                      }`}>
                        Market Impact: {comm.impact_on_markets}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Global Risk Factors */}
            <div className="mt-6 pt-6 border-t border-dark-300">
              <h3 className="text-lg font-medium text-accent-blue mb-3">Global Risk Assessment</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {Object.entries(macroSentiment.global_risks || {
                  'geopolitical_tension': { score: 0.67, trend: 'increasing' },
                  'trade_war_risk': { score: 0.43, trend: 'stable' },
                  'supply_chain_disruption': { score: 0.52, trend: 'decreasing' }
                }).map(([risk, data], index) => (
                  <div key={index} className="text-center p-4 bg-dark-300/30 sharp-card">
                    <div className="text-sm text-white font-medium mb-2">
                      {risk.replace('_', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}
                    </div>
                    <div className={`text-2xl font-bold mb-1 ${
                      Number(data.score) > 0.7 ? 'text-trading-loss' :
                      Number(data.score) > 0.5 ? 'text-yellow-400' : 'text-trading-profit'
                    }`}>
                      {(Number(data.score) * 100).toFixed(0)}%
                    </div>
                    <div className={`text-xs font-medium ${
                      data.trend === 'increasing' ? 'text-trading-loss' :
                      data.trend === 'stable' ? 'text-yellow-400' : 'text-trading-profit'
                    }`}>
                      {data.trend}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Market Sector Impact */}
            <div className="mt-6 pt-6 border-t border-dark-300">
              <h3 className="text-lg font-medium text-accent-blue mb-3">Sector Impact Analysis</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                {Object.entries(macroSentiment.sector_impacts || {
                  'Technology': 0.78,
                  'Healthcare': 0.65,
                  'Financial': 0.42,
                  'Energy': 0.35,
                  'Consumer': 0.59,
                  'Industrial': 0.48,
                  'Materials': 0.51,
                  'Utilities': 0.73
                }).map(([sector, impact], index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">{sector}</span>
                      <span className={`text-xs font-bold ${
                        Number(impact) > 0.7 ? 'text-trading-profit' :
                        Number(impact) > 0.5 ? 'text-yellow-400' : 'text-trading-loss'
                      }`}>
                        {(Number(impact) * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="w-full bg-dark-300 h-1.5">
                      <motion.div
                        className={`h-1.5 ${
                          Number(impact) > 0.7 ? 'bg-trading-profit' :
                          Number(impact) > 0.5 ? 'bg-yellow-400' : 'bg-trading-loss'
                        }`}
                        initial={{ width: 0 }}
                        animate={{ width: `${Number(impact) * 100}%` }}
                        transition={{ duration: 1, delay: index * 0.1 }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Enhanced Social Media Analysis */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Twitter Analysis */}
          <div className="bg-dark-200 p-6 sharp-card border border-dark-300">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <ChatBubbleLeftRightIcon className="w-5 h-5 mr-2 text-accent-blue" />
              Twitter Market Sentiment
            </h3>

            <div className="space-y-4">
              <div className="text-center">
                <div className={`text-3xl font-bold mb-2 ${
                  (social_media_sentiment?.twitter?.overall_score || 0.72) > 0.6 ? 'text-trading-profit' :
                  (social_media_sentiment?.twitter?.overall_score || 0.72) > 0.4 ? 'text-yellow-400' : 'text-trading-loss'
                }`}>
                  {((social_media_sentiment?.twitter?.overall_score || 0.72) * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-dark-500">Overall Twitter Sentiment</div>
              </div>

              <div>
                <h4 className="text-sm font-medium text-accent-blue mb-2">Trending Topics</h4>
                <div className="space-y-2">
                  {(social_media_sentiment?.twitter?.trending_topics || []).map((topic: any, index: number) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-dark-300/50 sharp-card">
                      <div>
                        <div className="text-sm text-white font-mono">{topic.topic}</div>
                        <div className="text-xs text-dark-500">{topic.mentions.toLocaleString()} mentions</div>
                      </div>
                      <div className={`text-sm font-bold ${
                        topic.sentiment > 0.7 ? 'text-trading-profit' :
                        topic.sentiment > 0.5 ? 'text-yellow-400' : 'text-trading-loss'
                      }`}>
                        {(topic.sentiment * 100).toFixed(0)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Reddit Analysis */}
          <div className="bg-dark-200 p-6 sharp-card border border-dark-300">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <ChatBubbleLeftRightIcon className="w-5 h-5 mr-2 text-accent-purple" />
              Reddit Discussion Analysis
            </h3>

            <div className="space-y-4">
              <div className="text-center">
                <div className={`text-3xl font-bold mb-2 ${
                  (social_media_sentiment?.reddit?.overall_score || 0.61) > 0.6 ? 'text-trading-profit' :
                  (social_media_sentiment?.reddit?.overall_score || 0.61) > 0.4 ? 'text-yellow-400' : 'text-trading-loss'
                }`}>
                  {((social_media_sentiment?.reddit?.overall_score || 0.61) * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-dark-500">Overall Reddit Sentiment</div>
              </div>

              <div>
                <h4 className="text-sm font-medium text-accent-blue mb-2">Top Discussions</h4>
                <div className="space-y-2">
                  {(social_media_sentiment?.reddit?.top_discussions || []).map((discussion: any, index: number) => (
                    <div key={index} className="p-2 bg-dark-300/50 sharp-card">
                      <div className="flex items-center justify-between mb-1">
                        <div className="text-sm text-white font-medium">{discussion.title}</div>
                        <div className={`text-sm font-bold ${
                          discussion.sentiment > 0.7 ? 'text-trading-profit' :
                          discussion.sentiment > 0.5 ? 'text-yellow-400' : 'text-trading-loss'
                        }`}>
                          {(discussion.sentiment * 100).toFixed(0)}%
                        </div>
                      </div>
                      <div className="text-xs text-dark-500">{discussion.upvotes} upvotes</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

      </motion.div>
    </Layout>
  );
}
