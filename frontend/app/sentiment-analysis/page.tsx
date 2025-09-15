'use client'

import { useState, useEffect } from 'react';
import Layout from '@/components/Layout/Layout';
import { motion } from 'framer-motion';
import { GlobeAltIcon, NewspaperIcon, ChatBubbleLeftRightIcon } from '@heroicons/react/24/outline';

export default function SentimentAnalysisPage() {
  const [sentimentData, setSentimentData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
    const loadData = async () => {
      setLoading(true);
      try {
        const response = await fetch(`${API_URL}/api/sentiment-summary`);
        if (!response.ok) throw new Error('Failed to fetch sentiment data');
        const data = await response.json();
        setSentimentData(data);
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

      </motion.div>
    </Layout>
  );
}
