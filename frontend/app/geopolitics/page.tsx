'use client'

import { useState, useEffect } from 'react';
import Layout from '@/components/Layout/Layout';
import { motion } from 'framer-motion';
import { GlobeAltIcon, ShieldExclamationIcon, FireIcon, BuildingLibraryIcon, CubeIcon, ChartBarIcon } from '@heroicons/react/24/outline';

// Define interfaces for our data structure
interface Risk {
  risk_type: string;
  source: string;
  title: string;
  link: string;
  risk_score: number;
  impact_sectors: string[];
  timestamp: string;
}

interface Summary {
  overall_risk_score: number;
  top_risk_type: string;
  top_impacted_sectors: string[];
}

interface RiskData {
  risks: Risk[];
  summary: Summary;
}

export default function GeopoliticsPage() {
  const [riskData, setRiskData] = useState<RiskData | null>(null);
  const [macroAnalysis, setMacroAnalysis] = useState<any>(null);
  const [commoditiesAnalysis, setCommoditiesAnalysis] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
    const loadData = async () => {
      setLoading(true);
      try {
        // Load geopolitical risk data
        const response = await fetch(`${API_URL}/api/geopolitical-risk`);
        if (!response.ok) throw new Error('Failed to fetch geopolitical data');
        const data: RiskData = await response.json();
        setRiskData(data);

        // Load macro-economic analysis
        try {
          const macroResponse = await fetch(`${API_URL}/api/macro-economic-analysis`);
          if (macroResponse.ok) {
            const macroData = await macroResponse.json();
            setMacroAnalysis(macroData);
          }
        } catch (macroError) {
          console.log('Macro-economic analysis not available');
        }

        // Load commodities analysis
        try {
          const commoditiesResponse = await fetch(`${API_URL}/api/commodities-analysis`);
          if (commoditiesResponse.ok) {
            const commoditiesData = await commoditiesResponse.json();
            setCommoditiesAnalysis(commoditiesData);
          }
        } catch (commoditiesError) {
          console.log('Commodities analysis not available');
        }

      } catch (error) {
        console.error(error);
        // Set a default state on error to avoid crash
        setRiskData({
          risks: [],
          summary: { overall_risk_score: 0, top_risk_type: 'Error', top_impacted_sectors: [] }
        });
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, []);

  if (loading || !riskData) {
    return <Layout title="Geopolitical Analysis"><div>Loading...</div></Layout>;
  }

  const { risks, summary } = riskData;

  return (
    <Layout title="Geopolitical Analysis" subtitle="Global Risk Dashboard">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <motion.div className="bg-dark-200 p-6 sharp-card">
            <h3 className="text-lg font-semibold text-white flex items-center"><GlobeAltIcon className="w-6 h-6 mr-2" />Overall Risk Score</h3>
            <p className="text-4xl font-bold text-red-400">{(summary.overall_risk_score * 100).toFixed(1)}%</p>
          </motion.div>
          <motion.div className="bg-dark-200 p-6 sharp-card">
            <h3 className="text-lg font-semibold text-white flex items-center"><FireIcon className="w-6 h-6 mr-2" />Top Risk Type</h3>
            <p className="text-2xl font-semibold text-white">{summary.top_risk_type.replace('_', ' ').split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}</p>
          </motion.div>
          <motion.div className="bg-dark-200 p-6 sharp-card">
            <h3 className="text-lg font-semibold text-white flex items-center"><ShieldExclamationIcon className="w-6 h-6 mr-2" />Top Impacted Sectors</h3>
            <div className="flex flex-wrap gap-2 mt-2">
              {summary.top_impacted_sectors.map(sector => (
                <span key={sector} className="bg-red-500/20 text-red-300 text-xs font-medium px-2.5 py-0.5 sharp-button">{sector}</span>
              ))}
            </div>
          </motion.div>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-white mb-4">Active Risk Events</h2>
          <div className="space-y-4">
            {risks.map((risk, index) => (
              <motion.div key={index} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.1 }} className="bg-dark-200 p-4 sharp-card border border-dark-300">
                <h4 className="font-semibold text-white">{risk.title}</h4>
                <div className="flex items-center justify-between text-sm text-dark-500 mt-1">
                  <span>Source: {risk.source} | Type: {risk.risk_type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
                  <a href={risk.link} target="_blank" rel="noopener noreferrer" className="text-accent-blue hover:underline">Read More</a>
                </div>
                <div className="w-full bg-dark-300 h-2.5 mt-2 rounded-full">
                  <div className="bg-red-500 h-2.5" style={{ width: `${risk.risk_score * 100}%` }}></div>
                </div>
                <div className="text-right text-xs text-white mt-1">Risk Score: {(risk.risk_score * 100).toFixed(0)}</div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Macro-Economic Context */}
        {macroAnalysis && (
          <div>
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
              <BuildingLibraryIcon className="w-6 h-6 mr-2 text-accent-blue" />
              Macro-Economic Context
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <motion.div className="bg-dark-200 p-6 sharp-card border border-dark-300">
                <h3 className="text-lg font-semibold text-white">Economic Score</h3>
                <p className="text-3xl font-bold text-accent-blue">{((macroAnalysis.score || 0.5) * 100).toFixed(1)}%</p>
                <p className="text-sm text-dark-500 mt-2">Overall economic strength indicator</p>
              </motion.div>
              <motion.div className="bg-dark-200 p-6 sharp-card border border-dark-300">
                <h3 className="text-lg font-semibold text-white">Inflation Risk</h3>
                <p className="text-3xl font-bold text-yellow-400">{((macroAnalysis.economic_indicators?.inflation_risk || 0) * 100).toFixed(1)}%</p>
                <p className="text-sm text-dark-500 mt-2">Inflation pressure assessment</p>
              </motion.div>
              <motion.div className="bg-dark-200 p-6 sharp-card border border-dark-300">
                <h3 className="text-lg font-semibold text-white">GDP Growth</h3>
                <p className="text-3xl font-bold text-trading-profit">{((macroAnalysis.economic_indicators?.gdp_growth || 0) * 100).toFixed(1)}%</p>
                <p className="text-sm text-dark-500 mt-2">Economic growth trend</p>
              </motion.div>
            </div>
          </div>
        )}

        {/* Commodities & Market Stress */}
        {commoditiesAnalysis && (
          <div>
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
              <CubeIcon className="w-6 h-6 mr-2 text-accent-purple" />
              Commodities & Market Stress
            </h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Inflation Hedges */}
              <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
                <h3 className="text-lg font-semibold text-white mb-4">Inflation Hedge Signals</h3>
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-white">Gold Correlation</span>
                      <span className="text-xs text-dark-500">
                        {((commoditiesAnalysis.analysis?.gold_correlation || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-dark-300 h-2 sharp-card overflow-hidden">
                      <motion.div
                        className="h-full bg-gradient-to-r from-yellow-400 to-yellow-600"
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.abs((commoditiesAnalysis.analysis?.gold_correlation || 0) * 100)}%` }}
                        transition={{ duration: 1 }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-white">Inflation Hedge Score</span>
                      <span className="text-xs text-dark-500">
                        {((commoditiesAnalysis.analysis?.inflation_hedge_score || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-dark-300 h-2 sharp-card overflow-hidden">
                      <motion.div
                        className="h-full bg-gradient-to-r from-orange-400 to-orange-600"
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.abs((commoditiesAnalysis.analysis?.inflation_hedge_score || 0) * 100)}%` }}
                        transition={{ duration: 1, delay: 0.2 }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Market Volatility */}
              <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
                <h3 className="text-lg font-semibold text-white mb-4">Market Volatility Indicators</h3>
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-white">Commodities Momentum</span>
                      <span className="text-xs text-dark-500">
                        {((commoditiesAnalysis.analysis?.commodities_momentum || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-dark-300 h-2 sharp-card overflow-hidden">
                      <motion.div
                        className={`h-full ${(commoditiesAnalysis.analysis?.commodities_momentum || 0) >= 0 ? 'bg-gradient-to-r from-trading-profit to-green-600' : 'bg-gradient-to-r from-trading-loss to-red-600'}`}
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.abs((commoditiesAnalysis.analysis?.commodities_momentum || 0) * 100)}%` }}
                        transition={{ duration: 1 }}
                      />
                    </div>
                  </div>
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-white">Volatility Score</span>
                      <span className="text-xs text-dark-500">
                        {((commoditiesAnalysis.analysis?.volatility_score || 0.5) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-dark-300 h-2 sharp-card overflow-hidden">
                      <motion.div
                        className="h-full bg-gradient-to-r from-accent-purple to-purple-600"
                        initial={{ width: 0 }}
                        animate={{ width: `${(commoditiesAnalysis.analysis?.volatility_score || 0.5) * 100}%` }}
                        transition={{ duration: 1, delay: 0.2 }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Top Commodities */}
            {commoditiesAnalysis.raw_data && Object.keys(commoditiesAnalysis.raw_data).length > 0 && (
              <div className="mt-6">
                <h3 className="text-lg font-semibold text-white mb-4">Key Commodities</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(commoditiesAnalysis.raw_data).slice(0, 6).map(([symbol, data]: [string, any], index) => (
                    <motion.div
                      key={symbol}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="bg-dark-300/50 p-4 sharp-card"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <h4 className="font-semibold text-white">{data.name || symbol}</h4>
                          <p className="text-sm text-dark-500">{symbol}</p>
                        </div>
                        <div className="text-right">
                          <p className="font-mono text-white">${data.price?.toFixed(2) || 'N/A'}</p>
                          <p className={`text-xs ${data.change_pct_24h >= 0 ? 'text-trading-profit' : 'text-trading-loss'}`}>
                            {data.change_pct_24h >= 0 ? '+' : ''}{data.change_pct_24h?.toFixed(2) || 0}%
                          </p>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </motion.div>
    </Layout>
  );
}