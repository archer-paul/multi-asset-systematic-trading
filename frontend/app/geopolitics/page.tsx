'use client'

import { useState, useEffect } from 'react';
import Layout from '@/components/Layout/Layout';
import { motion } from 'framer-motion';
import { GlobeAltIcon, ShieldExclamationIcon, FireIcon } from '@heroicons/react/24/outline';

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
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
    const loadData = async () => {
      setLoading(true);
      try {
        const response = await fetch(`${API_URL}/api/geopolitical-risk`);
        if (!response.ok) throw new Error('Failed to fetch geopolitical data');
        const data: RiskData = await response.json();
        setRiskData(data);
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
      </motion.div>
    </Layout>
  );
}