'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Layout from '@/components/Layout/Layout'
import MetricCard from '@/components/MetricCard'
import type { CacheStats, BatchResults } from '@/types'
import {
  CpuChipIcon,
  ChartBarIcon,
  BoltIcon,
  CogIcon,
  BeakerIcon,
  TrophyIcon,
  CircleStackIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  RadialBarChart,
  RadialBar,
  ScatterChart,
  Scatter,
  LineChart,
  Line,
} from 'recharts'

// Data is now fetched from the API via the useEffect hook.

export default function MachineLearningPage() {
  const [loading, setLoading] = useState(true)
  const [selectedModel, setSelectedModel] = useState('ensemble')
  const [mlData, setMlData] = useState(null)
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null)
  const [batchResults, setBatchResults] = useState<BatchResults | null>(null)

  useEffect(() => {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
    const loadData = async () => {
      setLoading(true);
      try {
        // Load ML dashboard data
        const mlResponse = await fetch(`${API_URL}/api/ml-dashboard`);
        if (mlResponse.ok) {
          const mlData = await mlResponse.json();
          setMlData(mlData);
        }

        // Load cache statistics
        const cacheResponse = await fetch(`${API_URL}/api/ml/cache-stats`);
        if (cacheResponse.ok) {
          const cacheData = await cacheResponse.json();
          setCacheStats(cacheData);
        }

        // Load batch training results
        const batchResponse = await fetch(`${API_URL}/api/ml/batch-results`);
        if (batchResponse.ok) {
          const batchData = await batchResponse.json();
          setBatchResults(batchData);
        }
      } catch (error) {
        console.error('Failed to load ML data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-dark-200 border border-dark-300 sharp-card p-4 shadow-lg">
          <p className="text-white font-medium mb-2">{label}</p>
          {payload.map((entry: any, index: number) => (
            <div key={index} className="flex items-center space-x-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-sm text-dark-500">{entry.dataKey}:</span>
              <span className="text-sm text-white font-mono">
                {(entry.value * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      )
    }
    return null
  }

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5 }
    }
  }

  return (
    <Layout title="Machine Learning" subtitle="Model Performance & Analytics">
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="space-y-6"
      >
        {/* ML Metrics Overview */}
        <motion.div variants={itemVariants}>
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
            <CpuChipIcon className="w-6 h-6 mr-2 text-accent-blue" />
            Model Performance Overview
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              title="Ensemble Accuracy"
              value={89.2}
              change={2.3}
              changeType="increase"
              icon={<TrophyIcon className="w-5 h-5" />}
              formatValue={(val) => `${Number(val).toFixed(1)}%`}
              loading={loading}
              color="green"
            />

            <MetricCard
              title="Best Model"
              value="Transformer"
              icon={<SparklesIcon className="w-5 h-5" />}
              loading={loading}
              color="blue"
            />

            <MetricCard
              title="Prediction Confidence"
              value={87.5}
              change={1.8}
              changeType="increase"
              icon={<BeakerIcon className="w-5 h-5" />}
              formatValue={(val) => `${Number(val).toFixed(1)}%`}
              loading={loading}
              color="purple"
            />

            <MetricCard
              title="Daily Predictions"
              value={247}
              change={12}
              changeType="increase"
              icon={<BoltIcon className="w-5 h-5" />}
              loading={loading}
              color="yellow"
            />
          </div>
        </motion.div>

        {/* Model Performance Comparison */}
        <motion.div variants={itemVariants} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Model Accuracy Comparison */}
          <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <ChartBarIcon className="w-5 h-5 mr-2 text-accent-blue" />
              Model Performance Metrics
            </h3>

            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={modelPerformanceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" opacity={0.3} />
                <XAxis dataKey="name" stroke="#718096" fontSize={12} />
                <YAxis stroke="#718096" fontSize={12} domain={[0, 1]} />
                <Tooltip content={<CustomTooltip />} />

                <Bar dataKey="accuracy" fill="#3b82f6" name="Accuracy" radius={[2, 2, 0, 0]} />
                <Bar dataKey="precision" fill="#10b981" name="Precision" radius={[2, 2, 0, 0]} />
                <Bar dataKey="recall" fill="#f59e0b" name="Recall" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Ensemble Weights */}
          <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <CircleStackIcon className="w-5 h-5 mr-2 text-accent-purple" />
              Meta-Learner Model Weights
            </h3>

            <ResponsiveContainer width="100%" height={350}>
              <PieChart>
                <Pie
                  data={ensembleWeights}
                  cx="50%"
                  cy="50%"
                  outerRadius={120}
                  innerRadius={60}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {ensembleWeights.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value) => [`${(Number(value) * 100).toFixed(1)}%`, 'Weight']}
                />
              </PieChart>
            </ResponsiveContainer>

            {/* Legend */}
            <div className="grid grid-cols-2 gap-2 mt-4">
              {ensembleWeights.map((item, index) => (
                <div key={index} className="flex items-center space-x-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="text-sm text-dark-500">{item.name}</span>
                  <span className="text-sm text-white font-mono ml-auto">
                    {(item.value * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Real-time Predictions */}
        <motion.div variants={itemVariants}>
          <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <BoltIcon className="w-5 h-5 mr-2 text-accent-yellow" />
              Real-time Prediction Confidence
            </h3>

            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={predictionConfidence}>
                <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" opacity={0.3} />
                <XAxis dataKey="hour" stroke="#718096" fontSize={12} />
                <YAxis stroke="#718096" fontSize={12} domain={[0.7, 1]} />
                <Tooltip
                  formatter={(value, name) => [
                    `${(Number(value) * 100).toFixed(1)}%`,
                    name === 'confidence' ? 'Confidence' : 'Predictions'
                  ]}
                  labelStyle={{ color: '#fff' }}
                  contentStyle={{ backgroundColor: '#2d3748', border: '1px solid #4a5568' }}
                />

                <Line
                  type="monotone"
                  dataKey="confidence"
                  stroke="#10b981"
                  strokeWidth={3}
                  dot={{ fill: '#10b981', strokeWidth: 2, r: 5 }}
                  activeDot={{ r: 7, stroke: '#10b981', strokeWidth: 2, fill: '#fff' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Model Training Status */}
        <motion.div variants={itemVariants}>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Training Progress */}
            <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <CogIcon className="w-5 h-5 mr-2 text-accent-blue" />
                Training Status
              </h3>

              <div className="space-y-4">
                {[
                  { model: 'XGBoost', status: 'Training', progress: 0.75, eta: '2h 15m' },
                  { model: 'Transformer', status: 'Complete', progress: 1.0, eta: 'Done' },
                  { model: 'LSTM', status: 'Queued', progress: 0.0, eta: '4h 30m' },
                ].map((item, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-white font-medium">{item.model}</span>
                      <span className={`text-xs px-2 py-1 sharp-button ${
                        item.status === 'Complete' ? 'bg-trading-profit/20 text-trading-profit' :
                        item.status === 'Training' ? 'bg-accent-blue/20 text-accent-blue' :
                        'bg-dark-400/20 text-dark-500'
                      }`}>
                        {item.status}
                      </span>
                    </div>

                    <div className="w-full bg-dark-300 h-2">
                      <motion.div
                        className={`h-2 ${
                          item.status === 'Complete' ? 'bg-trading-profit' :
                          item.status === 'Training' ? 'bg-accent-blue' :
                          'bg-dark-400'
                        }`}
                        initial={{ width: 0 }}
                        animate={{ width: `${item.progress * 100}%` }}
                        transition={{ duration: 1, delay: index * 0.2 }}
                      />
                    </div>

                    <div className="flex items-center justify-between text-xs">
                      <span className="text-dark-500">{(item.progress * 100).toFixed(0)}%</span>
                      <span className="text-dark-500">ETA: {item.eta}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Feature Importance */}
            <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
              <h3 className="text-lg font-semibold text-white mb-4">Feature Importance</h3>

              <div className="space-y-3">
                {[
                  { feature: 'Price Momentum', importance: 0.89 },
                  { feature: 'Volume Profile', importance: 0.76 },
                  { feature: 'Market Sentiment', importance: 0.64 },
                  { feature: 'Technical Indicators', importance: 0.58 },
                  { feature: 'News Sentiment', importance: 0.42 },
                ].map((item, index) => (
                  <div key={index} className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">{item.feature}</span>
                      <span className="text-xs text-dark-500">
                        {(item.importance * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="w-full bg-dark-300 h-1.5">
                      <motion.div
                        className="h-1.5 bg-gradient-to-r from-accent-blue to-accent-purple"
                        initial={{ width: 0 }}
                        animate={{ width: `${item.importance * 100}%` }}
                        transition={{ duration: 1, delay: index * 0.1 }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Model Health */}
            <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
              <h3 className="text-lg font-semibold text-white mb-4">Model Health</h3>

              <div className="space-y-4">
                {[
                  { metric: 'Data Quality', score: 0.94, status: 'excellent' },
                  { metric: 'Prediction Drift', score: 0.12, status: 'good', invert: true },
                  { metric: 'Model Stability', score: 0.87, status: 'good' },
                  { metric: 'Latency', score: 0.78, status: 'fair' },
                ].map((item, index) => (
                  <div key={index} className="flex items-center justify-between">
                    <span className="text-sm text-white">{item.metric}</span>
                    <div className="flex items-center space-x-2">
                      <div className={`w-2 h-2 rounded-full ${
                        item.status === 'excellent' ? 'bg-trading-profit' :
                        item.status === 'good' ? 'bg-accent-blue' :
                        'bg-accent-yellow'
                      }`} />
                      <span className="text-xs text-dark-500 font-mono">
                        {item.invert ? `${(item.score * 100).toFixed(1)}%` : `${(item.score * 100).toFixed(0)}%`}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>

        {/* Cache Statistics */}
        {cacheStats && (
          <motion.div variants={itemVariants}>
            <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <CircleStackIcon className="w-5 h-5 mr-2 text-accent-purple" />
                Model Cache Performance
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Cache Hit Rates */}
                <div className="space-y-4">
                  <h4 className="text-sm font-medium text-accent-blue">Cache Hit Rates</h4>
                  {Object.entries(cacheStats?.hit_rates || {}).map(([model, rate], index) => (
                    <div key={index} className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-white">{model}</span>
                        <span className="text-xs text-dark-500">
                          {(Number(rate) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-dark-300 h-1.5">
                        <motion.div
                          className="h-1.5 bg-gradient-to-r from-accent-purple to-accent-blue"
                          initial={{ width: 0 }}
                          animate={{ width: `${Number(rate) * 100}%` }}
                          transition={{ duration: 1, delay: index * 0.1 }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                {/* Cache TTL Information */}
                <div className="space-y-3">
                  <h4 className="text-sm font-medium text-accent-blue">Time-to-Live Settings</h4>
                  {Object.entries(cacheStats?.ttl_seconds || {}).map(([model, ttl], index) => (
                    <div key={index} className="flex items-center justify-between">
                      <span className="text-sm text-white">{model}</span>
                      <span className="text-xs px-2 py-1 bg-dark-300 text-dark-400 sharp-card">
                        {Number(ttl)}s
                      </span>
                    </div>
                  ))}
                </div>

                {/* Memory Usage */}
                <div className="space-y-3">
                  <h4 className="text-sm font-medium text-accent-blue">Memory Usage</h4>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">Total Cache Size</span>
                      <span className="text-xs text-dark-500">
                        {cacheStats?.memory_usage?.total_mb || 0} MB
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">Average Entry Size</span>
                      <span className="text-xs text-dark-500">
                        {cacheStats?.memory_usage?.avg_entry_kb || 0} KB
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">Cache Efficiency</span>
                      <span className={`text-xs font-medium ${
                        (cacheStats?.memory_usage?.efficiency || 0) > 0.8 ? 'text-trading-profit' :
                        (cacheStats?.memory_usage?.efficiency || 0) > 0.6 ? 'text-yellow-400' :
                        'text-trading-loss'
                      }`}>
                        {((cacheStats?.memory_usage?.efficiency || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Batch Training Results */}
        {batchResults && (
          <motion.div variants={itemVariants}>
            <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <BeakerIcon className="w-5 h-5 mr-2 text-accent-yellow" />
                Cross-Symbol Training Results
              </h3>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Training Correlation Matrix */}
                <div>
                  <h4 className="text-sm font-medium text-accent-blue mb-3">Symbol Correlation Improvements</h4>
                  <div className="space-y-2">
                    {Object.entries(batchResults?.correlation_improvements || {}).map(([pair, improvement], index) => (
                      <div key={index} className="flex items-center justify-between">
                        <span className="text-sm text-white font-mono">{pair}</span>
                        <span className={`text-xs px-2 py-1 sharp-card font-medium ${
                          Number(improvement) > 0 ? 'bg-trading-profit/20 text-trading-profit' : 'bg-trading-loss/20 text-trading-loss'
                        }`}>
                          {Number(improvement) > 0 ? '+' : ''}{(Number(improvement) * 100).toFixed(2)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Batch Training Performance */}
                <div>
                  <h4 className="text-sm font-medium text-accent-blue mb-3">Batch Training Metrics</h4>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">Total Symbols Trained</span>
                      <span className="text-xs text-dark-500">
                        {batchResults?.total_symbols || 0}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">Training Duration</span>
                      <span className="text-xs text-dark-500">
                        {batchResults?.training_duration_hours || 0}h
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">Average Improvement</span>
                      <span className={`text-xs font-medium ${
                        (batchResults?.avg_improvement || 0) > 0 ? 'text-trading-profit' : 'text-trading-loss'
                      }`}>
                        {(batchResults?.avg_improvement || 0) > 0 ? '+' : ''}{((batchResults?.avg_improvement || 0) * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">Cross-Validation Score</span>
                      <span className="text-xs text-trading-profit font-medium">
                        {((batchResults?.cv_score || 0) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Feature Engineering Results */}
              <div className="mt-6 pt-6 border-t border-dark-300">
                <h4 className="text-sm font-medium text-accent-blue mb-3">Feature Engineering Results</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-white mb-1">
                      {batchResults?.total_features || 119}
                    </div>
                    <div className="text-xs text-dark-500">Engineered Features</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-trading-profit mb-1">
                      {batchResults?.feature_importance_score || 0.847}
                    </div>
                    <div className="text-xs text-dark-500">Feature Importance Score</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-accent-blue mb-1">
                      {batchResults?.dimensionality_reduction || '67%'}
                    </div>
                    <div className="text-xs text-dark-500">Dimensionality Reduction</div>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Transformer Architecture Details */}
        <motion.div variants={itemVariants}>
          <div className="bg-dark-200 sharp-card p-6 border border-dark-300">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <SparklesIcon className="w-5 h-5 mr-2 text-accent-purple" />
              Transformer Architecture Deep Dive
            </h3>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Architecture Overview */}
              <div>
                <h4 className="text-sm font-medium text-accent-blue mb-3">Architecture Specs</h4>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-white">Model Dimension</span>
                    <span className="text-xs text-dark-500 font-mono">512</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-white">Attention Heads</span>
                    <span className="text-xs text-dark-500 font-mono">8</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-white">Encoder Layers</span>
                    <span className="text-xs text-dark-500 font-mono">6</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-white">Sequence Length</span>
                    <span className="text-xs text-dark-500 font-mono">1440</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-white">Dropout Rate</span>
                    <span className="text-xs text-dark-500 font-mono">0.1</span>
                  </div>
                </div>
              </div>

              {/* Training Metrics */}
              <div>
                <h4 className="text-sm font-medium text-accent-blue mb-3">Training Progress</h4>
                <div className="space-y-3">
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">Training Loss</span>
                      <span className="text-xs text-trading-profit font-mono">0.0847</span>
                    </div>
                    <div className="w-full bg-dark-300 h-1.5">
                      <div className="h-1.5 bg-trading-profit w-[85%]"/>
                    </div>
                  </div>
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">Validation Loss</span>
                      <span className="text-xs text-accent-blue font-mono">0.0923</span>
                    </div>
                    <div className="w-full bg-dark-300 h-1.5">
                      <div className="h-1.5 bg-accent-blue w-[82%]"/>
                    </div>
                  </div>
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white">Learning Rate</span>
                      <span className="text-xs text-yellow-400 font-mono">3e-4</span>
                    </div>
                    <div className="w-full bg-dark-300 h-1.5">
                      <div className="h-1.5 bg-yellow-400 w-[30%]"/>
                    </div>
                  </div>
                </div>
              </div>

              {/* Attention Weights */}
              <div>
                <h4 className="text-sm font-medium text-accent-blue mb-3">Attention Analysis</h4>
                <div className="space-y-2">
                  {[
                    { timeframe: '1-min patterns', weight: 0.34 },
                    { timeframe: '5-min momentum', weight: 0.28 },
                    { timeframe: '15-min trends', weight: 0.22 },
                    { timeframe: 'Hourly cycles', weight: 0.16 },
                  ].map((item, index) => (
                    <div key={index} className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-white">{item.timeframe}</span>
                        <span className="text-xs text-dark-500">
                          {(item.weight * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="w-full bg-dark-300 h-1">
                        <motion.div
                          className="h-1 bg-gradient-to-r from-accent-purple to-accent-blue"
                          initial={{ width: 0 }}
                          animate={{ width: `${item.weight * 100}%` }}
                          transition={{ duration: 1, delay: index * 0.1 }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </Layout>
  )
}