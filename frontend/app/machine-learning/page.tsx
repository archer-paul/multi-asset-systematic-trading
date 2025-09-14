'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Layout from '@/components/Layout/Layout'
import MetricCard from '@/components/MetricCard'
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

// Mock data for ML models
const modelPerformanceData = [
  { name: 'XGBoost', accuracy: 0.872, precision: 0.845, recall: 0.891, f1Score: 0.867, sharpe: 2.34 },
  { name: 'AdaBoost', accuracy: 0.834, precision: 0.821, recall: 0.847, f1Score: 0.834, sharpe: 1.98 },
  { name: 'Transformer', accuracy: 0.889, precision: 0.876, recall: 0.902, f1Score: 0.889, sharpe: 2.67 },
  { name: 'LSTM', accuracy: 0.801, precision: 0.798, recall: 0.804, f1Score: 0.801, sharpe: 1.76 },
  { name: 'Random Forest', accuracy: 0.856, precision: 0.841, recall: 0.871, f1Score: 0.856, sharpe: 2.12 },
]

const ensembleWeights = [
  { name: 'XGBoost', value: 0.28, color: '#3b82f6' },
  { name: 'Transformer', value: 0.35, color: '#10b981' },
  { name: 'AdaBoost', value: 0.15, color: '#f59e0b' },
  { name: 'LSTM', value: 0.12, color: '#ef4444' },
  { name: 'Random Forest', value: 0.10, color: '#8b5cf6' },
]

const predictionConfidence = [
  { hour: '09:00', confidence: 0.82, predictions: 15 },
  { hour: '10:00', confidence: 0.89, predictions: 23 },
  { hour: '11:00', confidence: 0.76, predictions: 18 },
  { hour: '12:00', confidence: 0.91, predictions: 31 },
  { hour: '13:00', confidence: 0.85, predictions: 27 },
  { hour: '14:00', confidence: 0.78, predictions: 21 },
  { hour: '15:00', confidence: 0.93, predictions: 34 },
  { hour: '16:00', confidence: 0.87, predictions: 28 },
]

export default function MachineLearningPage() {
  const [loading, setLoading] = useState(true)
  const [selectedModel, setSelectedModel] = useState('ensemble')

  const [mlData, setMlData] = useState(null);

  useEffect(() => {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
    const loadData = async () => {
      setLoading(true);
      try {
        const response = await fetch(`${API_URL}/api/ml-dashboard`);
        if (!response.ok) {
          throw new Error('Failed to fetch ML dashboard data');
        }
        const data = await response.json();
        setMlData(data);
      } catch (error) {
        console.error(error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-dark-200 border border-dark-300 rounded-lg p-4 shadow-lg">
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
          <div className="bg-dark-200 rounded-xl p-6 border border-dark-300">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
              <ChartBarIcon className="w-5 h-5 mr-2 text-accent-blue" />
              Model Performance Metrics
            </h3>

            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={model_details}>
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
          <div className="bg-dark-200 rounded-xl p-6 border border-dark-300">
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
          <div className="bg-dark-200 rounded-xl p-6 border border-dark-300">
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
            <div className="bg-dark-200 rounded-xl p-6 border border-dark-300">
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
                      <span className={`text-xs px-2 py-1 rounded ${
                        item.status === 'Complete' ? 'bg-trading-profit/20 text-trading-profit' :
                        item.status === 'Training' ? 'bg-accent-blue/20 text-accent-blue' :
                        'bg-dark-400/20 text-dark-500'
                      }`}>
                        {item.status}
                      </span>
                    </div>

                    <div className="w-full bg-dark-300 rounded-full h-2">
                      <motion.div
                        className={`h-2 rounded-full ${
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
            <div className="bg-dark-200 rounded-xl p-6 border border-dark-300">
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
                    <div className="w-full bg-dark-300 rounded-full h-1.5">
                      <motion.div
                        className="h-1.5 bg-gradient-to-r from-accent-blue to-accent-purple rounded-full"
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
            <div className="bg-dark-200 rounded-xl p-6 border border-dark-300">
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
      </motion.div>
    </Layout>
  )
}