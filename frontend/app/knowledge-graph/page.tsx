'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  ShareIcon,
  MagnifyingGlassIcon,
  FunnelIcon,
  ArrowsPointingOutIcon,
  PlayIcon,
  PauseIcon,
  ArrowDownTrayIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline'

import KnowledgeGraphNetwork from '../../components/KnowledgeGraph/KnowledgeGraphNetwork'
import EntityDetailsPanel from '../../components/KnowledgeGraph/EntityDetailsPanel'
import CascadeAnalysisPanel from '../../components/KnowledgeGraph/CascadeAnalysisPanel'
import GraphFiltersPanel from '../../components/KnowledgeGraph/GraphFiltersPanel'
import GraphStatsPanel from '../../components/KnowledgeGraph/GraphStatsPanel'
import { useSocket } from '../../hooks/useSocket'
import { knowledgeGraphAPI } from '../../lib/api'
import toast from 'react-hot-toast'

interface GraphStats {
  nodes: number
  edges: number
  density: number
  is_connected: boolean
  entities_by_type: Record<string, number>
  relationships_by_type: Record<string, number>
  top_entities: Array<[string, number]>
}

interface Entity {
  id: string
  name: string
  type: string
  region: string
  importance: number
  metadata: Record<string, any>
}

interface EntityDetails extends Entity {
  relations: {
    outgoing: Array<{
      target: string
      target_name: string
      type: string
      strength: number
    }>
    incoming: Array<{
      source: string
      source_name: string
      type: string
      strength: number
    }>
    total_connections: number
  }
  centrality_metrics: {
    degree_centrality: number
    betweenness_centrality: number
    closeness_centrality?: number
  }
}

interface CascadeAnalysis {
  event: {
    type: string
    entity: string
    magnitude: number
  }
  total_effects: number
  effects_by_horizon: {
    immediate: number
    short_term: number
    medium_term: number
    long_term: number
  }
  top_impacts: Array<{
    affected_entity: string
    impact_magnitude: number
    confidence: number
    time_horizon: string
    explanation: string
    propagation_path: string[]
  }>
  analysis_timestamp: string
}

export default function KnowledgeGraphPage() {
  const [isLoading, setIsLoading] = useState(true)
  const [networkRef, setNetworkRef] = useState<any>(null)
  const [selectedEntity, setSelectedEntity] = useState<EntityDetails | null>(null)
  const [stats, setStats] = useState<GraphStats | null>(null)
  const [isPhysicsEnabled, setIsPhysicsEnabled] = useState(false)
  const [showFilters, setShowFilters] = useState(true)
  const [showStats, setShowStats] = useState(true)
  const [showCascadeAnalysis, setShowCascadeAnalysis] = useState(false)
  const [cascadeResults, setCascadeResults] = useState<CascadeAnalysis | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const { socket, isConnected } = useSocket()

  useEffect(() => {
    // Subscribe to knowledge graph updates
    if (socket && isConnected) {
      socket.emit('kg_subscribe')

      socket.on('kg_update', handleGraphUpdate)
      socket.on('kg_analysis_result', handleAnalysisResult)
      socket.on('kg_analysis_error', handleAnalysisError)

      return () => {
        socket.off('kg_update', handleGraphUpdate)
        socket.off('kg_analysis_result', handleAnalysisResult)
        socket.off('kg_analysis_error', handleAnalysisError)
        socket.emit('kg_unsubscribe')
      }
    }
  }, [socket, isConnected])

  useEffect(() => {
    loadGraphStats()
  }, [])

  const handleGraphUpdate = (data: any) => {
    console.log('Graph update received:', data)
    if (data.reload_required && networkRef) {
      networkRef.loadData()
    }
  }

  const handleAnalysisResult = (data: CascadeAnalysis) => {
    console.log('Analysis result received:', data)
    setCascadeResults(data)
    setIsAnalyzing(false)
    setShowCascadeAnalysis(true)
    toast.success(`Analyse terminée: ${data.total_effects} effets détectés`)
  }

  const handleAnalysisError = (error: { error: string }) => {
    console.error('Analysis error:', error)
    setIsAnalyzing(false)
    toast.error(`Erreur d'analyse: ${error.error}`)
  }

  const loadGraphStats = async () => {
    try {
      const response = await knowledgeGraphAPI.getStatus()
      if (response.data.status === 'active' && response.data.statistics) {
        setStats(response.data.statistics)
      }
    } catch (error) {
      console.error('Error loading stats:', error)
    }
  }

  const handleEntitySelect = async (entityId: string) => {
    if (!entityId) {
      setSelectedEntity(null)
      return
    }

    try {
      const response = await knowledgeGraphAPI.getEntityDetails(entityId)
      setSelectedEntity(response.data)
    } catch (error) {
      console.error('Error loading entity details:', error)
      toast.error('Erreur lors du chargement des détails')
    }
  }

  const handleCascadeAnalysis = async (analysisData: {
    type: string
    entity: string
    magnitude: number
  }) => {
    setIsAnalyzing(true)

    try {
      const response = await knowledgeGraphAPI.analyzeCascade(analysisData)
      setCascadeResults(response.data)
      setShowCascadeAnalysis(true)
      toast.success(`Analyse terminée: ${response.data.total_effects} effets détectés`)
    } catch (error: any) {
      console.error('Error analyzing cascade:', error)
      toast.error(`Erreur d'analyse: ${error.response?.data?.error || error.message}`)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const exportToPNG = () => {
    if (networkRef) {
      networkRef.exportToPNG()
    }
  }

  const togglePhysics = () => {
    setIsPhysicsEnabled(!isPhysicsEnabled)
    if (networkRef) {
      networkRef.togglePhysics(!isPhysicsEnabled)
    }
  }

  const fitNetwork = () => {
    if (networkRef) {
      networkRef.fitNetwork()
    }
  }

  return (
    <div className="min-h-screen bg-dark-100">
      {/* Header */}
      <div className="border-b border-dark-300 bg-dark-200">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-accent-blue to-accent-purple rounded-lg flex items-center justify-center">
                <ShareIcon className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Knowledge Graph</h1>
                <p className="text-dark-400">Relations économiques et effets de cascade</p>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              {/* Connection Status */}
              <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
                isConnected ? 'bg-trading-profit/20 text-trading-profit' : 'bg-trading-loss/20 text-trading-loss'
              }`}>
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-trading-profit' : 'bg-trading-loss'}`} />
                <span>{isConnected ? 'Connecté' : 'Déconnecté'}</span>
              </div>

              {/* Controls */}
              <div className="flex items-center space-x-1">
                <button
                  onClick={() => setShowFilters(!showFilters)}
                  className={`p-2 rounded-lg transition-colors ${
                    showFilters ? 'bg-accent-blue text-white' : 'bg-dark-300 text-dark-400 hover:text-white'
                  }`}
                  title="Filtres"
                >
                  <FunnelIcon className="w-5 h-5" />
                </button>

                <button
                  onClick={() => setShowStats(!showStats)}
                  className={`p-2 rounded-lg transition-colors ${
                    showStats ? 'bg-accent-blue text-white' : 'bg-dark-300 text-dark-400 hover:text-white'
                  }`}
                  title="Statistiques"
                >
                  <ChartBarIcon className="w-5 h-5" />
                </button>

                <button
                  onClick={togglePhysics}
                  className="p-2 rounded-lg bg-dark-300 text-dark-400 hover:text-white transition-colors"
                  title={isPhysicsEnabled ? 'Désactiver physique' : 'Activer physique'}
                >
                  {isPhysicsEnabled ? <PauseIcon className="w-5 h-5" /> : <PlayIcon className="w-5 h-5" />}
                </button>

                <button
                  onClick={fitNetwork}
                  className="p-2 rounded-lg bg-dark-300 text-dark-400 hover:text-white transition-colors"
                  title="Ajuster la vue"
                >
                  <ArrowsPointingOutIcon className="w-5 h-5" />
                </button>

                <button
                  onClick={exportToPNG}
                  className="p-2 rounded-lg bg-dark-300 text-dark-400 hover:text-white transition-colors"
                  title="Exporter en PNG"
                >
                  <ArrowDownTrayIcon className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="flex h-[calc(100vh-73px)]">
        {/* Left Sidebar */}
        <motion.div
          initial={{ width: showFilters ? 320 : 0 }}
          animate={{ width: showFilters ? 320 : 0 }}
          transition={{ duration: 0.3 }}
          className="border-r border-dark-300 bg-dark-200 overflow-hidden"
        >
          <div className="h-full overflow-y-auto">
            {showFilters && (
              <div className="p-4 space-y-6">
                {/* Statistics Panel */}
                {showStats && stats && (
                  <GraphStatsPanel stats={stats} />
                )}

                {/* Filters Panel */}
                <GraphFiltersPanel
                  onFiltersChange={(filters) => {
                    if (networkRef) {
                      networkRef.applyFilters(filters)
                    }
                  }}
                />

                {/* Cascade Analysis Panel */}
                <CascadeAnalysisPanel
                  onAnalyze={handleCascadeAnalysis}
                  isAnalyzing={isAnalyzing}
                />
              </div>
            )}
          </div>
        </motion.div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          {/* Network Visualization */}
          <div className="flex-1 relative">
            <KnowledgeGraphNetwork
              ref={setNetworkRef}
              onEntitySelect={handleEntitySelect}
              physicsEnabled={isPhysicsEnabled}
              onLoading={setIsLoading}
            />

            {/* Loading Overlay */}
            {isLoading && (
              <div className="absolute inset-0 bg-dark-100/80 flex items-center justify-center z-10">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent-blue mb-4"></div>
                  <p className="text-dark-400">Chargement du graphe de connaissance...</p>
                </div>
              </div>
            )}
          </div>

          {/* Entity Details Panel */}
          {selectedEntity && (
            <motion.div
              initial={{ height: 0 }}
              animate={{ height: 'auto' }}
              exit={{ height: 0 }}
              className="border-t border-dark-300 bg-dark-200"
            >
              <EntityDetailsPanel
                entity={selectedEntity}
                onClose={() => setSelectedEntity(null)}
              />
            </motion.div>
          )}
        </div>

        {/* Right Sidebar - Cascade Analysis Results */}
        <motion.div
          initial={{ width: showCascadeAnalysis ? 400 : 0 }}
          animate={{ width: showCascadeAnalysis ? 400 : 0 }}
          transition={{ duration: 0.3 }}
          className="border-l border-dark-300 bg-dark-200 overflow-hidden"
        >
          {showCascadeAnalysis && cascadeResults && (
            <div className="h-full flex flex-col">
              <div className="p-4 border-b border-dark-300 flex items-center justify-between">
                <h3 className="text-lg font-semibold text-white">Analyse des Effets</h3>
                <button
                  onClick={() => setShowCascadeAnalysis(false)}
                  className="p-1 rounded hover:bg-dark-300 text-dark-400 hover:text-white"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-4">
                {/* Event Info */}
                <div className="bg-dark-300 rounded-lg p-4 mb-4">
                  <h4 className="font-semibold text-white mb-2">Événement analysé</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-dark-400">Type:</span>
                      <span className="text-white capitalize">{cascadeResults.event.type.replace('_', ' ')}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-dark-400">Entité:</span>
                      <span className="text-white">{cascadeResults.event.entity}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-dark-400">Magnitude:</span>
                      <span className={`font-medium ${
                        cascadeResults.event.magnitude > 0 ? 'text-trading-profit' : 'text-trading-loss'
                      }`}>
                        {cascadeResults.event.magnitude > 0 ? '+' : ''}{cascadeResults.event.magnitude.toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Summary Stats */}
                <div className="grid grid-cols-2 gap-2 mb-4">
                  <div className="bg-dark-300 rounded-lg p-3 text-center">
                    <div className="text-2xl font-bold text-accent-blue">{cascadeResults.total_effects}</div>
                    <div className="text-xs text-dark-400">Effets totaux</div>
                  </div>
                  <div className="bg-dark-300 rounded-lg p-3 text-center">
                    <div className="text-2xl font-bold text-orange-400">{cascadeResults.effects_by_horizon.immediate}</div>
                    <div className="text-xs text-dark-400">Immédiats</div>
                  </div>
                </div>

                {/* Top Impacts */}
                <div className="space-y-3">
                  <h4 className="font-semibold text-white">Impacts significatifs</h4>
                  {cascadeResults.top_impacts.map((impact, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className={`bg-dark-300 rounded-lg p-3 border-l-4 ${
                        impact.impact_magnitude > 0
                          ? 'border-trading-profit'
                          : impact.impact_magnitude < 0
                            ? 'border-trading-loss'
                            : 'border-yellow-400'
                      }`}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <span className="font-medium text-white">{impact.affected_entity}</span>
                        <div className="text-right">
                          <div className={`font-bold ${
                            impact.impact_magnitude > 0 ? 'text-trading-profit' : 'text-trading-loss'
                          }`}>
                            {impact.impact_magnitude > 0 ? '+' : ''}{impact.impact_magnitude.toFixed(3)}
                          </div>
                          <div className="text-xs text-dark-400">
                            Conf: {(impact.confidence * 100).toFixed(0)}%
                          </div>
                        </div>
                      </div>

                      <div className="text-sm text-dark-400 mb-2">
                        {impact.explanation}
                      </div>

                      <div className="flex justify-between items-center text-xs">
                        <span className={`px-2 py-1 rounded ${
                          impact.time_horizon === 'immediate' ? 'bg-red-900 text-red-200' :
                          impact.time_horizon === 'short_term' ? 'bg-orange-900 text-orange-200' :
                          impact.time_horizon === 'medium_term' ? 'bg-yellow-900 text-yellow-200' :
                          'bg-blue-900 text-blue-200'
                        }`}>
                          {impact.time_horizon.replace('_', ' ')}
                        </span>
                        <span className="text-dark-400 truncate ml-2">
                          {impact.propagation_path.join(' → ')}
                        </span>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </motion.div>
      </div>
    </div>
  )
}